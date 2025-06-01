"""
PDF processing tools for stock trade statements.
Handles PDF parsing, data extraction, and portfolio analysis.
"""
import re
import io
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
import pandas as pd
import structlog

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    PyPDF2 = None
    pdfplumber = None

logger = structlog.get_logger(__name__)

class PortfolioPosition:
    """Represents a single stock position in the portfolio."""
    
    def __init__(self, symbol: str, name: str, shares: float, 
                 avg_cost: float, current_value: float, market_price: Optional[float] = None):
        self.symbol = symbol.upper()
        self.name = name
        self.shares = shares
        self.avg_cost = avg_cost
        self.current_value = current_value
        self.market_price = market_price
        
    @property
    def total_cost(self) -> float:
        """Calculate total cost basis."""
        return self.shares * self.avg_cost
    
    @property
    def gain_loss(self) -> float:
        """Calculate unrealized gain/loss."""
        return self.current_value - self.total_cost
    
    @property
    def gain_loss_percentage(self) -> float:
        """Calculate percentage gain/loss."""
        if self.total_cost == 0:
            return 0
        return (self.gain_loss / self.total_cost) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "shares": self.shares,
            "avg_cost": self.avg_cost,
            "current_value": self.current_value,
            "market_price": self.market_price,
            "total_cost": self.total_cost,
            "gain_loss": self.gain_loss,
            "gain_loss_percentage": self.gain_loss_percentage
        }

class PortfolioAnalysis:
    """Complete portfolio analysis results."""
    
    def __init__(self, positions: List[PortfolioPosition], 
                 total_value: float, total_cost: float, cash_balance: float = 0.0):
        self.positions = positions
        self.total_value = total_value
        self.total_cost = total_cost
        self.cash_balance = cash_balance
        
    @property
    def total_gain_loss(self) -> float:
        """Total portfolio gain/loss."""
        return self.total_value - self.total_cost
    
    @property
    def total_gain_loss_percentage(self) -> float:
        """Total portfolio gain/loss percentage."""
        if self.total_cost == 0:
            return 0
        return (self.total_gain_loss / self.total_cost) * 100
    
    @property
    def position_count(self) -> int:
        """Number of positions in portfolio."""
        return len(self.positions)
    
    def get_allocation_by_symbol(self) -> Dict[str, float]:
        """Get allocation percentage by symbol."""
        if self.total_value == 0:
            return {}
        
        return {
            position.symbol: (position.current_value / self.total_value) * 100
            for position in self.positions
        }
    
    def get_top_positions(self, limit: int = 10) -> List[PortfolioPosition]:
        """Get top positions by value."""
        return sorted(self.positions, key=lambda p: p.current_value, reverse=True)[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "positions": [pos.to_dict() for pos in self.positions],
            "total_value": self.total_value,
            "total_cost": self.total_cost,
            "cash_balance": self.cash_balance,
            "total_gain_loss": self.total_gain_loss,
            "total_gain_loss_percentage": self.total_gain_loss_percentage,
            "position_count": self.position_count,
            "allocation_by_symbol": self.get_allocation_by_symbol(),
            "top_positions": [pos.to_dict() for pos in self.get_top_positions(5)]
        }

class PDFPortfolioParser:
    """Parser for extracting portfolio information from PDF statements."""
    
    def __init__(self):
        if not pdfplumber or not PyPDF2:
            raise ImportError("pdfplumber and PyPDF2 are required for PDF processing")
    
    def parse_portfolio_pdf(self, pdf_path: str) -> PortfolioAnalysis:
        """
        Parse a portfolio statement PDF and extract positions.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PortfolioAnalysis object with extracted data
        """
        try:
            logger.info(f"Parsing portfolio PDF: {pdf_path}")
            
            # Try different parsing strategies
            text = self._extract_text_pdfplumber(pdf_path)
            if not text.strip():
                text = self._extract_text_pypdf2(pdf_path)
            
            if not text.strip():
                raise ValueError("Could not extract text from PDF")
            
            # Parse the extracted text
            positions = self._parse_positions_from_text(text)
            total_value, total_cost, cash_balance = self._parse_totals_from_text(text)
            
            analysis = PortfolioAnalysis(positions, total_value, total_cost, cash_balance)
            
            logger.info(f"Successfully parsed {len(positions)} positions from PDF")
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise
    
    def _extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return ""
    
    def _extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 as fallback."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def _parse_positions_from_text(self, text: str) -> List[PortfolioPosition]:
        """Parse stock positions from extracted text."""
        positions = []
        
        # Common patterns for different brokerage statements
        patterns = [
            # Pattern 1: Symbol, Name, Shares, Avg Cost, Current Value
            r'([A-Z]{1,5})\s+([A-Za-z\s&.,]+?)\s+(\d+(?:\.\d+)?)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)',
            
            # Pattern 2: More flexible pattern
            r'([A-Z]{2,5})\s+(.+?)\s+(\d+(?:\.\d+)?)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)',
            
            # Pattern 3: Table format with pipes or tabs
            r'([A-Z]{1,5})\s*[|\t]\s*([^|\t]+?)\s*[|\t]\s*(\d+(?:\.\d+)?)\s*[|\t]\s*\$?([\d,]+\.?\d*)\s*[|\t]\s*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                try:
                    symbol = match.group(1).strip()
                    name = match.group(2).strip()
                    shares = float(match.group(3))
                    avg_cost = float(match.group(4).replace(',', '').replace('$', ''))
                    current_value = float(match.group(5).replace(',', '').replace('$', ''))
                    
                    # Validate the data
                    if shares > 0 and avg_cost > 0 and current_value > 0:
                        position = PortfolioPosition(symbol, name, shares, avg_cost, current_value)
                        positions.append(position)
                        logger.debug(f"Parsed position: {symbol} - {shares} shares")
                        
                except (ValueError, IndexError) as e:
                    logger.debug(f"Skipped invalid position match: {e}")
                    continue
        
        # Remove duplicates (keep the one with highest value)
        unique_positions = {}
        for position in positions:
            if position.symbol not in unique_positions or position.current_value > unique_positions[position.symbol].current_value:
                unique_positions[position.symbol] = position
        
        return list(unique_positions.values())
    
    def _parse_totals_from_text(self, text: str) -> Tuple[float, float, float]:
        """Parse total values from the text."""
        total_value = 0.0
        total_cost = 0.0
        cash_balance = 0.0
        
        # Patterns for finding totals
        total_patterns = [
            r'Total\s+(?:Value|Market\s+Value):\s*\$?([\d,]+\.?\d*)',
            r'Portfolio\s+Value:\s*\$?([\d,]+\.?\d*)',
            r'Total\s+Account\s+Value:\s*\$?([\d,]+\.?\d*)',
        ]
        
        cost_patterns = [
            r'Total\s+Cost:\s*\$?([\d,]+\.?\d*)',
            r'Cost\s+Basis:\s*\$?([\d,]+\.?\d*)',
        ]
        
        cash_patterns = [
            r'Cash\s+Balance:\s*\$?([\d,]+\.?\d*)',
            r'Available\s+Cash:\s*\$?([\d,]+\.?\d*)',
            r'Settled\s+Cash:\s*\$?([\d,]+\.?\d*)',
        ]
        
        # Find total value
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    total_value = float(match.group(1).replace(',', ''))
                    break
                except ValueError:
                    continue
        
        # Find total cost
        for pattern in cost_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    total_cost = float(match.group(1).replace(',', ''))
                    break
                except ValueError:
                    continue
        
        # Find cash balance
        for pattern in cash_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    cash_balance = float(match.group(1).replace(',', ''))
                    break
                except ValueError:
                    continue
        
        return total_value, total_cost, cash_balance

# Factory function for creating the parser
def create_portfolio_parser() -> PDFPortfolioParser:
    """Create a portfolio parser instance."""
    return PDFPortfolioParser()

# Helper function for quick parsing
def parse_portfolio_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Quick function to parse a portfolio PDF and return results as dict.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with portfolio analysis results
    """
    parser = create_portfolio_parser()
    analysis = parser.parse_portfolio_pdf(pdf_path)
    return analysis.to_dict() 