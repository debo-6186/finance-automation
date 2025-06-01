"""
Financial analysis tools for portfolio optimization and investment advice.
Provides algorithms for portfolio rebalancing and investment recommendations.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class AllocationRecommendation:
    """Represents an allocation recommendation for a stock."""
    symbol: str
    current_allocation: float  # Current percentage
    target_allocation: float   # Recommended percentage
    amount_to_invest: float    # Dollar amount to invest
    reasoning: str            # Explanation for the recommendation

@dataclass
class PortfolioAdvice:
    """Complete portfolio investment advice."""
    monthly_allocation: float
    recommendations: List[AllocationRecommendation]
    rebalancing_needed: bool
    risk_assessment: str
    diversification_score: float
    total_recommended_investment: float
    cash_allocation: float
    summary: str

class RiskCalculator:
    """Calculate portfolio risk metrics."""
    
    @staticmethod
    def calculate_concentration_risk(allocations: Dict[str, float]) -> float:
        """
        Calculate concentration risk using Herfindahl-Hirschman Index.
        Returns a score from 0 (perfectly diversified) to 1 (highly concentrated).
        """
        if not allocations:
            return 0.0
        
        # Convert percentages to decimals
        weights = [alloc / 100.0 for alloc in allocations.values()]
        
        # Calculate HHI
        hhi = sum(w ** 2 for w in weights)
        
        # Normalize: HHI ranges from 1/n (perfectly diversified) to 1 (single asset)
        n = len(weights)
        if n <= 1:
            return 1.0
        
        min_hhi = 1.0 / n
        normalized_score = (hhi - min_hhi) / (1.0 - min_hhi)
        
        return max(0.0, min(1.0, normalized_score))
    
    @staticmethod
    def assess_diversification(allocations: Dict[str, float], position_count: int) -> Tuple[float, str]:
        """
        Assess portfolio diversification.
        Returns a score from 0-10 and a descriptive assessment.
        """
        if not allocations or position_count == 0:
            return 0.0, "No positions"
        
        concentration_risk = RiskCalculator.calculate_concentration_risk(allocations)
        
        # Score factors
        position_score = min(10, position_count * 2)  # More positions = better (up to 5 positions = 10 points)
        concentration_score = (1 - concentration_risk) * 10  # Lower concentration = better
        
        # Top holding concentration penalty
        max_holding = max(allocations.values()) if allocations else 0
        if max_holding > 50:
            concentration_penalty = (max_holding - 50) / 10  # Penalty for >50% in single stock
        else:
            concentration_penalty = 0
        
        final_score = max(0, min(10, (position_score + concentration_score) / 2 - concentration_penalty))
        
        # Descriptive assessment
        if final_score >= 8:
            assessment = "Well diversified"
        elif final_score >= 6:
            assessment = "Moderately diversified"
        elif final_score >= 4:
            assessment = "Somewhat concentrated"
        else:
            assessment = "Highly concentrated - consider diversification"
        
        return final_score, assessment

class AllocationStrategy:
    """Different allocation strategies for portfolio optimization."""
    
    @staticmethod
    def equal_weight_strategy(symbols: List[str], monthly_amount: float) -> Dict[str, float]:
        """Equal weight allocation across all positions."""
        if not symbols:
            return {}
        
        per_symbol = monthly_amount / len(symbols)
        return {symbol: per_symbol for symbol in symbols}
    
    @staticmethod
    def market_cap_weighted_strategy(symbols: List[str], monthly_amount: float, 
                                   market_caps: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Market cap weighted allocation (simplified without real market cap data)."""
        if not symbols:
            return {}
        
        # If no market cap data, fall back to equal weight
        if not market_caps:
            return AllocationStrategy.equal_weight_strategy(symbols, monthly_amount)
        
        total_market_cap = sum(market_caps.get(symbol, 1.0) for symbol in symbols)
        if total_market_cap == 0:
            return AllocationStrategy.equal_weight_strategy(symbols, monthly_amount)
        
        allocations = {}
        for symbol in symbols:
            weight = market_caps.get(symbol, 1.0) / total_market_cap
            allocations[symbol] = monthly_amount * weight
        
        return allocations
    
    @staticmethod
    def balanced_growth_strategy(symbols: List[str], monthly_amount: float,
                               current_allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Balanced growth strategy that considers current allocations.
        Tends to invest more in underweighted positions.
        """
        if not symbols:
            return {}
        
        if not current_allocations:
            return AllocationStrategy.equal_weight_strategy(symbols, monthly_amount)
        
        # Calculate target equal weight
        target_weight = 100.0 / len(symbols)
        
        # Calculate how much each position is under/over weighted
        weight_differences = {}
        total_underweight = 0
        
        for symbol in symbols:
            current_weight = current_allocations.get(symbol, 0)
            difference = target_weight - current_weight
            weight_differences[symbol] = difference
            if difference > 0:
                total_underweight += difference
        
        # Allocate more to underweighted positions
        allocations = {}
        if total_underweight > 0:
            for symbol in symbols:
                if weight_differences[symbol] > 0:
                    # Allocate proportionally based on how underweighted
                    proportion = weight_differences[symbol] / total_underweight
                    allocations[symbol] = monthly_amount * proportion
                else:
                    # Small allocation to maintain position
                    allocations[symbol] = monthly_amount * 0.1 / len(symbols)
        else:
            # All positions are at or above target, use equal weight
            allocations = AllocationStrategy.equal_weight_strategy(symbols, monthly_amount)
        
        return allocations

class InvestmentAdvisor:
    """Main class for generating investment advice."""
    
    def __init__(self, risk_tolerance: str = "moderate"):
        self.risk_tolerance = risk_tolerance.lower()
        self.risk_calculator = RiskCalculator()
    
    def generate_advice(self, portfolio_data: Dict[str, Any], market_data: Dict[str, Any], 
                       monthly_allocation: float) -> PortfolioAdvice:
        """
        Generate comprehensive investment advice.
        
        Args:
            portfolio_data: Current portfolio information
            market_data: Current market prices for portfolio stocks
            monthly_allocation: Monthly investment amount
            
        Returns:
            PortfolioAdvice with recommendations
        """
        try:
            # Extract current portfolio information
            current_positions = portfolio_data.get('positions', [])
            current_allocations = portfolio_data.get('allocation_by_symbol', {})
            total_portfolio_value = portfolio_data.get('total_value', 0)
            cash_balance = portfolio_data.get('cash_balance', 0)
            
            # Get list of current symbols
            symbols = [pos['symbol'] for pos in current_positions if pos.get('symbol')]
            
            # Calculate diversification metrics
            diversification_score, risk_assessment = self.risk_calculator.assess_diversification(
                current_allocations, len(current_positions)
            )
            
            # Determine allocation strategy based on risk tolerance and diversification
            if diversification_score < 5:
                strategy = "rebalancing"
                allocations = AllocationStrategy.balanced_growth_strategy(
                    symbols, monthly_allocation, current_allocations
                )
            elif self.risk_tolerance == "conservative":
                strategy = "equal_weight"
                allocations = AllocationStrategy.equal_weight_strategy(symbols, monthly_allocation)
            else:
                strategy = "balanced_growth"
                allocations = AllocationStrategy.balanced_growth_strategy(
                    symbols, monthly_allocation, current_allocations
                )
            
            # Generate recommendations
            recommendations = []
            total_recommended = 0
            
            for symbol, amount in allocations.items():
                if amount > 0:
                    current_alloc = current_allocations.get(symbol, 0)
                    
                    # Calculate target allocation after investment
                    future_portfolio_value = total_portfolio_value + monthly_allocation
                    target_alloc = ((current_alloc / 100 * total_portfolio_value) + amount) / future_portfolio_value * 100
                    
                    # Generate reasoning
                    reasoning = self._generate_reasoning(symbol, current_alloc, target_alloc, 
                                                       market_data.get(symbol, {}), strategy)
                    
                    recommendation = AllocationRecommendation(
                        symbol=symbol,
                        current_allocation=current_alloc,
                        target_allocation=target_alloc,
                        amount_to_invest=amount,
                        reasoning=reasoning
                    )
                    recommendations.append(recommendation)
                    total_recommended += amount
            
            # Check if rebalancing is needed
            rebalancing_needed = diversification_score < 6 or any(
                alloc > 40 for alloc in current_allocations.values()
            )
            
            # Calculate cash allocation
            cash_allocation = monthly_allocation - total_recommended
            
            # Generate summary
            summary = self._generate_summary(
                recommendations, diversification_score, rebalancing_needed, 
                monthly_allocation, total_recommended
            )
            
            return PortfolioAdvice(
                monthly_allocation=monthly_allocation,
                recommendations=recommendations,
                rebalancing_needed=rebalancing_needed,
                risk_assessment=risk_assessment,
                diversification_score=diversification_score,
                total_recommended_investment=total_recommended,
                cash_allocation=cash_allocation,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error generating investment advice: {e}")
            raise
    
    def _generate_reasoning(self, symbol: str, current_alloc: float, target_alloc: float, 
                          market_data: Dict[str, Any], strategy: str) -> str:
        """Generate reasoning for an allocation recommendation."""
        price_info = f"Current price: ${market_data.get('price', 'N/A')}" if market_data else ""
        
        if strategy == "rebalancing":
            if current_alloc < 15:
                return f"Underweighted position - increasing allocation to improve diversification. {price_info}"
            elif current_alloc > 30:
                return f"Overweighted position - reducing relative allocation. {price_info}"
            else:
                return f"Maintaining balanced allocation. {price_info}"
        
        elif strategy == "equal_weight":
            return f"Equal weight strategy - targeting balanced allocation across all positions. {price_info}"
        
        else:  # balanced_growth
            if target_alloc > current_alloc:
                return f"Growth opportunity - increasing allocation based on portfolio balance. {price_info}"
            else:
                return f"Maintaining current allocation with modest investment. {price_info}"
    
    def _generate_summary(self, recommendations: List[AllocationRecommendation], 
                         diversification_score: float, rebalancing_needed: bool,
                         monthly_allocation: float, total_recommended: float) -> str:
        """Generate a summary of the investment advice."""
        summary_parts = []
        
        # Diversification assessment
        if diversification_score >= 8:
            summary_parts.append("Your portfolio shows excellent diversification.")
        elif diversification_score >= 6:
            summary_parts.append("Your portfolio is moderately well-diversified.")
        elif diversification_score >= 4:
            summary_parts.append("Your portfolio shows some concentration risk - consider diversification.")
        else:
            summary_parts.append("Your portfolio is highly concentrated - diversification is recommended.")
        
        # Investment strategy
        if rebalancing_needed:
            summary_parts.append("This month's allocation focuses on rebalancing your portfolio.")
        else:
            summary_parts.append("This month's allocation maintains your current balanced approach.")
        
        # Allocation summary
        if len(recommendations) > 0:
            top_allocation = max(recommendations, key=lambda r: r.amount_to_invest)
            summary_parts.append(
                f"Largest allocation recommended: ${top_allocation.amount_to_invest:.2f} to {top_allocation.symbol}."
            )
        
        # Cash remainder
        cash_remainder = monthly_allocation - total_recommended
        if cash_remainder > 50:
            summary_parts.append(f"Consider keeping ${cash_remainder:.2f} in cash for future opportunities.")
        
        return " ".join(summary_parts)

def create_investment_advisor(risk_tolerance: str = "moderate") -> InvestmentAdvisor:
    """Create an investment advisor instance."""
    return InvestmentAdvisor(risk_tolerance)

def calculate_portfolio_metrics(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate various portfolio metrics.
    
    Args:
        portfolio_data: Portfolio data dictionary
        
    Returns:
        Dictionary with calculated metrics
    """
    risk_calc = RiskCalculator()
    
    allocations = portfolio_data.get('allocation_by_symbol', {})
    position_count = portfolio_data.get('position_count', 0)
    
    concentration_risk = risk_calc.calculate_concentration_risk(allocations)
    diversification_score, risk_assessment = risk_calc.assess_diversification(allocations, position_count)
    
    return {
        'concentration_risk': concentration_risk,
        'diversification_score': diversification_score,
        'risk_assessment': risk_assessment,
        'largest_holding_percentage': max(allocations.values()) if allocations else 0,
        'position_count': position_count
    } 