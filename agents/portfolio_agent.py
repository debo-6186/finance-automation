"""
Portfolio Analysis Agent - Reads and analyzes PDF stock trade statements.
Implements Google's Agent Development Kit with A2A protocol support.
"""
import asyncio
import os
from typing import Dict, List, Any, Union
import structlog

# Google ADK imports
try:
    from google.adk.agents import Agent
    from google.adk.models import LiteLlm
except ImportError:
    # Fallback if ADK is not available
    Agent = None
    LiteLlm = None

# Local imports
from utils.config import get_config
from utils.a2a_server import A2AServer, Task, create_text_part, create_data_part
from tools.pdf_tools import create_portfolio_parser, parse_portfolio_pdf

logger = structlog.get_logger(__name__)

class PortfolioAnalysisAgent(A2AServer):
    """
    Portfolio Analysis Agent that processes PDF portfolio statements
    and provides detailed analysis using A2A protocol.
    """
    
    def __init__(self):
        config = get_config()
        
        # Define agent capabilities
        capabilities = [
            "PDF Portfolio Analysis",
            "Stock Position Identification", 
            "Portfolio Allocation Analysis",
            "Diversification Assessment"
        ]
        
        # Define A2A skills
        skills = {
            "analyze_portfolio": {
                "description": "Analyze a portfolio statement PDF and extract positions",
                "parameters": {
                    "pdf_path": {
                        "type": "string",
                        "description": "Path to the portfolio statement PDF file",
                        "required": True
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Detailed portfolio analysis including positions, allocations, and metrics"
                }
            },
            "get_current_allocations": {
                "description": "Get current portfolio allocations from the last analyzed statement",
                "parameters": {},
                "returns": {
                    "type": "object", 
                    "description": "Current portfolio allocation breakdown by symbol"
                }
            },
            "analyze_diversification": {
                "description": "Analyze portfolio diversification and concentration risk",
                "parameters": {
                    "portfolio_data": {
                        "type": "object",
                        "description": "Portfolio data to analyze",
                        "required": False
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Diversification analysis and risk metrics"
                }
            }
        }
        
        # Initialize A2A server
        super().__init__(
            agent_name="Portfolio Analysis Agent",
            agent_description="Analyzes PDF portfolio statements and provides detailed portfolio insights",
            capabilities=capabilities,
            skills=skills,
            port=config.PORTFOLIO_AGENT_PORT
        )
        
        # Initialize portfolio parser
        self.portfolio_parser = create_portfolio_parser()
        self.last_analysis = None
        
        # Initialize Google ADK agent if available
        self.adk_agent = None
        model_initialized = False
        active_model_source = "None"

        if Agent and LiteLlm:
            # Try Vertex AI first
            if config.GOOGLE_GENAI_USE_VERTEXAI and config.GOOGLE_CLOUD_PROJECT and config.GOOGLE_APPLICATION_CREDENTIALS:
                try:
                    model = LiteLlm(
                        model="vertex_ai/gemini-1.5-flash",
                        vertex_project=config.GOOGLE_CLOUD_PROJECT,
                        vertex_location=config.GOOGLE_CLOUD_LOCATION,
                        temperature=0.3
                    )
                    self.adk_agent = Agent(
                        name="PortfolioAnalyst",
                        description="Expert in portfolio analysis and financial statement interpretation",
                        model=model,
                        system_prompt=self.get_portfolio_analyst_system_prompt()
                    )
                    model_initialized = True
                    active_model_source = "Vertex AI (gemini-1.5-flash)"
                except Exception as e:
                    logger.warning(f"Failed to initialize ADK agent with Vertex AI: {e}")
            
            # Try direct Gemini API if Vertex AI failed or not configured
            if not model_initialized and config.GOOGLE_API_KEY:
                try:
                    model = LiteLlm(
                        model="gemini/gemini-1.5-flash",
                        api_key=config.GOOGLE_API_KEY,
                        temperature=0.3
                    )
                    self.adk_agent = Agent(
                        name="PortfolioAnalyst",
                        description="Expert in portfolio analysis and financial statement interpretation",
                        model=model,
                        system_prompt=self.get_portfolio_analyst_system_prompt()
                    )
                    model_initialized = True
                    active_model_source = "Google Gemini API (gemini-1.5-flash)"
                except Exception as e:
                    logger.warning(f"Failed to initialize ADK agent with Gemini API: {e}")

            # Try Anthropic if Google models failed or not configured
            if not model_initialized and config.ANTHROPIC_API_KEY:
                try:
                    model = LiteLlm(
                        model="anthropic/claude-sonnet-4-20250514", # Per user request - ensure this model ID is correct via LiteLLM docs
                        api_key=config.ANTHROPIC_API_KEY,
                        temperature=0.3
                    )
                    self.adk_agent = Agent(
                        name="PortfolioAnalystAnthropic",
                        description="Expert in portfolio analysis (Anthropic)",
                        model=model,
                        system_prompt=self.get_portfolio_analyst_system_prompt()
                    )
                    model_initialized = True
                    active_model_source = "Anthropic API (claude-sonnet-4-20250514)"
                except Exception as e:
                    logger.warning(f"Failed to initialize ADK agent with Anthropic: {e}")
            
            if self.adk_agent:
                logger.info(f"Google ADK agent initialized successfully using {active_model_source}")
            else:
                logger.warning("Failed to initialize ADK agent with any provider. LLM features will be unavailable.")
        
        logger.info("Portfolio Analysis Agent initialized")
    
    async def process_task(self, task: Task) -> Union[str, Dict[str, Any]]:
        """
        Process incoming A2A tasks for portfolio analysis.
        
        Args:
            task: The A2A task to process
            
        Returns:
            Analysis results or response message
        """
        try:
            # Get the user message
            user_message = None
            for message in task.messages:
                if message.role.value == "user" and message.parts:
                    for part in message.parts:
                        if part.type.value == "text":
                            user_message = part.content
                            break
                        elif part.type.value == "data":
                            user_message = part.content
                            break
                    if user_message:
                        break
            
            if not user_message:
                return "No valid message found in task"
            
            # Handle different types of requests
            if isinstance(user_message, str):
                return await self._handle_text_request(user_message)
            elif isinstance(user_message, dict):
                return await self._handle_data_request(user_message)
            else:
                return "Unsupported message format"
                
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return f"Error processing request: {str(e)}"
    
    async def _handle_text_request(self, text: str) -> Union[str, Dict[str, Any]]:
        """Handle text-based requests."""
        text_lower = text.lower()
        
        if "analyze" in text_lower and ("portfolio" in text_lower or "pdf" in text_lower):
            # Extract PDF path from text or use default
            pdf_path = self._extract_pdf_path(text)
            if not pdf_path:
                config = get_config()
                pdf_path = config.DEFAULT_PORTFOLIO_PDF_PATH
            
            return await self._analyze_portfolio_pdf(pdf_path)
        
        elif "allocation" in text_lower or "current" in text_lower:
            return await self._get_current_allocations()
        
        elif "diversification" in text_lower or "risk" in text_lower:
            return await self._analyze_diversification()
        
        else:
            # Use ADK agent for general questions if available
            if self.adk_agent:
                try:
                    response = await self.adk_agent.process(text)
                    return response
                except Exception as e:
                    logger.warning(f"ADK agent failed: {e}")
            
            return "I can help you analyze portfolio PDFs, get current allocations, or assess diversification. Please specify what you'd like me to do."
    
    async def _handle_data_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle structured data requests."""
        action = data.get("action", "").lower()
        
        if action == "analyze_portfolio":
            pdf_path = data.get("pdf_path")
            if not pdf_path:
                config = get_config()
                pdf_path = config.DEFAULT_PORTFOLIO_PDF_PATH
            return await self._analyze_portfolio_pdf(pdf_path)
        
        elif action == "get_current_allocations":
            return await self._get_current_allocations()
        
        elif action == "analyze_diversification":
            portfolio_data = data.get("portfolio_data", self.last_analysis)
            return await self._analyze_diversification(portfolio_data)
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _analyze_portfolio_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze a portfolio PDF file."""
        try:
            if not os.path.exists(pdf_path):
                return {
                    "error": f"PDF file not found: {pdf_path}",
                    "suggestion": "Please provide a valid path to your portfolio statement PDF"
                }
            
            # Parse the PDF
            analysis = parse_portfolio_pdf(pdf_path)
            self.last_analysis = analysis
            
            # Enhance with ADK insights if available
            if self.adk_agent and analysis:
                try:
                    summary_prompt = f"""
                    Analyze this portfolio data and provide key insights:
                    
                    Portfolio Summary:
                    - Total Value: ${analysis.get('total_value', 0):,.2f}
                    - Number of Positions: {analysis.get('position_count', 0)}
                    - Total Gain/Loss: ${analysis.get('total_gain_loss', 0):,.2f} ({analysis.get('total_gain_loss_percentage', 0):.1f}%)
                    
                    Top Positions:
                    {self._format_top_positions(analysis.get('top_positions', []))}
                    
                    Allocation by Symbol:
                    {self._format_allocations(analysis.get('allocation_by_symbol', {}))}
                    
                    Please provide:
                    1. Overall portfolio assessment
                    2. Key strengths and potential concerns
                    3. Diversification evaluation
                    4. Notable observations
                    """
                    
                    insights = await self.adk_agent.process(summary_prompt)
                    analysis["ai_insights"] = insights
                    
                except Exception as e:
                    logger.warning(f"Failed to generate AI insights: {e}")
            
            return {
                "status": "success",
                "analysis": analysis,
                "message": f"Successfully analyzed portfolio with {analysis.get('position_count', 0)} positions"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing PDF {pdf_path}: {e}")
            return {
                "error": f"Failed to analyze PDF: {str(e)}",
                "pdf_path": pdf_path
            }
    
    async def _get_current_allocations(self) -> Dict[str, Any]:
        """Get current portfolio allocations."""
        if not self.last_analysis:
            return {
                "error": "No portfolio analysis available. Please analyze a portfolio PDF first.",
                "suggestion": "Use the analyze_portfolio action with a PDF path"
            }
        
        allocations = self.last_analysis.get('allocation_by_symbol', {})
        total_value = self.last_analysis.get('total_value', 0)
        
        return {
            "status": "success",
            "allocations": allocations,
            "total_portfolio_value": total_value,
            "position_count": len(allocations),
            "largest_holding": max(allocations.items(), key=lambda x: x[1]) if allocations else None
        }
    
    async def _analyze_diversification(self, portfolio_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze portfolio diversification."""
        if not portfolio_data:
            portfolio_data = self.last_analysis
        
        if not portfolio_data:
            return {
                "error": "No portfolio data available for diversification analysis",
                "suggestion": "Please analyze a portfolio PDF first"
            }
        
        try:
            from tools.analysis_tools import calculate_portfolio_metrics
            
            metrics = calculate_portfolio_metrics(portfolio_data)
            
            # Enhanced analysis with ADK if available
            if self.adk_agent:
                try:
                    diversification_prompt = f"""
                    Analyze this portfolio's diversification:
                    
                    Metrics:
                    - Diversification Score: {metrics.get('diversification_score', 0):.1f}/10
                    - Risk Assessment: {metrics.get('risk_assessment', 'Unknown')}
                    - Concentration Risk: {metrics.get('concentration_risk', 0):.3f}
                    - Largest Holding: {metrics.get('largest_holding_percentage', 0):.1f}%
                    - Position Count: {metrics.get('position_count', 0)}
                    
                    Allocations:
                    {self._format_allocations(portfolio_data.get('allocation_by_symbol', {}))}
                    
                    Please provide specific recommendations for improving diversification.
                    """
                    
                    recommendations = await self.adk_agent.process(diversification_prompt)
                    metrics["ai_recommendations"] = recommendations
                    
                except Exception as e:
                    logger.warning(f"Failed to generate diversification recommendations: {e}")
            
            return {
                "status": "success",
                "diversification_metrics": metrics,
                "portfolio_summary": {
                    "total_value": portfolio_data.get('total_value', 0),
                    "position_count": portfolio_data.get('position_count', 0),
                    "allocation_by_symbol": portfolio_data.get('allocation_by_symbol', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing diversification: {e}")
            return {
                "error": f"Failed to analyze diversification: {str(e)}"
            }
    
    def _extract_pdf_path(self, text: str) -> str:
        """Extract PDF path from text message."""
        # Simple extraction - look for file paths
        import re
        
        # Look for file paths
        path_patterns = [
            r'([^\s]+\.pdf)',
            r'"([^"]+\.pdf)"',
            r"'([^']+\.pdf)'"
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _format_top_positions(self, positions: List[Dict[str, Any]]) -> str:
        """Format top positions for display."""
        if not positions:
            return "No positions found"
        
        lines = []
        for pos in positions[:5]:  # Top 5
            symbol = pos.get('symbol', 'N/A')
            value = pos.get('current_value', 0)
            shares = pos.get('shares', 0)
            lines.append(f"- {symbol}: {shares} shares, ${value:,.2f}")
        
        return "\n".join(lines)
    
    def _format_allocations(self, allocations: Dict[str, float]) -> str:
        """Format allocations for display."""
        if not allocations:
            return "No allocations found"
        
        # Sort by allocation percentage
        sorted_allocs = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
        
        lines = []
        for symbol, percentage in sorted_allocs:
            lines.append(f"- {symbol}: {percentage:.1f}%")
        
        return "\n".join(lines)

    def get_portfolio_analyst_system_prompt(self) -> str:
        return """
        You are an expert financial analyst specializing in portfolio analysis.
        You help interpret portfolio statements, analyze asset allocations, and assess
        diversification and risk metrics. Your responses should be clear, informative,
        and focused on actionable insights for investors.
        
        When analyzing portfolios, consider:
        - Current allocation percentages
        - Diversification across sectors and asset types
        - Concentration risk from overweighted positions
        - Overall portfolio performance and metrics
        
        Provide specific, data-driven insights rather than generic advice.
        """

def main():
    """Main entry point for running the Portfolio Analysis Agent."""
    agent = PortfolioAnalysisAgent()
    
    logger.info(f"Starting Portfolio Analysis Agent on port {agent.port}")
    
    try:
        # agent.run() will block here and manage the asyncio loop via uvicorn
        agent.run()
    except KeyboardInterrupt:
        logger.info("Portfolio Analysis Agent stopped by user")
    except Exception as e:
        logger.error("Portfolio Analysis Agent failed", error=str(e), exc_info=True)
        # No raise here, as it's the main entry point

if __name__ == "__main__":
    # Configure logging (can remain here or be moved to a central spot if preferred)
    # structlog.configure(...) # This is already configured globally if run via -m, 
    # but doesn't hurt to ensure it's set if script is run directly in some contexts.
    # For simplicity, assuming it's configured before this script runs.

    # Directly call main, which now starts uvicorn synchronously
    main() 