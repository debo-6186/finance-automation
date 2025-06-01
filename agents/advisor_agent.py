"""
Investment Advisor Agent - Provides investment advice based on portfolio and market data.
Implements Google's Agent Development Kit with A2A protocol support.
"""
import asyncio
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
from tools.analysis_tools import create_investment_advisor, calculate_portfolio_metrics

logger = structlog.get_logger(__name__)

class InvestmentAdvisorAgent(A2AServer):
    """
    Investment Advisor Agent that provides personalized investment advice
    based on portfolio analysis and current market conditions using A2A protocol.
    """
    
    def __init__(self):
        config = get_config()
        
        # Define agent capabilities
        capabilities = [
            "Investment Advice Generation",
            "Portfolio Rebalancing Recommendations",
            "Risk Assessment and Analysis",
            "Monthly Allocation Planning"
        ]
        
        # Define A2A skills
        skills = {
            "generate_advice": {
                "description": "Generate comprehensive investment advice based on portfolio and market data",
                "parameters": {
                    "portfolio_data": {
                        "type": "object",
                        "description": "Current portfolio analysis data",
                        "required": True
                    },
                    "market_data": {
                        "type": "object", 
                        "description": "Current market prices for portfolio stocks",
                        "required": True
                    },
                    "monthly_allocation": {
                        "type": "number",
                        "description": "Monthly investment amount in USD",
                        "required": False
                    },
                    "risk_tolerance": {
                        "type": "string",
                        "description": "Risk tolerance: conservative, moderate, or aggressive",
                        "required": False
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Comprehensive investment advice with specific recommendations"
                }
            },
            "calculate_rebalancing": {
                "description": "Calculate portfolio rebalancing recommendations",
                "parameters": {
                    "portfolio_data": {
                        "type": "object",
                        "description": "Current portfolio data",
                        "required": True
                    },
                    "target_allocation": {
                        "type": "object",
                        "description": "Target allocation percentages by symbol",
                        "required": False
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Rebalancing recommendations and trade suggestions"
                }
            },
            "assess_risk": {
                "description": "Assess portfolio risk and diversification",
                "parameters": {
                    "portfolio_data": {
                        "type": "object",
                        "description": "Portfolio data for risk assessment",
                        "required": True
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Detailed risk assessment and recommendations"
                }
            },
            "monthly_allocation_plan": {
                "description": "Create a monthly investment allocation plan",
                "parameters": {
                    "portfolio_data": {
                        "type": "object",
                        "description": "Current portfolio data",
                        "required": True
                    },
                    "market_data": {
                        "type": "object",
                        "description": "Current market prices",
                        "required": True
                    },
                    "amount": {
                        "type": "number",
                        "description": "Monthly investment amount",
                        "required": True
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Monthly allocation plan with specific investment amounts"
                }
            }
        }
        
        # Initialize A2A server
        super().__init__(
            agent_name="Investment Advisor Agent",
            agent_description="Provides personalized investment advice and portfolio optimization recommendations",
            capabilities=capabilities,
            skills=skills,
            port=config.ADVISOR_AGENT_PORT
        )
        
        # Initialize investment advisor
        self.investment_advisor = create_investment_advisor("moderate")
        self.default_monthly_allocation = config.MONTHLY_ALLOCATION_USD
        
        # Initialize Google ADK agent if available
        self.adk_agent = None
        model_initialized = False
        active_model_source = "None"

        if Agent and LiteLlm:
            # Try Vertex AI first
            if config.GOOGLE_GENAI_USE_VERTEXAI and config.GOOGLE_CLOUD_PROJECT and config.GOOGLE_APPLICATION_CREDENTIALS:
                try:
                    model = LiteLlm(
                        model="vertex_ai/gemini-1.5-pro", # Use pro for advisor
                        vertex_project=config.GOOGLE_CLOUD_PROJECT,
                        vertex_location=config.GOOGLE_CLOUD_LOCATION,
                        temperature=0.4
                    )
                    self.adk_agent = Agent(
                        name="InvestmentAdvisor",
                        description="Expert investment advisor specializing in portfolio optimization and financial planning",
                        model=model,
                        system_prompt=self.get_advisor_system_prompt()
                    )
                    model_initialized = True
                    active_model_source = "Vertex AI (gemini-1.5-pro)"
                except Exception as e:
                    logger.warning(f"Failed to initialize ADK agent with Vertex AI: {e}")
            
            # Try direct Gemini API if Vertex AI failed or not configured
            if not model_initialized and config.GOOGLE_API_KEY:
                try:
                    model = LiteLlm(
                        model="gemini/gemini-1.5-pro",
                        api_key=config.GOOGLE_API_KEY,
                        temperature=0.4
                    )
                    self.adk_agent = Agent(
                        name="InvestmentAdvisor",
                        description="Expert investment advisor specializing in portfolio optimization and financial planning",
                        model=model,
                        system_prompt=self.get_advisor_system_prompt()
                    )
                    model_initialized = True
                    active_model_source = "Google Gemini API (gemini-1.5-pro)"
                except Exception as e:
                    logger.warning(f"Failed to initialize ADK agent with Gemini API: {e}")

            # Try Anthropic if Google models failed or not configured
            if not model_initialized and config.ANTHROPIC_API_KEY:
                try:
                    model = LiteLlm(
                        model="anthropic/claude-sonnet-4-20250514", # Per user request
                        api_key=config.ANTHROPIC_API_KEY,
                        temperature=0.4
                    )
                    self.adk_agent = Agent(
                        name="InvestmentAdvisorAnthropic",
                        description="Expert investment advisor (Anthropic)",
                        model=model,
                        system_prompt=self.get_advisor_system_prompt()
                    )
                    model_initialized = True
                    active_model_source = "Anthropic API (claude-sonnet-4-20250514)"
                except Exception as e:
                    logger.warning(f"Failed to initialize ADK agent with Anthropic: {e}")
            
            if self.adk_agent:
                logger.info(f"Google ADK agent initialized successfully using {active_model_source}")
            else:
                logger.warning("Failed to initialize ADK agent with any provider. LLM features will be unavailable.")
        
        logger.info("Investment Advisor Agent initialized")
    
    def get_advisor_system_prompt(self) -> str:
        return """
        You are a seasoned investment advisor with expertise in:
        - Portfolio optimization and asset allocation
        - Risk management and diversification strategies  
        - Market analysis and investment timing
        - Financial planning and wealth building
        
        Your advice should be:
        - Personalized based on individual portfolio and risk tolerance
        - Data-driven and backed by financial principles
        - Clear and actionable with specific recommendations
        - Focused on long-term wealth building
        - Conservative and risk-appropriate
        
        Always consider:
        - Current portfolio allocation and diversification
        - Market conditions and timing
        - Risk tolerance and investment goals
        - Dollar-cost averaging and systematic investing
        
        Provide specific, actionable advice with clear reasoning.
        Include both allocation recommendations and strategic insights.
        """
    
    async def process_task(self, task: Task) -> Union[str, Dict[str, Any]]:
        """
        Process incoming A2A tasks for investment advice requests.
        
        Args:
            task: The A2A task to process
            
        Returns:
            Investment advice results or response message
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
        
        # Use ADK agent for general investment questions
        if self.adk_agent:
            try:
                # Enhance the prompt with context
                enhanced_prompt = f"""
                Investment Question: {text}
                
                Default Monthly Allocation: ${self.default_monthly_allocation}
                
                Please provide investment advice. If you need specific portfolio or market data 
                to give detailed recommendations, please let me know what information is required.
                
                Focus on actionable advice and specific recommendations when possible.
                """
                
                response = await self.adk_agent.process(enhanced_prompt)
                return response
            except Exception as e:
                logger.warning(f"ADK agent failed: {e}")
        
        return "I can provide investment advice based on your portfolio and market data. Please provide your portfolio analysis and current market prices, or ask specific investment questions."
    
    async def _handle_data_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle structured data requests."""
        action = data.get("action", "").lower()
        
        if action == "generate_advice":
            portfolio_data = data.get("portfolio_data")
            market_data = data.get("market_data") 
            monthly_allocation = data.get("monthly_allocation", self.default_monthly_allocation)
            risk_tolerance = data.get("risk_tolerance", "moderate")
            
            if not portfolio_data or not market_data:
                return {"error": "Both portfolio_data and market_data are required for advice generation"}
            
            return await self._generate_comprehensive_advice(
                portfolio_data, market_data, monthly_allocation, risk_tolerance
            )
        
        elif action == "calculate_rebalancing":
            portfolio_data = data.get("portfolio_data")
            target_allocation = data.get("target_allocation")
            
            if not portfolio_data:
                return {"error": "Portfolio data is required for rebalancing calculations"}
            
            return await self._calculate_rebalancing(portfolio_data, target_allocation)
        
        elif action == "assess_risk":
            portfolio_data = data.get("portfolio_data")
            
            if not portfolio_data:
                return {"error": "Portfolio data is required for risk assessment"}
            
            return await self._assess_portfolio_risk(portfolio_data)
        
        elif action == "monthly_allocation_plan":
            portfolio_data = data.get("portfolio_data")
            market_data = data.get("market_data")
            amount = data.get("amount", self.default_monthly_allocation)
            
            if not portfolio_data or not market_data:
                return {"error": "Both portfolio_data and market_data are required"}
            
            return await self._create_monthly_allocation_plan(portfolio_data, market_data, amount)
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _generate_comprehensive_advice(self, portfolio_data: Dict[str, Any], 
                                          market_data: Dict[str, Any], 
                                          monthly_allocation: float,
                                          risk_tolerance: str) -> Dict[str, Any]:
        """Generate comprehensive investment advice."""
        try:
            # Update advisor risk tolerance
            self.investment_advisor.risk_tolerance = risk_tolerance.lower()
            
            # Generate advice using the investment advisor
            advice = self.investment_advisor.generate_advice(
                portfolio_data, market_data, monthly_allocation
            )
            
            # Convert advice to dictionary format
            advice_dict = {
                "monthly_allocation": advice.monthly_allocation,
                "recommendations": [
                    {
                        "symbol": rec.symbol,
                        "current_allocation": rec.current_allocation,
                        "target_allocation": rec.target_allocation,
                        "amount_to_invest": rec.amount_to_invest,
                        "reasoning": rec.reasoning
                    }
                    for rec in advice.recommendations
                ],
                "rebalancing_needed": advice.rebalancing_needed,
                "risk_assessment": advice.risk_assessment,
                "diversification_score": advice.diversification_score,
                "total_recommended_investment": advice.total_recommended_investment,
                "cash_allocation": advice.cash_allocation,
                "summary": advice.summary
            }
            
            # Enhance with ADK insights if available
            if self.adk_agent:
                try:
                    insight_prompt = f"""
                    Review and enhance this investment advice:
                    
                    Portfolio Summary:
                    - Total Value: ${portfolio_data.get('total_value', 0):,.2f}
                    - Diversification Score: {advice.diversification_score:.1f}/10
                    - Risk Assessment: {advice.risk_assessment}
                    - Rebalancing Needed: {advice.rebalancing_needed}
                    
                    Monthly Allocation: ${monthly_allocation:,.2f}
                    Risk Tolerance: {risk_tolerance}
                    
                    Current Recommendations:
                    {self._format_recommendations(advice.recommendations)}
                    
                    Market Context:
                    {self._format_market_data(market_data)}
                    
                    Please provide:
                    1. Strategic insights and market timing considerations
                    2. Risk management recommendations
                    3. Long-term portfolio building strategies
                    4. Any additional considerations or warnings
                    
                    Focus on actionable insights that complement the quantitative recommendations.
                    """
                    
                    strategic_insights = await self.adk_agent.process(insight_prompt)
                    advice_dict["strategic_insights"] = strategic_insights
                    
                except Exception as e:
                    logger.warning(f"Failed to generate strategic insights: {e}")
            
            return {
                "status": "success",
                "advice": advice_dict,
                "parameters": {
                    "monthly_allocation": monthly_allocation,
                    "risk_tolerance": risk_tolerance,
                    "portfolio_value": portfolio_data.get('total_value', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating advice: {e}")
            return {
                "error": f"Failed to generate investment advice: {str(e)}"
            }
    
    async def _calculate_rebalancing(self, portfolio_data: Dict[str, Any], 
                                   target_allocation: Dict[str, float] = None) -> Dict[str, Any]:
        """Calculate portfolio rebalancing recommendations."""
        try:
            current_allocations = portfolio_data.get('allocation_by_symbol', {})
            total_value = portfolio_data.get('total_value', 0)
            
            if not current_allocations:
                return {"error": "No current allocations found in portfolio data"}
            
            # If no target provided, use equal weight
            if not target_allocation:
                symbols = list(current_allocations.keys())
                target_weight = 100.0 / len(symbols) if symbols else 0
                target_allocation = {symbol: target_weight for symbol in symbols}
            
            # Calculate rebalancing trades
            trades = []
            for symbol in current_allocations:
                current_pct = current_allocations[symbol]
                target_pct = target_allocation.get(symbol, 0)
                difference_pct = target_pct - current_pct
                difference_value = (difference_pct / 100.0) * total_value
                
                if abs(difference_value) > 50:  # Only recommend if difference > $50
                    action = "BUY" if difference_value > 0 else "SELL"
                    trades.append({
                        "symbol": symbol,
                        "action": action,
                        "current_allocation": current_pct,
                        "target_allocation": target_pct,
                        "difference_percent": difference_pct,
                        "estimated_amount": abs(difference_value),
                        "reasoning": f"Rebalance from {current_pct:.1f}% to {target_pct:.1f}%"
                    })
            
            result = {
                "status": "success",
                "rebalancing_trades": trades,
                "current_allocations": current_allocations,
                "target_allocations": target_allocation,
                "total_portfolio_value": total_value,
                "rebalancing_needed": len(trades) > 0
            }
            
            # Add ADK analysis if available
            if self.adk_agent and trades:
                try:
                    rebalancing_prompt = f"""
                    Analyze these rebalancing recommendations:
                    
                    Portfolio Value: ${total_value:,.2f}
                    Number of Trades: {len(trades)}
                    
                    Recommended Trades:
                    {self._format_trades(trades)}
                    
                    Provide insights on:
                    1. The benefits and risks of this rebalancing
                    2. Timing considerations
                    3. Tax implications to consider
                    4. Alternative approaches
                    """
                    
                    analysis = await self.adk_agent.process(rebalancing_prompt)
                    result["rebalancing_analysis"] = analysis
                    
                except Exception as e:
                    logger.warning(f"Failed to generate rebalancing analysis: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing: {e}")
            return {
                "error": f"Failed to calculate rebalancing: {str(e)}"
            }
    
    async def _assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio risk and diversification."""
        try:
            metrics = calculate_portfolio_metrics(portfolio_data)
            
            # Enhanced risk assessment
            risk_factors = []
            
            if metrics['concentration_risk'] > 0.5:
                risk_factors.append("High concentration risk - portfolio is not well diversified")
            
            if metrics['largest_holding_percentage'] > 40:
                risk_factors.append(f"Single position exceeds 40% of portfolio ({metrics['largest_holding_percentage']:.1f}%)")
            
            if metrics['position_count'] < 5:
                risk_factors.append(f"Low diversification - only {metrics['position_count']} positions")
            
            result = {
                "status": "success",
                "risk_metrics": metrics,
                "risk_factors": risk_factors,
                "overall_risk_level": self._determine_risk_level(metrics),
                "recommendations": self._generate_risk_recommendations(metrics)
            }
            
            # Add ADK risk analysis if available
            if self.adk_agent:
                try:
                    risk_prompt = f"""
                    Conduct a comprehensive risk analysis of this portfolio:
                    
                    Risk Metrics:
                    - Diversification Score: {metrics['diversification_score']:.1f}/10
                    - Concentration Risk: {metrics['concentration_risk']:.3f}
                    - Largest Holding: {metrics['largest_holding_percentage']:.1f}%
                    - Position Count: {metrics['position_count']}
                    - Risk Assessment: {metrics['risk_assessment']}
                    
                    Risk Factors Identified:
                    {chr(10).join(f'- {factor}' for factor in risk_factors)}
                    
                    Provide detailed risk analysis including:
                    1. Market risk considerations
                    2. Sector/correlation risks
                    3. Recommendations for risk mitigation
                    4. Stress testing scenarios to consider
                    """
                    
                    risk_analysis = await self.adk_agent.process(risk_prompt)
                    result["detailed_risk_analysis"] = risk_analysis
                    
                except Exception as e:
                    logger.warning(f"Failed to generate detailed risk analysis: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {
                "error": f"Failed to assess portfolio risk: {str(e)}"
            }
    
    async def _create_monthly_allocation_plan(self, portfolio_data: Dict[str, Any],
                                            market_data: Dict[str, Any], 
                                            amount: float) -> Dict[str, Any]:
        """Create a monthly investment allocation plan."""
        # This is essentially the same as generate_advice but focused on monthly planning
        return await self._generate_comprehensive_advice(
            portfolio_data, market_data, amount, "moderate"
        )
    
    def _determine_risk_level(self, metrics: Dict[str, Any]) -> str:
        """Determine overall portfolio risk level."""
        score = metrics['diversification_score']
        concentration = metrics['concentration_risk']
        
        if score >= 8 and concentration < 0.3:
            return "LOW"
        elif score >= 6 and concentration < 0.5:
            return "MODERATE" 
        elif score >= 4:
            return "HIGH"
        else:
            return "VERY HIGH"
    
    def _generate_risk_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []
        
        if metrics['diversification_score'] < 6:
            recommendations.append("Consider adding more positions to improve diversification")
        
        if metrics['largest_holding_percentage'] > 30:
            recommendations.append("Consider reducing the largest position to below 30% of portfolio")
        
        if metrics['position_count'] < 10:
            recommendations.append("Gradually build towards 10+ positions for better diversification")
        
        if metrics['concentration_risk'] > 0.4:
            recommendations.append("Focus on equal-weight allocation to reduce concentration risk")
        
        return recommendations
    
    def _format_recommendations(self, recommendations) -> str:
        """Format recommendations for display."""
        lines = []
        for rec in recommendations:
            lines.append(f"- {rec.symbol}: ${rec.amount_to_invest:.2f} ({rec.reasoning})")
        return "\n".join(lines)
    
    def _format_market_data(self, market_data: Dict[str, Any]) -> str:
        """Format market data for display."""
        if isinstance(market_data, dict) and 'prices' in market_data:
            prices = market_data['prices']
            lines = []
            for symbol, price_info in prices.items():
                price = price_info.get('price', 'N/A')
                lines.append(f"- {symbol}: ${price}")
            return "\n".join(lines)
        return "Market data format not recognized"
    
    def _format_trades(self, trades: List[Dict[str, Any]]) -> str:
        """Format trades for display."""
        lines = []
        for trade in trades:
            lines.append(f"- {trade['action']} {trade['symbol']}: ${trade['estimated_amount']:.2f}")
        return "\n".join(lines)

def main():
    """Main entry point for running the Investment Advisor Agent."""
    agent = InvestmentAdvisorAgent()
    
    logger.info(f"Starting Investment Advisor Agent on port {agent.port}")
    
    try:
        # agent.run() will block here and manage the asyncio loop via uvicorn
        agent.run()
    except KeyboardInterrupt:
        logger.info("Investment Advisor Agent stopped by user")
    except Exception as e:
        logger.error("Investment Advisor Agent failed", error=str(e), exc_info=True)
        # No raise here

if __name__ == "__main__":
    # Directly call main
    main() 