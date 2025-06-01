"""
Market Data Agent - Fetches real-time stock prices and market information.
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
from tools.market_tools import create_market_data_client, get_stock_price, get_multiple_stock_prices, get_market_status

logger = structlog.get_logger(__name__)

class MarketDataAgent(A2AServer):
    """
    Market Data Agent that provides real-time stock prices and market information
    using A2A protocol.
    """
    
    def __init__(self):
        config = get_config()
        
        # Define agent capabilities
        capabilities = [
            "Real-time Stock Prices",
            "Market Status Information",
            "Multiple Stock Quote Retrieval",
            "Trading Hours Detection"
        ]
        
        # Define A2A skills
        skills = {
            "get_stock_price": {
                "description": "Get current stock price for a single symbol",
                "parameters": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., AAPL, GOOGL, MSFT)",
                        "required": True
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Stock price information including current price, timestamp, and metadata"
                }
            },
            "get_multiple_prices": {
                "description": "Get current prices for multiple stock symbols",
                "parameters": {
                    "symbols": {
                        "type": "array",
                        "description": "List of stock symbols",
                        "required": True
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Dictionary mapping symbols to price information"
                }
            },
            "get_market_status": {
                "description": "Get current market status (open/closed)",
                "parameters": {},
                "returns": {
                    "type": "object",
                    "description": "Market status information including open/closed state and trading hours"
                }
            },
            "check_trading_hours": {
                "description": "Check if markets are currently in trading hours",
                "parameters": {},
                "returns": {
                    "type": "object",
                    "description": "Trading hours status and next market open/close times"
                }
            }
        }
        
        # Initialize A2A server
        super().__init__(
            agent_name="Market Data Agent",
            agent_description="Provides real-time stock prices and market information",
            capabilities=capabilities,
            skills=skills,
            port=config.MARKET_AGENT_PORT
        )
        
        # Initialize market data client
        self.market_client = create_market_data_client(config.ALPHA_VANTAGE_API_KEY)
        
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
                        temperature=0.2
                    )
                    self.adk_agent = Agent(
                        name="MarketAnalyst",
                        description="Expert in market data interpretation and stock price analysis",
                        model=model,
                        system_prompt=self.get_market_analyst_system_prompt()
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
                        temperature=0.2
                    )
                    self.adk_agent = Agent(
                        name="MarketAnalyst",
                        description="Expert in market data interpretation and stock price analysis",
                        model=model,
                        system_prompt=self.get_market_analyst_system_prompt()
                    )
                    model_initialized = True
                    active_model_source = "Google Gemini API (gemini-1.5-flash)"
                except Exception as e:
                    logger.warning(f"Failed to initialize ADK agent with Gemini API: {e}")

            # Try Anthropic if Google models failed or not configured
            if not model_initialized and config.ANTHROPIC_API_KEY:
                try:
                    model = LiteLlm(
                        model="anthropic/claude-sonnet-4-20250514", # Per user request
                        api_key=config.ANTHROPIC_API_KEY,
                        temperature=0.2
                    )
                    self.adk_agent = Agent(
                        name="MarketAnalystAnthropic",
                        description="Expert in market data interpretation (Anthropic)",
                        model=model,
                        system_prompt=self.get_market_analyst_system_prompt()
                    )
                    model_initialized = True
                    active_model_source = "Anthropic API (claude-sonnet-4-20250514)"
                except Exception as e:
                    logger.warning(f"Failed to initialize ADK agent with Anthropic: {e}")
            
            if self.adk_agent:
                logger.info(f"Google ADK agent initialized successfully using {active_model_source}")
            else:
                logger.warning("Failed to initialize ADK agent with any provider. LLM features will be unavailable.")

        logger.info("Market Data Agent initialized")
    
    async def process_task(self, task: Task) -> Union[str, Dict[str, Any]]:
        """
        Process incoming A2A tasks for market data requests.
        
        Args:
            task: The A2A task to process
            
        Returns:
            Market data results or response message
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
        
        # Extract symbols from text
        symbols = self._extract_symbols_from_text(text)
        
        if ("price" in text_lower or "quote" in text_lower) and symbols:
            if len(symbols) == 1:
                return await self._get_single_stock_price(symbols[0])
            else:
                return await self._get_multiple_stock_prices(symbols)
        
        elif "market" in text_lower and ("status" in text_lower or "open" in text_lower or "closed" in text_lower):
            return await self._get_market_status()
        
        elif "trading" in text_lower and "hours" in text_lower:
            return await self._check_trading_hours()
        
        else:
            # Use ADK agent for general market questions
            if self.adk_agent:
                try:
                    # First try to get market status for context
                    market_status = await self._get_market_status()
                    
                    enhanced_prompt = f"""
                    Market Context: {market_status.get('status', 'unknown')}
                    Market is currently: {'open' if market_status.get('is_open', False) else 'closed'}
                    
                    User Question: {text}
                    
                    Please answer the question about market data or stock prices.
                    If specific symbols are mentioned, let me know and I can fetch current prices.
                    """
                    
                    response = await self.adk_agent.process(enhanced_prompt)
                    return response
                except Exception as e:
                    logger.warning(f"ADK agent failed: {e}")
            
            return f"I can help you get stock prices, market status, or trading hours information. Example: 'Get price for AAPL' or 'Is the market open?'"
    
    async def _handle_data_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle structured data requests."""
        action = data.get("action", "").lower()
        
        if action == "get_stock_price":
            symbol = data.get("symbol")
            if not symbol:
                return {"error": "Symbol is required for get_stock_price action"}
            return await self._get_single_stock_price(symbol)
        
        elif action == "get_multiple_prices":
            symbols = data.get("symbols", [])
            if not symbols:
                return {"error": "Symbols list is required for get_multiple_prices action"}
            return await self._get_multiple_stock_prices(symbols)
        
        elif action == "get_market_status":
            return await self._get_market_status()
        
        elif action == "check_trading_hours":
            return await self._check_trading_hours()
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _get_single_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get price for a single stock symbol."""
        try:
            price_data = await self.market_client.get_stock_price(symbol.upper())
            
            # Enhance with market analysis if ADK is available
            if self.adk_agent:
                try:
                    analysis_prompt = f"""
                    Analyze this stock price data:
                    
                    Symbol: {price_data.symbol}
                    Current Price: ${price_data.price:.2f}
                    Timestamp: {price_data.timestamp}
                    Source: {price_data.source}
                    Is Real-time: {price_data.is_realtime}
                    
                    Provide brief context about:
                    1. Whether this is current market pricing
                    2. Any notable observations about the price
                    3. Market timing context
                    """
                    
                    analysis = await self.adk_agent.process(analysis_prompt)
                    
                    return {
                        "status": "success",
                        "symbol": symbol.upper(),
                        "price_data": price_data.to_dict(),
                        "ai_analysis": analysis
                    }
                except Exception as e:
                    logger.warning(f"Failed to generate AI analysis: {e}")
            
            return {
                "status": "success",
                "symbol": symbol.upper(),
                "price_data": price_data.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return {
                "error": f"Failed to fetch price for {symbol}: {str(e)}",
                "symbol": symbol.upper()
            }
    
    async def _get_multiple_stock_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """Get prices for multiple stock symbols."""
        try:
            # Normalize symbols
            symbols = [s.upper() for s in symbols]
            
            prices_data = await self.market_client.get_multiple_prices(symbols)
            
            # Convert to serializable format
            result = {
                "status": "success",
                "symbols": symbols,
                "prices": {},
                "summary": {
                    "requested_count": len(symbols),
                    "retrieved_count": len(prices_data),
                    "missing_symbols": [s for s in symbols if s not in prices_data]
                }
            }
            
            for symbol, price_obj in prices_data.items():
                result["prices"][symbol] = price_obj.to_dict()
            
            # Enhance with market summary if ADK is available
            if self.adk_agent and len(prices_data) > 1:
                try:
                    summary_prompt = f"""
                    Provide a brief market summary for these stocks:
                    
                    {self._format_prices_for_analysis(prices_data)}
                    
                    Include:
                    1. Overall market timing context
                    2. Any notable patterns across the stocks
                    3. Data quality/freshness notes
                    """
                    
                    market_summary = await self.adk_agent.process(summary_prompt)
                    result["market_summary"] = market_summary
                    
                except Exception as e:
                    logger.warning(f"Failed to generate market summary: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching multiple prices: {e}")
            return {
                "error": f"Failed to fetch prices: {str(e)}",
                "symbols": symbols
            }
    
    async def _get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        try:
            status = await self.market_client.get_market_status()
            
            result = {
                "status": "success",
                "market_status": status.to_dict(),
                "interpretation": {
                    "is_trading_day": status.is_open,
                    "market_state": "open" if status.is_open else "closed"
                }
            }
            
            # Enhance with context if ADK is available
            if self.adk_agent:
                try:
                    context_prompt = f"""
                    Explain the current market status:
                    
                    Market is: {'Open' if status.is_open else 'Closed'}
                    Timezone: {status.timezone_name}
                    
                    Provide context about:
                    1. What this means for traders and investors
                    2. When the next market session begins/ends
                    3. Any important timing considerations
                    """
                    
                    context = await self.adk_agent.process(context_prompt)
                    result["market_context"] = context
                    
                except Exception as e:
                    logger.warning(f"Failed to generate market context: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                "error": f"Failed to get market status: {str(e)}"
            }
    
    async def _check_trading_hours(self) -> Dict[str, Any]:
        """Check trading hours status."""
        # This is essentially the same as market status for our implementation
        return await self._get_market_status()
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        import re
        
        # Common patterns for stock symbols
        patterns = [
            r'\b([A-Z]{1,5})\b',  # 1-5 uppercase letters
            r'\$([A-Z]{1,5})\b',  # Dollar sign prefix
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.update(matches)
        
        # Filter out common false positives
        false_positives = {
            'A', 'I', 'THE', 'AND', 'OR', 'BUT', 'FOR', 'GET', 'PRICE', 
            'STOCK', 'MARKET', 'DATA', 'API', 'USD', 'IS', 'OF', 'TO'
        }
        
        return [s for s in symbols if s not in false_positives and len(s) >= 2]
    
    def _format_prices_for_analysis(self, prices_data: Dict[str, Any]) -> str:
        """Format price data for AI analysis."""
        lines = []
        for symbol, price_obj in prices_data.items():
            lines.append(f"- {symbol}: ${price_obj.price:.2f} ({price_obj.source}, {price_obj.timestamp})")
        
        return "\n".join(lines)

    def get_market_analyst_system_prompt(self) -> str:
        return """
        You are a market data specialist and financial analyst with expertise in:
        - Real-time stock price interpretation
        - Market conditions and trading patterns
        - Technical analysis basics
        - Market hours and trading schedules
        
        When providing market insights:
        - Focus on factual price movements and data
        - Explain market conditions clearly
        - Note if markets are open or closed
        - Provide context for price changes when possible
        
        Always be clear about the timestamp and source of price data.
        Do not provide investment advice - focus on data interpretation.
        """

def main():
    """Main entry point for running the Market Data Agent."""
    agent = MarketDataAgent()
    
    logger.info(f"Starting Market Data Agent on port {agent.port}")
    
    try:
        # agent.run() will block here and manage the asyncio loop via uvicorn
        agent.run()
    except KeyboardInterrupt:
        logger.info("Market Data Agent stopped by user")
    except Exception as e:
        logger.error("Market Data Agent failed", error=str(e), exc_info=True)
        # No raise here

if __name__ == "__main__":
    # Directly call main
    main() 