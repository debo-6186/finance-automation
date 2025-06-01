import asyncio
import structlog

from utils.config import get_config
from utils.a2a_client import A2AClient, Message, MessageRole, Part, PartType
from utils.a2a_server import TaskState # Import TaskState

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()  # Pretty printing for development
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

async def main():
    """
    Main orchestrator for the 3-agent financial automation system.
    Coordinates portfolio analysis, market data retrieval, and investment advice.
    """
    config = get_config()
    
    logger.info("Starting financial automation orchestration...")
    
    async with A2AClient() as client:
        try:
            # 1. Discover agents (optional, but good practice)
            logger.info("Discovering agents...")
            portfolio_agent_card = await client.discover_agent(config.portfolio_agent_url)
            market_agent_card = await client.discover_agent(config.market_agent_url)
            advisor_agent_card = await client.discover_agent(config.advisor_agent_url)
            logger.info("Agents discovered successfully.", 
                        portfolio_agent=portfolio_agent_card.name,
                        market_agent=market_agent_card.name,
                        advisor_agent=advisor_agent_card.name)

            # --- Step 1: Analyze Portfolio PDF ---
            logger.info("Step 1: Analyzing Portfolio PDF", pdf_path=config.DEFAULT_PORTFOLIO_PDF_PATH)
            portfolio_task_payload = {
                "action": "analyze_portfolio",
                "pdf_path": config.DEFAULT_PORTFOLIO_PDF_PATH
            }
            portfolio_message = Message(
                role=MessageRole.USER,
                parts=[Part(type=PartType.DATA, content=portfolio_task_payload)]
            )
            
            portfolio_task = await client.send_task(config.portfolio_agent_url, portfolio_message)
            logger.info("Portfolio analysis task sent.", task_id=portfolio_task.id)
            
            portfolio_task = await client.wait_for_task_completion(config.portfolio_agent_url, portfolio_task.id, max_wait=120.0)
            
            if portfolio_task.state != TaskState.COMPLETED:
                logger.error("Portfolio analysis task failed.", task_id=portfolio_task.id, error=portfolio_task.error)
                return

            # Extract portfolio data from the last agent message
            portfolio_analysis_data = None
            for msg in reversed(portfolio_task.messages):
                if msg.role == MessageRole.AGENT:
                    for part in msg.parts:
                        if part.type == PartType.DATA and isinstance(part.content, dict):
                            portfolio_analysis_data = part.content
                            break
                    if portfolio_analysis_data:
                        break
            
            if not portfolio_analysis_data or portfolio_analysis_data.get("status") != "success":
                logger.error("Failed to retrieve portfolio analysis data.", response=portfolio_analysis_data)
                return
            
            logger.info("Portfolio analysis successful.", 
                        num_positions=portfolio_analysis_data.get('analysis', {}).get('position_count', 'N/A'),
                        total_value=portfolio_analysis_data.get('analysis', {}).get('total_value', 'N/A'))

            # --- Step 2: Fetch Market Data ---
            portfolio_positions = portfolio_analysis_data.get('analysis', {}).get('positions', [])
            if not portfolio_positions:
                logger.warning("No positions found in portfolio analysis. Skipping market data and advice.")
                return

            stock_symbols = list(set([pos['symbol'] for pos in portfolio_positions if 'symbol' in pos]))
            if not stock_symbols:
                logger.warning("No stock symbols extracted from portfolio. Skipping market data and advice.")
                return

            logger.info("Step 2: Fetching Market Data", symbols=stock_symbols)
            market_task_payload = {
                "action": "get_multiple_prices",
                "symbols": stock_symbols
            }
            market_message = Message(
                role=MessageRole.USER,
                parts=[Part(type=PartType.DATA, content=market_task_payload)]
            )
            
            market_task = await client.send_task(config.market_agent_url, market_message)
            logger.info("Market data task sent.", task_id=market_task.id)
            
            market_task = await client.wait_for_task_completion(config.market_agent_url, market_task.id, max_wait=120.0)

            if market_task.state != TaskState.COMPLETED:
                logger.error("Market data task failed.", task_id=market_task.id, error=market_task.error)
                return

            market_data_response = None
            for msg in reversed(market_task.messages):
                if msg.role == MessageRole.AGENT:
                    for part in msg.parts:
                        if part.type == PartType.DATA and isinstance(part.content, dict):
                            market_data_response = part.content
                            break
                    if market_data_response:
                        break
            
            if not market_data_response or market_data_response.get("status") != "success":
                logger.error("Failed to retrieve market data.", response=market_data_response)
                return
            
            market_prices = market_data_response.get("prices", {})
            logger.info("Market data fetched successfully.", retrieved_count=len(market_prices))

            # --- Step 3: Generate Investment Advice ---
            logger.info("Step 3: Generating Investment Advice", monthly_allocation=config.MONTHLY_ALLOCATION_USD)
            advice_task_payload = {
                "action": "generate_advice",
                "portfolio_data": portfolio_analysis_data.get('analysis'), # Pass the 'analysis' sub-dictionary
                "market_data": {"prices": market_prices}, # Structure as expected by advisor agent
                "monthly_allocation": config.MONTHLY_ALLOCATION_USD,
                "risk_tolerance": "moderate" # Can be configured or made dynamic
            }
            advice_message = Message(
                role=MessageRole.USER,
                parts=[Part(type=PartType.DATA, content=advice_task_payload)]
            )
            
            advice_task = await client.send_task(config.advisor_agent_url, advice_message)
            logger.info("Investment advice task sent.", task_id=advice_task.id)
            
            advice_task = await client.wait_for_task_completion(config.advisor_agent_url, advice_task.id, max_wait=180.0)

            if advice_task.state != TaskState.COMPLETED:
                logger.error("Investment advice task failed.", task_id=advice_task.id, error=advice_task.error)
                return

            investment_advice_response = None
            for msg in reversed(advice_task.messages):
                if msg.role == MessageRole.AGENT:
                    for part in msg.parts:
                        if part.type == PartType.DATA and isinstance(part.content, dict):
                            investment_advice_response = part.content
                            break
                    if investment_advice_response:
                        break

            if not investment_advice_response or investment_advice_response.get("status") != "success":
                logger.error("Failed to retrieve investment advice.", response=investment_advice_response)
                return
                
            logger.info("Investment advice generated successfully.")

            # --- Display Results ---
            print_results(portfolio_analysis_data.get('analysis'), market_prices, investment_advice_response.get('advice'))

        except ConnectionRefusedError:
            logger.error("Connection refused. Ensure all agents are running on their configured ports.")
        except Exception as e:
            logger.error("An error occurred during orchestration.", error=str(e), exc_info=True)

def print_results(portfolio_analysis, market_prices, investment_advice):
    """Helper function to pretty-print the results."""
    
    print("\n" + "="*50)
    print("FINANCIAL AUTOMATION SYSTEM REPORT")
    print("="*50 + "\n")

    if portfolio_analysis:
        print("-" * 20 + " PORTFOLIO ANALYSIS " + "-" * 20)
        print(f"Total Portfolio Value: ${portfolio_analysis.get('total_value', 0):,.2f}")
        print(f"Number of Positions: {portfolio_analysis.get('position_count', 0)}")
        print(f"Cash Balance: ${portfolio_analysis.get('cash_balance', 0):,.2f}")
        print("Top 5 Positions:")
        for pos in portfolio_analysis.get('top_positions', [])[:5]:
            print(f"  - {pos['symbol']}: {pos['shares']} shares, Value: ${pos['current_value']:,.2f}")
        if 'ai_insights' in portfolio_analysis:
            print("\nAI Insights (Portfolio):")
            print(portfolio_analysis['ai_insights'])
        print("-" * 50 + "\n")

    if market_prices:
        print("-" * 20 + " MARKET DATA " + "-" * 20)
        print("Current Stock Prices:")
        for symbol, data in market_prices.items():
            print(f"  - {symbol}: ${data.get('price', 'N/A'):.2f} (Source: {data.get('source', 'N/A')})")
        print("-" * 50 + "\n")

    if investment_advice:
        print("-" * 20 + " INVESTMENT ADVICE " + "-" * 20)
        print(f"Monthly Allocation: ${investment_advice.get('monthly_allocation', 0):,.2f}")
        print(f"Risk Assessment: {investment_advice.get('risk_assessment', 'N/A')}")
        print(f"Diversification Score: {investment_advice.get('diversification_score', 0):.1f}/10")
        
        print("\nRecommendations:")
        for rec in investment_advice.get('recommendations', []):
            print(f"  - Invest ${rec['amount_to_invest']:,.2f} in {rec['symbol']}")
            print(f"    Reasoning: {rec['reasoning']}")
            print(f"    New Target Allocation: {rec['target_allocation']:.1f}%")
        
        print(f"\nTotal Recommended Investment: ${investment_advice.get('total_recommended_investment', 0):,.2f}")
        print(f"Cash to Keep: ${investment_advice.get('cash_allocation', 0):,.2f}")
        
        print(f"\nSummary: {investment_advice.get('summary', 'N/A')}")
        
        if 'strategic_insights' in investment_advice:
            print("\nStrategic Insights (Advisor):")
            print(investment_advice['strategic_insights'])
        print("-" * 50 + "\n")

if __name__ == "__main__":
    asyncio.run(main()) 