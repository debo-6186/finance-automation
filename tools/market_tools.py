"""
Market data tools for fetching real-time stock prices and market information.
Supports multiple data providers with fallback options.
"""
import asyncio
from datetime import datetime, time, timezone
from typing import Dict, List, Any, Optional, Union
import structlog

try:
    import yfinance as yf
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    yf = None
    TimeSeries = None

logger = structlog.get_logger(__name__)

class StockPrice:
    """Represents a stock price with metadata."""
    
    def __init__(self, symbol: str, price: float, timestamp: datetime, 
                 currency: str = "USD", source: str = "unknown"):
        self.symbol = symbol.upper()
        self.price = price
        self.timestamp = timestamp
        self.currency = currency
        self.source = source
        
    @property
    def is_realtime(self) -> bool:
        """Check if the price is from today (real-time or delayed)."""
        return self.timestamp.date() == datetime.now().date()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "currency": self.currency,
            "source": self.source,
            "is_realtime": self.is_realtime
        }

class MarketStatus:
    """Represents market status information."""
    
    def __init__(self, is_open: bool, next_open: Optional[datetime] = None, 
                 next_close: Optional[datetime] = None, timezone_name: str = "UTC"):
        self.is_open = is_open
        self.next_open = next_open
        self.next_close = next_close
        self.timezone_name = timezone_name
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "is_open": self.is_open,
            "next_open": self.next_open.isoformat() if self.next_open else None,
            "next_close": self.next_close.isoformat() if self.next_close else None,
            "timezone": self.timezone_name
        }

class MarketDataProvider:
    """Base class for market data providers."""
    
    async def get_stock_price(self, symbol: str) -> StockPrice:
        """Get current stock price."""
        raise NotImplementedError
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, StockPrice]:
        """Get prices for multiple stocks."""
        raise NotImplementedError
    
    async def get_market_status(self) -> MarketStatus:
        """Get current market status."""
        raise NotImplementedError

class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider."""
    
    def __init__(self):
        if not yf:
            raise ImportError("yfinance is required for YFinanceProvider")
        self.source = "yahoo_finance"
    
    async def get_stock_price(self, symbol: str) -> StockPrice:
        """Get current stock price from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current data
            info = ticker.info
            history = ticker.history(period="1d", interval="1m")
            
            if history.empty:
                # Fallback to info data
                price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                timestamp = datetime.now()
            else:
                # Use latest price from history
                price = float(history['Close'].iloc[-1])
                timestamp = history.index[-1].to_pydatetime()
            
            currency = info.get('currency', 'USD')
            
            return StockPrice(symbol, price, timestamp, currency, self.source)
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol} from Yahoo Finance: {e}")
            raise
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, StockPrice]:
        """Get prices for multiple stocks from Yahoo Finance."""
        try:
            # Use yfinance's ability to download multiple tickers
            tickers = yf.Tickers(" ".join(symbols))
            results = {}
            
            for symbol in symbols:
                try:
                    ticker = getattr(tickers.tickers, symbol)
                    info = ticker.info
                    history = ticker.history(period="1d", interval="1m")
                    
                    if history.empty:
                        price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                        timestamp = datetime.now()
                    else:
                        price = float(history['Close'].iloc[-1])
                        timestamp = history.index[-1].to_pydatetime()
                    
                    currency = info.get('currency', 'USD')
                    results[symbol] = StockPrice(symbol, price, timestamp, currency, self.source)
                    
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multiple prices from Yahoo Finance: {e}")
            raise
    
    async def get_market_status(self) -> MarketStatus:
        """Get US market status (simplified version)."""
        try:
            # Get SPY as a proxy for market status
            spy = yf.Ticker("SPY")
            info = spy.info
            
            # Simple check based on market state
            market_state = info.get('marketState', 'CLOSED')
            is_open = market_state in ['REGULAR', 'PRE', 'POST']
            
            return MarketStatus(is_open, timezone_name="US/Eastern")
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            # Default to closed if error
            return MarketStatus(False, timezone_name="US/Eastern")

class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self, api_key: str):
        if not TimeSeries:
            raise ImportError("alpha_vantage is required for AlphaVantageProvider")
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.source = "alpha_vantage"
    
    async def get_stock_price(self, symbol: str) -> StockPrice:
        """Get current stock price from Alpha Vantage."""
        try:
            # Get intraday data (last 1 minute)
            data, meta_data = self.ts.get_intraday(symbol, interval='1min', outputsize='compact')
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Get the latest price
            latest_timestamp = data.index[-1]
            latest_price = float(data['4. close'].iloc[-1])
            
            return StockPrice(symbol, latest_price, latest_timestamp, "USD", self.source)
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol} from Alpha Vantage: {e}")
            raise
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, StockPrice]:
        """Get prices for multiple stocks from Alpha Vantage."""
        results = {}
        
        # Alpha Vantage has API rate limits, so we process sequentially
        for symbol in symbols:
            try:
                price = await self.get_stock_price(symbol)
                results[symbol] = price
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
                continue
        
        return results
    
    async def get_market_status(self) -> MarketStatus:
        """Get market status (simplified for Alpha Vantage)."""
        # Alpha Vantage doesn't provide direct market status
        # We'll do a simple time-based check for US markets
        now = datetime.now()
        
        # Simple heuristic: US markets are typically open 9:30 AM - 4:00 PM ET on weekdays
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            market_open_time = time(9, 30)  # 9:30 AM
            market_close_time = time(16, 0)  # 4:00 PM
            current_time = now.time()
            
            is_open = market_open_time <= current_time <= market_close_time
        else:
            is_open = False
        
        return MarketStatus(is_open, timezone_name="US/Eastern")

class MultiProviderMarketData:
    """Market data aggregator that tries multiple providers."""
    
    def __init__(self, providers: List[MarketDataProvider]):
        self.providers = providers
        if not providers:
            raise ValueError("At least one provider must be specified")
    
    async def get_stock_price(self, symbol: str) -> StockPrice:
        """Get stock price, trying providers in order."""
        last_exception = None
        
        for provider in self.providers:
            try:
                return await provider.get_stock_price(symbol)
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed for {symbol}: {e}")
                last_exception = e
                continue
        
        # If all providers failed, raise the last exception
        raise last_exception or Exception("All providers failed")
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, StockPrice]:
        """Get multiple prices, trying providers in order."""
        results = {}
        remaining_symbols = symbols.copy()
        
        for provider in self.providers:
            if not remaining_symbols:
                break
            
            try:
                provider_results = await provider.get_multiple_prices(remaining_symbols)
                results.update(provider_results)
                
                # Remove successfully fetched symbols
                remaining_symbols = [s for s in remaining_symbols if s not in provider_results]
                
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed for multiple prices: {e}")
                continue
        
        return results
    
    async def get_market_status(self) -> MarketStatus:
        """Get market status from the first available provider."""
        for provider in self.providers:
            try:
                return await provider.get_market_status()
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed for market status: {e}")
                continue
        
        # Default to closed if all providers fail
        return MarketStatus(False)

def create_market_data_client(alpha_vantage_api_key: Optional[str] = None) -> MultiProviderMarketData:
    """
    Create a market data client with available providers.
    
    Args:
        alpha_vantage_api_key: Optional Alpha Vantage API key
        
    Returns:
        MultiProviderMarketData instance
    """
    providers = []
    
    # Add Yahoo Finance provider (free, no API key required)
    if yf:
        providers.append(YFinanceProvider())
    
    # Add Alpha Vantage provider if API key is provided
    if alpha_vantage_api_key and TimeSeries:
        try:
            providers.append(AlphaVantageProvider(alpha_vantage_api_key))
        except Exception as e:
            logger.warning(f"Failed to initialize Alpha Vantage provider: {e}")
    
    if not providers:
        raise RuntimeError("No market data providers available. Install yfinance and/or alpha_vantage.")
    
    return MultiProviderMarketData(providers)

# Helper functions for quick access
async def get_stock_price(symbol: str, alpha_vantage_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to get a single stock price.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        alpha_vantage_api_key: Optional Alpha Vantage API key
        
    Returns:
        Dictionary with price information
    """
    client = create_market_data_client(alpha_vantage_api_key)
    price = await client.get_stock_price(symbol)
    return price.to_dict()

async def get_multiple_stock_prices(symbols: List[str], alpha_vantage_api_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Quick function to get multiple stock prices.
    
    Args:
        symbols: List of stock symbols
        alpha_vantage_api_key: Optional Alpha Vantage API key
        
    Returns:
        Dictionary mapping symbols to price information
    """
    client = create_market_data_client(alpha_vantage_api_key)
    prices = await client.get_multiple_prices(symbols)
    return {symbol: price.to_dict() for symbol, price in prices.items()}

async def get_market_status(alpha_vantage_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to get market status.
    
    Args:
        alpha_vantage_api_key: Optional Alpha Vantage API key
        
    Returns:
        Dictionary with market status information
    """
    client = create_market_data_client(alpha_vantage_api_key)
    status = await client.get_market_status()
    return status.to_dict() 