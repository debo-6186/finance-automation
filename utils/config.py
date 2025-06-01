"""
Configuration management for the Finance Automation Agent System.
Handles environment variables, API keys, and system settings.
"""
import os
from typing import Optional
from dotenv import load_dotenv
import structlog

# Load environment variables
load_dotenv()

logger = structlog.get_logger(__name__)

class Config:
    """Configuration class for the Finance Automation system."""
    
    def __init__(self):
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables."""
        # Google Cloud and Gemini Configuration
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Alternative LLM providers
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        
        # Financial data APIs
        self.ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # A2A Protocol Configuration
        self.A2A_BASE_URL = os.getenv("A2A_BASE_URL", "http://localhost")
        self.PORTFOLIO_AGENT_PORT = int(os.getenv("PORTFOLIO_AGENT_PORT", 8001))
        self.MARKET_AGENT_PORT = int(os.getenv("MARKET_AGENT_PORT", 8002))
        self.ADVISOR_AGENT_PORT = int(os.getenv("ADVISOR_AGENT_PORT", 8003))
        
        # Agent Configuration
        self.MONTHLY_ALLOCATION_USD = float(os.getenv("MONTHLY_ALLOCATION_USD", 2000))
        self.DEFAULT_PORTFOLIO_PDF_PATH = os.getenv("DEFAULT_PORTFOLIO_PDF_PATH", "./data/portfolio_statement.pdf")
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Validate required configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        if self.GOOGLE_GENAI_USE_VERTEXAI:
            if not self.GOOGLE_CLOUD_PROJECT:
                logger.warning("GOOGLE_CLOUD_PROJECT not set. Required for Vertex AI.")
            if not self.GOOGLE_APPLICATION_CREDENTIALS:
                logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Recommended for Vertex AI authentication.")
            elif not os.path.exists(self.GOOGLE_APPLICATION_CREDENTIALS):
                logger.warning(f"Service account key file not found at {self.GOOGLE_APPLICATION_CREDENTIALS}")
        elif not self.GOOGLE_API_KEY:
            # If not using Vertex and no Google API key, check for Anthropic as a fallback
            if not self.ANTHROPIC_API_KEY:
                logger.warning("Neither GOOGLE_API_KEY (for direct Gemini) nor ANTHROPIC_API_KEY is set. LLM features will be limited if Vertex AI is not used.")
            else:
                logger.info("GOOGLE_API_KEY not set, will attempt to use ANTHROPIC_API_KEY as fallback if Vertex AI is disabled.")
        
        if not self.ALPHA_VANTAGE_API_KEY:
            logger.warning("ALPHA_VANTAGE_API_KEY not set. Market data features may not work.")
    
    @property
    def portfolio_agent_url(self) -> str:
        """Get the full URL for the Portfolio Agent."""
        return f"{self.A2A_BASE_URL}:{self.PORTFOLIO_AGENT_PORT}"
    
    @property
    def market_agent_url(self) -> str:
        """Get the full URL for the Market Data Agent."""
        return f"{self.A2A_BASE_URL}:{self.MARKET_AGENT_PORT}"
    
    @property
    def advisor_agent_url(self) -> str:
        """Get the full URL for the Investment Advisor Agent."""
        return f"{self.A2A_BASE_URL}:{self.ADVISOR_AGENT_PORT}"
    
    def get_agent_urls(self) -> dict:
        """Get all agent URLs as a dictionary."""
        return {
            "portfolio": self.portfolio_agent_url,
            "market": self.market_agent_url,
            "advisor": self.advisor_agent_url
        }

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config 