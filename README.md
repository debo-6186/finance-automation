# Financial Automation Agent System

A sophisticated 3-agent system built with Google's Agent Development Kit (ADK) and Agent2Agent (A2A) protocol for financial portfolio analysis and investment advice.

## System Architecture

This system consists of three specialized agents that communicate using Google's A2A protocol:

### 1. Portfolio Analysis Agent (Port 8001)
- **Purpose**: Reads and analyzes PDF stock trade statements
- **Capabilities**: 
  - PDF parsing and data extraction
  - Current portfolio allocation analysis
  - Stock position identification
  - Portfolio diversification analysis
- **A2A Skills**: `analyze_portfolio`, `get_current_allocations`

### 2. Market Data Agent (Port 8002)
- **Purpose**: Fetches real-time or last available stock prices
- **Capabilities**:
  - Real-time stock price retrieval
  - Trading hours detection
  - Historical price data
  - Market status information
- **A2A Skills**: `get_stock_price`, `get_market_status`, `get_multiple_prices`

### 3. Investment Advisor Agent (Port 8003)
- **Purpose**: Provides investment advice based on current portfolio and market data
- **Capabilities**:
  - Portfolio rebalancing recommendations
  - Monthly allocation advice ($2000 USD default)
  - Risk assessment
  - Diversification strategies
- **A2A Skills**: `generate_advice`, `calculate_rebalancing`

## Features

- **A2A Protocol Integration**: Full implementation of Google's Agent2Agent protocol
- **Multi-Agent Communication**: Seamless communication between specialized agents
- **Real-time Market Data**: Integration with financial APIs for current pricing
- **PDF Analysis**: Advanced PDF parsing for portfolio statements
- **Investment Intelligence**: AI-powered investment recommendations
- **Configurable Allocation**: Customizable monthly investment amounts

## Prerequisites

- Python 3.10+
- Google API key (for Gemini models, if not using Vertex AI with service account)
- Alpha Vantage API key (for market data)
- Docker (optional, for containerized deployment)
- **Optional but Recommended for Vertex AI**: Google Cloud Service Account Key JSON file.
  - Create a service account in your GCP project with the "Vertex AI User" role (or more restrictive roles like "Vertex AI Service Agent" if appropriate).
  - Download the JSON key file for this service account.

## Quick Start

1. **Clone and Setup**:
```bash
git clone <repository-url>
cd finance_automation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure Environment**:
```bash
cp env.example .env
# Edit .env with your API keys and paths
```
   - Set `GOOGLE_API_KEY` if you intend to use the Gemini API directly (i.e., `GOOGLE_GENAI_USE_VERTEXAI=False`).
   - If using Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=True`):
     - Set `GOOGLE_CLOUD_PROJECT` to your GCP Project ID.
     - Set `GOOGLE_CLOUD_LOCATION` (e.g., `us-central1`).
     - **Recommended for Vertex AI Authentication**: Set `GOOGLE_APPLICATION_CREDENTIALS` in your `.env` file to the absolute path of your downloaded service account JSON key file. 
       Example: `GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json`
       The Google Cloud client libraries will automatically pick up this environment variable for authentication.
     - Alternatively, you can authenticate using `gcloud auth application-default login` if you have the gcloud CLI installed and configured, but setting the environment variable is more explicit for applications.
   - Set `ALPHA_VANTAGE_API_KEY` for market data.

3. **Prepare Data Directory**:
```bash
mkdir -p data
# Place your portfolio PDF statement in data/portfolio_statement.pdf
```

4. **Start All Agents**:
```bash
# Terminal 1 - Portfolio Agent
python agents/portfolio_agent.py

# Terminal 2 - Market Data Agent  
python agents/market_agent.py

# Terminal 3 - Investment Advisor Agent
python agents/advisor_agent.py

# Terminal 4 - Main Orchestrator
python main.py
```

**Note on Running Agents:**
It is recommended to run the agent scripts as modules from the project's root directory (`finance_automation`) to ensure Python's import system works correctly. If you encounter `ModuleNotFoundError` for `utils` or other local packages, use the following commands instead:

```bash
# In the finance_automation root directory:

# Terminal 1 - Portfolio Agent
python -m agents.portfolio_agent

# Terminal 2 - Market Data Agent
python -m agents.market_agent

# Terminal 3 - Investment Advisor Agent
python -m agents.advisor_agent

# Terminal 4 - Main Orchestrator (remains the same)
python main.py
```

## Usage Examples

### Basic Portfolio Analysis
```python
# The system automatically:
# 1. Analyzes your PDF portfolio statement
# 2. Fetches current market prices
# 3. Generates investment advice for your monthly allocation
```

### Custom Monthly Allocation
```bash
# Set custom allocation amount
export MONTHLY_ALLOCATION_USD=3000
python main.py
```

## Project Structure

```
finance_automation/
├── agents/                 # Individual agent implementations
│   ├── portfolio_agent.py  # Portfolio analysis agent
│   ├── market_agent.py     # Market data agent
│   └── advisor_agent.py    # Investment advisor agent
├── tools/                  # Agent tools and utilities
│   ├── pdf_tools.py        # PDF processing tools
│   ├── market_tools.py     # Market data tools
│   └── analysis_tools.py   # Financial analysis tools
├── utils/                  # Shared utilities
│   ├── a2a_client.py       # A2A protocol client
│   ├── a2a_server.py       # A2A protocol server
│   └── config.py           # Configuration management
├── data/                   # Data directory
│   └── portfolio_statement.pdf
├── tests/                  # Test suite
├── main.py                # Main orchestrator
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## A2A Protocol Implementation

Each agent implements the A2A protocol with:
- **Agent Cards**: Capability discovery at `/.well-known/agent.json`
- **A2A Endpoints**: RESTful APIs for agent communication
- **Streaming Support**: Real-time updates for long-running tasks
- **Security**: Enterprise-grade authentication and authorization

### Agent Discovery

```bash
# Check agent capabilities
curl http://localhost:8001/.well-known/agent.json  # Portfolio Agent
curl http://localhost:8002/.well-known/agent.json  # Market Agent
curl http://localhost:8003/.well-known/agent.json  # Advisor Agent
```

## Testing

```bash
# Run all tests
pytest

# Run specific agent tests
pytest tests/test_portfolio_agent.py
pytest tests/test_market_agent.py
pytest tests/test_advisor_agent.py

# Run A2A protocol tests
pytest tests/test_a2a_communication.py
```

## Deployment Options

### Local Development
- Run agents individually on different ports
- Use main.py as orchestrator

### Docker Deployment
```bash
docker-compose up -d
```

### Google Cloud Deployment
- Deploy to Vertex AI Agent Engine
- Use Cloud Run for individual agents
- Deploy to GKE for scalable production

## Configuration

Key configuration options in `.env`:

- `MONTHLY_ALLOCATION_USD`: Your monthly investment amount
- `DEFAULT_PORTFOLIO_PDF_PATH`: Path to your portfolio statement
- `GOOGLE_API_KEY`: Gemini API key (used if `GOOGLE_GENAI_USE_VERTEXAI` is `False`).
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (required for Vertex AI).
- `GOOGLE_CLOUD_LOCATION`: The GCP region for Vertex AI services (e.g., `us-central1`).
- `GOOGLE_GENAI_USE_VERTEXAI`: Set to `True` to use Vertex AI, `False` for direct Gemini API.
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Google Cloud service account key JSON file (recommended for Vertex AI).
- `ALPHA_VANTAGE_API_KEY`: For real-time market data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Check the GitHub Issues
- Review the Google ADK documentation
- Consult the A2A protocol specification

---

Built with ❤️ using Google's Agent Development Kit and Agent2Agent Protocol
