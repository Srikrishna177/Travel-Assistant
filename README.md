# Travel Assistant

A command-line travel assistant built with LangChain and Claude. Give it a
destination and it returns the current weather, a short forecast, and the top
tourist attractions to visit.

## How it works

The script runs a LangChain tool-calling agent backed by Claude. For each
destination, the agent calls two tools and combines what they return into one answer:

| Tool | Source | Returns |
| --- | --- | --- |
| `get_weather_forecast` | [WeatherAPI.com](https://www.weatherapi.com/) | Current conditions plus a 3-day forecast (temperature, rain chance, humidity, wind) |
| `get_top_attractions` | DuckDuckGo web search | Top search results for tourist attractions in the city |

The answer has two sections, **Weather** and **Top Attractions**, with 4 to 5
places and a one-line reason for each.

## Requirements

- Python 3.9+
- A Claude API key from [console.anthropic.com](https://console.anthropic.com/)
- A WeatherAPI.com API key (free tier works)

## Installation

```bash
pip install langchain langchain-classic langchain-anthropic langchain-community \
            ddgs duckduckgo-search requests python-dotenv
```

## Configuration

The script reads its API keys from environment variables or from a `.env` file
in the project folder. Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your-claude-api-key
WEATHER_API_KEY=your-weatherapi-key
CLAUDE_MODEL=claude-sonnet-5   # optional
```

`.env` is listed in `.gitignore`, so your keys are never committed.

## Usage

```bash
python travel_assistant.py
```

```
Travel Assistant ready. Type a destination, or 'quit' to exit.

Destination: Tokyo
```

Type `quit`, `exit`, or `q` to stop.

You can also use the agent from your own code:

```python
from travel_assistant import build_agent_executor, ask_travel_assistant

executor = build_agent_executor(verbose=False)
print(ask_travel_assistant(executor, "Paris"))
```

The file is split into `# %%` cells, so you can also run it step by step in
VS Code or Jupyter.

## Security note

Never commit real API keys. Keep them in `.env` (ignored by git) or in your
shell environment. If a key leaks, revoke it and generate a new one from the
Anthropic Console or the WeatherAPI.com dashboard.
