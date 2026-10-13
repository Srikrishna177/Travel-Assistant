"""Travel Assistant: a LangChain tool-calling agent powered by Claude.

The agent has two tools:
  1. get_weather_forecast : current weather + short forecast via WeatherAPI.com
  2. get_top_attractions  : top tourist attractions via DuckDuckGo web search

Setup:
    pip install langchain langchain-classic langchain-anthropic langchain-community \
                ddgs duckduckgo-search requests python-dotenv

Environment variables (or a .env file):
    ANTHROPIC_API_KEY   your Claude API key
    WEATHER_API_KEY     your WeatherAPI.com key
    CLAUDE_MODEL        optional, defaults to claude-sonnet-5

Run:
    python travel_assistant.py
"""

# %%
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# LangChain 1.x moved the legacy agent classes into `langchain_classic`.
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# %%
load_dotenv()

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
WEATHER_API_URL = "https://api.weatherapi.com/v1/forecast.json"
FORECAST_DAYS = 3
REQUEST_TIMEOUT_SECONDS = 10
MAX_SEARCH_RESULTS = 5

SYSTEM_PROMPT = (
    "You are a friendly and concise Travel Assistant. "
    "When the user names a destination, call BOTH tools: "
    "get_weather_forecast for the weather and get_top_attractions for places to visit. "
    "Only use facts returned by the tools. If a tool reports an error, say so plainly. "
    "Finish with a short answer in two sections: 'Weather' and 'Top Attractions'. "
    "Under Top Attractions, list 4 to 5 places with a one line reason each."
)

# %%
@tool
def get_weather_forecast(city: str) -> str:
    """Get the current weather and a short daily forecast for a city.

    Args:
        city: Name of the destination city, for example 'Paris' or 'Tokyo'.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Error: WEATHER_API_KEY is not set."

    params = {
        "key": api_key,
        "q": city,
        "days": FORECAST_DAYS,
        "aqi": "no",
        "alerts": "no",
    }

    try:
        response = requests.get(
            WEATHER_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        data = response.json()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json()["error"]["message"]
        except (ValueError, KeyError):
            detail = str(exc)
        return f"Could not fetch weather for '{city}': {detail}"
    except requests.RequestException as exc:
        return f"Could not reach the weather service: {exc}"

    location = data["location"]
    current = data["current"]
    lines = [
        f"Weather for {location['name']}, {location['country']}",
        (
            f"Now: {current['condition']['text']}, {current['temp_c']} C "
            f"(feels like {current['feelslike_c']} C), "
            f"humidity {current['humidity']}%, wind {current['wind_kph']} km/h"
        ),
        "Forecast:",
    ]
    for day in data["forecast"]["forecastday"]:
        info = day["day"]
        lines.append(
            f"  {day['date']}: {info['condition']['text']}, "
            f"{info['mintemp_c']} to {info['maxtemp_c']} C, "
            f"rain chance {info['daily_chance_of_rain']}%"
        )
    return "\n".join(lines)


_search = DuckDuckGoSearchResults(max_results=MAX_SEARCH_RESULTS)


@tool
def get_top_attractions(city: str) -> str:
    """Search the web for the top tourist attractions and places to visit in a city.

    Args:
        city: Name of the destination city, for example 'Paris' or 'Tokyo'.
    """
    try:
        return _search.invoke(f"top tourist attractions to visit in {city}")
    except Exception as exc:  # network errors, rate limits, missing packages
        return f"Attraction search failed for '{city}': {exc}"

# %%
def build_agent_executor(verbose: bool = True) -> AgentExecutor:
    """Bind the LLM and tools into a tool-calling agent wrapped in an executor."""
    llm = ChatAnthropic(model=MODEL_NAME, max_tokens=1024)
    tools = [get_weather_forecast, get_top_attractions]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=5,
        handle_parsing_errors=True,
    )

# %%
def extract_text(output: Any) -> str:
    """Claude can return a plain string or a list of content blocks."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in output
        )
    return str(output)


def ask_travel_assistant(executor: AgentExecutor, destination: str) -> str:
    result = executor.invoke(
        {"input": f"I am planning a trip to {destination}. "
                  "What is the weather like and what should I visit?"}
    )
    return extract_text(result["output"])

# %%
def main() -> None:
    missing = [k for k in ("ANTHROPIC_API_KEY", "WEATHER_API_KEY") if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")

    executor = build_agent_executor(verbose=True)
    print("Travel Assistant ready. Type a destination, or 'quit' to exit.\n")

    while True:
        destination = input("Destination: ").strip()
        if destination.lower() in {"quit", "exit", "q"}:
            break
        if not destination:
            continue
        print("\n" + ask_travel_assistant(executor, destination) + "\n")


if __name__ == "__main__":
    main()
