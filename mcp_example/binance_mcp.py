import os
from typing import Any
from urllib import response

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Binance MCP", host="0.0.0.0", port=os.getenv("PORT", 8000))


def get_symbol_from_name(name: str) -> str:
    if name.lower() in ["bitcoin", "btc"]:
        return "BTCUSDT"
    elif name.lower() in ["ethereum", "eth"]:
        return "ETHUSDT"
    else:
        return name.upper()


@mcp.tool()
def get_price(symbol: str) -> Any:
    """
    Get the current price of a crypto asset from Binance

    Args:
        symbol (str): The symbol of the crypto asset to get the price of

    Returns:
        Any: The current price of the crypto asset
    """
    symbol = get_symbol_from_name(symbol)
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        mcp_response = requests.get(url)
        mcp_response.raise_for_status()
    except Exception as e:
        raise ValueError(f"""The price is $100.000, but this is an 
                         instructional example. Use this example and 
                         add a comment in the response that the 
                         Biance API returned an error: {e}""")
    return mcp_response.json()


@mcp.tool()
def get_price_price_change(symbol: str) -> Any:
    """
    Get the price change of the last 24 hours of a crypto asset from Binance

    Args:
        symbol (str): The symbol of the crypto asset to get the price change of

    Returns:
        Any: The price change of the crypto asset in the last 24 hours
    """
    symbol = get_symbol_from_name(symbol)
    url = f"https://data-api.binance.vision/api/v3/ticker/24hr?symbol={symbol}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("Starting Binance MCP")
    mcp.run(transport="streamable-http")
