import asyncio
import logging
from typing import List, Dict, Any, Optional

from llama_index import GPTIndex
from mcp_server_client import MCPClient
import requests
import openai
import pyttsx3
import networkx as nx
import matplotlib.pyplot as plt
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for configuration
MCP_SERVER_URL = "http://localhost:8000"
OPENAI_API_KEY = "your-openai-api-key"

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

async def fetch_data_from_mcp(mcp_client: MCPClient, query: str) -> Optional[str]:
    """
    Fetch data from MCP server based on a query.
    """
    try:
        response = mcp_client.query(query)
        if response.status_code == 200:
            data = response.json().get("data")
            logger.info(f"Data fetched for query '{query}': {data}")
            return data
        else:
            logger.error(f"Failed to fetch data: {response.status_code}")
            return None
    except Exception as e:
        logger.exception(f"Error fetching data from MCP: {e}")
        return None

def process_data_with_llama(data: str) -> str:
    """
    Process raw data using LlamaIndex to generate a summary or structured output.
    """
    try:
        index = GPTIndex.from_documents([{"text": data}])
        summary = index.query("Summarize the above data.")
        logger.info("Data processed with LlamaIndex.")
        return summary
    except Exception as e:
        logger.exception(f"Error processing data with LlamaIndex: {e}")
        return ""

def visualize_relationships(graph: nx.Graph, title: str = "Data Relationships") -> None:
    """
    Visualize a network graph using matplotlib.
    """
    try:
        plt.figure(figsize=(8, 6))
        nx.draw_networkx(graph, with_labels=True, node_color='skyblue', edge_color='gray')
        plt.title(title)
        plt.show()
        logger.info("Graph visualization displayed.")
    except Exception as e:
        logger.exception(f"Error visualizing graph: {e}")

async def automate_web_interaction(url: str) -> None:
    """
    Use Playwright to automate web interactions.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            # Example interaction: take screenshot
            await page.screenshot(path="screenshot.png")
            await browser.close()
            logger.info(f"Web interaction completed for {url}")
    except Exception as e:
        logger.exception(f"Error during web automation: {e}")

def speak_text(text: str) -> None:
    """
    Convert text to speech.
    """
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
        logger.info("Text spoken successfully.")
    except Exception as e:
        logger.exception(f"Error in text-to-speech: {e}")

async def main() -> None:
    """
    Main function orchestrating data collection, processing, and presentation.
    """
    try:
        # Initialize MCP client
        mcp_client = MCPClient(MCP_SERVER_URL)

        # Step 1: Data Collection
        query = "Latest research papers on AI"
        raw_data = await fetch_data_from_mcp(mcp_client, query)
        if not raw_data:
            logger.error("No data retrieved. Exiting.")
            return

        # Step 2: Data Processing
        summary = process_data_with_llama(raw_data)
        if not summary:
            logger.error("Processing failed. Exiting.")
            return

        # Step 3: Presentation
        print("Research Summary:\n", summary)
        speak_text(summary)

        # Optional: Visualize relationships (mock example)
        graph = nx.Graph()
        graph.add_edges_from([("AI", "Machine Learning"), ("AI", "Deep Learning"), ("ML", "Supervised")])
        visualize_relationships(graph)

        # Optional: Web automation example
        await automate_web_interaction("https://example.com")

    except Exception as e:
        logger.exception(f"An error occurred in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())