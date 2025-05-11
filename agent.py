import requests
import openai
import pyttsx3
import networkx as nx
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright
from llama_index import GPTIndex, Document
from mcp_server_client import MCPClient
from typing import List, Dict, Any, Optional

class DataAgent:
    """
    An agent responsible for data collection, processing, and presentation
    using LlamaIndex and MCP servers, with multi-agent collaboration.
    """

    def __init__(self, mcp_server_url: str, openai_api_key: str):
        """
        Initialize the DataAgent with MCP server URL and OpenAI API key.

        Args:
            mcp_server_url (str): URL of the MCP server.
            openai_api_key (str): API key for OpenAI.
        """
        self.mcp_client = MCPClient(mcp_server_url)
        openai.api_key = openai_api_key
        self.tts_engine = pyttsx3.init()
        self.index = None  # Will hold the LlamaIndex instance

    def fetch_data_from_mcp(self, task_id: str) -> Optional[str]:
        """
        Fetch data associated with a task ID from the MCP server.

        Args:
            task_id (str): Identifier for the task/data.

        Returns:
            Optional[str]: The fetched data as a string, or None if failed.
        """
        try:
            response = self.mcp_client.get_data(task_id)
            response.raise_for_status()
            data = response.text
            return data
        except requests.RequestException as e:
            print(f"Error fetching data from MCP: {e}")
            return None

    def process_data_with_llama(self, data: str) -> bool:
        """
        Process raw data and build a LlamaIndex.

        Args:
            data (str): Raw data to index.

        Returns:
            bool: True if indexing succeeded, False otherwise.
        """
        try:
            documents = [Document(text=data)]
            self.index = GPTIndex.from_documents(documents)
            return True
        except Exception as e:
            print(f"Error creating LlamaIndex: {e}")
            return False

    def generate_summary(self, prompt: str) -> str:
        """
        Generate a summary or response using OpenAI's API.

        Args:
            prompt (str): The prompt to send to the language model.

        Returns:
            str: The generated response.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating response."

    def visualize_graph(self, graph: nx.Graph, title: str = "Graph Visualization") -> None:
        """
        Visualize a NetworkX graph.

        Args:
            graph (nx.Graph): The graph to visualize.
            title (str): Title of the plot.
        """
        try:
            plt.figure(figsize=(8, 6))
            nx.draw_networkx(graph, with_labels=True, node_color='skyblue', edge_color='gray')
            plt.title(title)
            plt.show()
        except Exception as e:
            print(f"Error visualizing graph: {e}")

    def text_to_speech(self, text: str) -> None:
        """
        Convert text to speech.

        Args:
            text (str): Text to vocalize.
        """
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def run_agent_workflow(self, task_id: str) -> None:
        """
        Execute the full workflow: fetch, process, summarize, visualize, and speak.

        Args:
            task_id (str): Identifier for the task/data.
        """
        data = self.fetch_data_from_mcp(task_id)
        if data is None:
            print("Failed to fetch data.")
            return

        success = self.process_data_with_llama(data)
        if not success:
            print("Failed to process data with LlamaIndex.")
            return

        # Example: Generate a summary of the indexed data
        if self.index:
            prompt = "Summarize the following data:\n" + data
            summary = self.generate_summary(prompt)
            print("Summary:\n", summary)
            self.text_to_speech(summary)

        # Example: Visualize a simple graph (placeholder)
        graph = nx.Graph()
        graph.add_edge("Data", "Summary")
        self.visualize_graph(graph, title="Data Processing Graph")