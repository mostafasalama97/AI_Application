import os
import subprocess
import time  # For performance measurement
from dotenv import load_dotenv, find_dotenv

# LlamaIndex core classes for settings, indexing, and storage.
from llama_index.core import Settings, VectorStoreIndex, StorageContext
# Node parsers: standard splitter vs. window-based splitter.
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
# Weaviate client and authentication.
import weaviate
from weaviate.auth import AuthApiKey
# Weaviate vector store integration.
from llama_index.vector_stores.weaviate import WeaviateVectorStore
# Dataset downloader.
from llama_index.core.llama_dataset import download_llama_dataset
# HuggingFace embedding for generating embeddings.
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Postprocessors for advanced retrieval.
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank

# Custom LLM classes.
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata
)
from pydantic import Field

# === Custom DeepSeek LLM Integration as a subclass of CustomLLM ===
class CustomDeepSeekLLM(CustomLLM):
    """
    Custom integration for DeepSeek LLM via the local Ollama API.
    This class calls an external command to generate completions.
    """
    # Declare model and temperature as fields.
    model: str = Field(..., description="Name of the DeepSeek model")
    temperature: float = Field(0.1, description="Temperature setting for generation")
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Generate a complete response for the given prompt using DeepSeek.
        Executes the 'ollama run' command.
        """
        command = ["ollama", "run", self.model, prompt]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"CustomDeepSeekLLM error: {result.stderr}")
        response_text = result.stdout.strip()
        return CompletionResponse(text=response_text)

    def chat(self, messages, **kwargs) -> CompletionResponse:
        """
        Implements a simple chat interface by concatenating a list of messages.
        """
        prompt = "\n".join([msg.get("content", "") for msg in messages])
        return self.complete(prompt, **kwargs)

    @property
    def is_chat_model(self) -> bool:
        """Indicates that this LLM supports chat-based interactions."""
        return True

    @property
    def metadata(self) -> LLMMetadata:
        """
        Returns metadata about the LLM such as context window size and maximum output tokens.
        """
        return LLMMetadata(
            context_window=4096,
            num_output=256,
            model_name=self.model
        )

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """
        Streams the response token by token.
        Uses a simple whitespace split for tokenization.
        """
        command = ["ollama", "run", self.model, prompt]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"CustomDeepSeekLLM error: {result.stderr}")
        response_text = result.stdout.strip()
        accumulated = ""
        tokens = response_text.split()
        for token in tokens:
            accumulated += token + " "
            yield CompletionResponse(text=accumulated.strip(), delta=token + " ")

# === End Custom Integration ===

# Define a re-ranker for post-retrieval optimization.
rerank = SentenceTransformerRerank(
    top_n=2, 
    model="BAAI/bge-reranker-base"
)

def build_index(client, index_name, documents, node_parser):
    """
    Builds a VectorStoreIndex using the provided node parser.
    
    Args:
        client: Connected Weaviate client.
        index_name: Name of the index.
        documents: List of documents.
        node_parser: An instance of a node parser (e.g. SentenceSplitter or SentenceWindowNodeParser).
    
    Returns:
        A built VectorStoreIndex.
    """
    nodes = node_parser.get_nodes_from_documents(documents)
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=index_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index

def run_query(index, query_text, advanced=False):
    """
    Runs a query using the given index.
    
    For advanced RAG, hybrid retrieval and post-retrieval reranking are applied.
    
    Args:
        index: The VectorStoreIndex to query.
        query_text: The query string.
        advanced: Boolean flag; if True, uses advanced settings.
    
    Returns:
        The response from the query engine.
    """
    if advanced:
        # Advanced configuration: hybrid search and re-ranking.
        query_engine = index.as_query_engine(
            similarity_top_k=6,
            node_postprocessors=[rerank],
            vector_store_query_mode="hybrid",
            alpha=0.5,  # 0 = keyword-only; 1 = pure vector search; 0.5 = balanced.
        )
    else:
        # Naive configuration: default query engine.
        query_engine = index.as_query_engine()
    
    return query_engine.query(query_text)

def main():
    """
    Main function to compare the response time of naive RAG versus advanced RAG.
    
    The pipeline:
      1. Loads environment variables.
      2. Connects to Weaviate.
      3. Configures the global LLM and embedding models.
      4. Downloads the dataset.
      5. Builds two indices:
         - Naive index: using SentenceSplitter.
         - Advanced index: using SentenceWindowNodeParser with additional optimizations.
      6. Runs the same query on both indices and prints both the responses and the time taken.
    """
    # Load environment variables.
    load_dotenv(find_dotenv())

    # --- Step 0: Connect to Weaviate ---
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
        )
        if client.is_ready():
            print("✅ Connected to Weaviate!")
        else:
            print("❌ Weaviate connection failed.")
            return
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {e}")
        return

    # --- Step 1: Configure global models ---
    Settings.llm = CustomDeepSeekLLM(model="deepseek-coder-v2:16b", temperature=0.1)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # --- Step 2: Download dataset ---
    start_time = time.time()
    rag_dataset, documents = download_llama_dataset("PaulGrahamEssayDataset", "./paul_graham")
    download_duration = time.time() - start_time
    print(f"Dataset downloaded in {download_duration:.2f} seconds.")

    # --- Step 3: Build two indices for comparison ---
    # Naive RAG index using the standard SentenceSplitter.
    naive_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    naive_index = build_index(client, "NaiveRag", documents, naive_parser)
    
    # Advanced RAG index using SentenceWindowNodeParser.
    advanced_parser = SentenceWindowNodeParser(
        chunk_size=1024,
        chunk_overlap=20,
        window_size=3,      # Group 3 sentences per window.
        window_overlap=1,   # Overlap between windows.
    )
    advanced_index = build_index(client, "AdvancedRag", documents, advanced_parser)

    # --- Step 4: Run the same query on both indices and measure response time ---
    query_text = "tell me every thing you know?"
    
    print("\n--- Naive RAG Response ---")
    start_naive = time.time()
    naive_response = run_query(naive_index, query_text, advanced=False)
    naive_time = time.time() - start_naive
    print(naive_response)
    print(f"Naive RAG Query Time: {naive_time:.2f} seconds")
    
    print("\n--- Advanced RAG Response ---")
    start_advanced = time.time()
    advanced_response = run_query(advanced_index, query_text, advanced=True)
    advanced_time = time.time() - start_advanced
    print(advanced_response)
    print(f"Advanced RAG Query Time: {advanced_time:.2f} seconds")

    # Clean up the Weaviate client.
    client.close()

if __name__ == "__main__":
    main()
