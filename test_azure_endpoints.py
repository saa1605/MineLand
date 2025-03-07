import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from openai import AzureOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'azure_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class AzureEndpointTester:
    """Test Azure OpenAI endpoints and configurations."""
    
    def __init__(
        self,
        chat_endpoint: str = os.environ.get("AZURE_OPENAI_ENDPOINT"),
        chat_api_key: str = os.environ.get("AZURE_OPENAI_API_KEY"),
        chat_api_version: str = "2024-08-01-preview",
        chat_deployment: str = "gpt-4o-v2",
        embeddings_endpoint: str = os.environ.get("AZURE_OPENAI_ENDPOINT_EMB"),
        embeddings_api_key: str = os.environ.get("AZURE_OPENAI_API_KEY_EMB"),
        embeddings_deployment: str = "text-embedding-3-large",
        embeddings_api_version: str = "2023-05-15",
    ):
        """
        Initialize the tester with Azure configurations.

        Args:
            chat_endpoint: Azure endpoint for chat completions
            chat_api_key: API key for chat completions
            chat_api_version: API version for chat completions
            chat_deployment: Model deployment name for chat
            embeddings_endpoint: Azure endpoint for embeddings
            embeddings_api_key: API key for embeddings
            embeddings_deployment: Model deployment name for embeddings
            embeddings_api_version: API version for embeddings
        """
        self.chat_endpoint = chat_endpoint
        self.chat_api_key = chat_api_key
        self.chat_api_version = chat_api_version
        self.chat_deployment = chat_deployment
        self.embeddings_endpoint = embeddings_endpoint
        self.embeddings_api_key = embeddings_api_key
        self.embeddings_deployment = embeddings_deployment
        self.embeddings_api_version = embeddings_api_version

        if not self.chat_api_key:
            logger.warning("Chat API key not provided!")

    def test_chat_completion(self) -> Dict[str, Any]:
        """Test the chat completion endpoint."""
        try:
            client = AzureOpenAI(
                azure_endpoint=self.chat_endpoint,
                api_key=self.chat_api_key,
                api_version=self.chat_api_version
            )

            response = client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello, this is a test!' in exactly those words."}
                ],
                max_tokens=50
            )

            return {
                "success": True,
                "response": response.choices[0].message.content,
                "usage": response.usage.total_tokens
            }

        except Exception as e:
            logger.error(f"Chat completion test failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def test_embeddings(self) -> Dict[str, Any]:
        """Test the embeddings endpoint."""
        try:
            client = AzureOpenAI(
                azure_endpoint=self.embeddings_endpoint,
                api_key=self.embeddings_api_key,
                api_version=self.embeddings_api_version
            )

            response = client.embeddings.create(
                model=self.embeddings_deployment,
                input="Hello, this is a test!"
            )

            return {
                "success": True,
                "embedding_length": len(response.data[0].embedding),
                "usage": response.usage.total_tokens
            }

        except Exception as e:
            logger.error(f"Embeddings test failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def run_all_tests(self) -> None:
        """Run all endpoint tests and log results."""
        logger.info("Starting Azure OpenAI endpoint tests...")

        # Test chat completion
        logger.info("\nTesting Chat Completion endpoint...")
        chat_result = self.test_chat_completion()
        if chat_result["success"]:
            logger.info("✅ Chat Completion test passed!")
            logger.info(f"Response: {chat_result['response']}")
            logger.info(f"Tokens used: {chat_result['usage']}")
        else:
            logger.error(f"❌ Chat Completion test failed: {chat_result['error']}")

        # Test embeddings
        logger.info("\nTesting Embeddings endpoint...")
        embeddings_result = self.test_embeddings()
        if embeddings_result["success"]:
            logger.info("✅ Embeddings test passed!")
            logger.info(f"Embedding vector length: {embeddings_result['embedding_length']}")
            logger.info(f"Tokens used: {embeddings_result['usage']}")
        else:
            logger.error(f"❌ Embeddings test failed: {embeddings_result['error']}")

def main():
    """Main function to run the tests."""
    # You would pass your API key here
    tester = AzureEndpointTester()
    
    # Run all tests
    tester.run_all_tests()

if __name__ == "__main__":
    main() 