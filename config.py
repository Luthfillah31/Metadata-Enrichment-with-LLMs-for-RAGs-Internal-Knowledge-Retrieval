import os


# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = "VARELab-GPT4o"
AZURE_API_VERSION = "2024-08-01-preview"
TEMPERATURE = 0.1

# Processing configuration
BATCH_SIZE = 1
REQUEST_TIMEOUT = 5  # seconds
RETRY_LIMIT = 2
RETRY_DELAY = 5
MODEL = "deepseek-v3.1:671b-cloud"

# Output configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "metadata_gen_output")
EVALUATION_DIR = os.getenv("EVALUATION_DIR", "evaluation")
