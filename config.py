import logging
from dotenv import load_dotenv
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
PATH_QDRANT_URL = os.getenv("PATH_QDRANT_URL")
API_KEY_QDRANT = os.getenv("API_KEY_QDRANT")
PATH_SLACK_BOT_TOKEN = os.getenv("PATH_SLACK_BOT_TOKEN")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found")

# Cấu hình file paths
CONTEXTS_PATH = './data/viquad.contexts'
QUESTIONS_PATH = './data/viquad.questions'
ANSWERS_PATH = './data/viquad.answers'

# Cấu hình API
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
# API_KEY = HF_API_KEY  # Thay bằng API key thực tế hoặc sử dụng Streamlit secrets

# Cấu hình mô hình
MODEL_PATH = 'halong_embedding'
# MODEL_PATH = 'halong_embedding'
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
BATCH_SIZE = 4
TOP_K = 30

QDRANT_URL = PATH_QDRANT_URL
QDRANT_API_KEY = API_KEY_QDRANT
SLACK_BOT_TOKEN = PATH_SLACK_BOT_TOKEN
VECTOR_SIZE = 768
COLLECTION_NAME = 'my_documents_768'

