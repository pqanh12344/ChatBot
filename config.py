import logging
from dotenv import load_dotenv
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found")

# Cấu hình file paths
CONTEXTS_PATH = './data/viquad.contexts'
QUESTIONS_PATH = './data/viquad.questions'
ANSWERS_PATH = './data/viquad.answers'

# Cấu hình API
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
API_KEY = HF_API_KEY  # Thay bằng API key thực tế hoặc sử dụng Streamlit secrets

# Cấu hình mô hình
MODEL_PATH = 'halong_embedding'
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
BATCH_SIZE = 4
TOP_K = 30