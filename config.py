import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình file paths
CONTEXTS_PATH = './data/viquad.contexts'
QUESTIONS_PATH = './data/viquad.questions'
ANSWERS_PATH = './data/viquad.answers'

# Cấu hình API
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
API_KEY = 'fbfbfbdbd'  # Thay bằng API key thực tế hoặc sử dụng Streamlit secrets

# Cấu hình mô hình
MODEL_PATH = './model/halong_embedding'
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
BATCH_SIZE = 4
TOP_K = 30