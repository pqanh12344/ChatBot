import streamlit as st
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import gc
import faiss
import re
import logging
import torch
from typing import List, Tuple
import time
import requests
import PyPDF2

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình API
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

def query(payload):
    """Gửi yêu cầu đến API Hugging Face"""
    # Thay 'YOUR_API_KEY' bằng API Key thực tế từ Hugging Face
    API_KEY = 'YOUR_API_KEY'
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_answer(prompt: str) -> str:
    """Tạo câu trả lời bằng cách gọi API Mixtral-8x7B"""
    prompt_with_marker = prompt + "\nTrả lời:"
    payload = {
        "inputs": prompt_with_marker,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "num_beams": 2
        }
    }
    output = query(payload)
    if isinstance(output, dict) and 'error' in output:
        return f"API error: {output['error']}"
    elif isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
        generated_text = output[0]['generated_text'].strip()
        marker_index = generated_text.find("\nTrả lời:")
        if marker_index != -1:
            answer = generated_text[marker_index + len("\nTrả lời:"):].strip()
            if not answer:
                return "Không có câu trả lời được tạo ra."
        else:
            return "Không tìm thấy câu trả lời."
        return answer
    else:
        return "Error: Unexpected API response format."

# Hàm xử lý văn bản
def preprocess_text(text: str) -> str:
    try:
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return text

# Tải dữ liệu từ PDF với cache
@st.cache_data
def load_documents(pdf_path: str) -> List[str]:
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        contexts = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    contexts.append(preprocess_text(text))
        if not contexts:
            raise ValueError("No text extracted from PDF")
        return contexts
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise

# Chia nhỏ văn bản với cache
@st.cache_data
def chunk_documents(contexts: List[str], _text_splitter) -> Tuple[List[str], List[dict]]:
    try:
        chunks = []
        metadata = []
        for context in tqdm(contexts, desc="Chunking documents"):
            doc_chunks = _text_splitter.split_text(context)
            for chunk in doc_chunks:
                chunks.append(chunk)
                metadata.append({"context": context})
        return chunks, metadata
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise

# Tạo embeddings với cache
@st.cache_data
def create_embeddings(chunks: List[str], model_path: str) -> np.ndarray:
    try:
        model = SentenceTransformer(model_path, device='cpu')
        gc.collect()
        chunk_size = 256
        batch_size = 4
        embeddings = []
        for i in tqdm(range(0, len(chunks), chunk_size), desc="Creating embeddings"):
            batch = chunks[i:i + chunk_size]
            batch_embeddings = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
            gc.collect()
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise

# Truy xuất tài liệu
def custom_retrieval(query: str, documents: List[str], doc_embeddings: np.ndarray, metadata: List[dict], model, top_k: int = 5) -> List[Tuple[str, dict]]:
    try:
        start_time = time.time()
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        _, top_indices = index.search(query_embedding, min(top_k, len(documents)))
        retrieved_docs = [(documents[i], metadata[i]) for i in top_indices[0]]
        elapsed_time = time.time() - start_time
        if elapsed_time < 1:
            time.sleep(1 - elapsed_time)
        return retrieved_docs
    except Exception as e:
        logger.error(f"Error in custom_retrieval: {e}")
        return []

# Trả lời câu hỏi
def answer_question(query: str, documents: List[str], doc_embeddings: np.ndarray, metadata: List[dict], model, language: str = "vi") -> Tuple[str, List[Tuple[str, dict]]]:
    try:
        start_time = time.time()
        retrieved_docs = custom_retrieval(query, documents, doc_embeddings, metadata, model)
        if not retrieved_docs:
            return "Không tìm thấy tài liệu liên quan.", []

        context = " ".join([doc for doc, _ in retrieved_docs[:3]])
        prompt = (
            f"Bạn là một chuyên gia thông tin, luôn sẵn sàng cung cấp câu trả lời chính xác và ngắn gọn.\n"
            f"Dựa trên thông tin sau đây: {context}\n"
            f"Câu hỏi của người dùng: {query}\n"
        )

        answer = generate_answer(prompt)

        elapsed_time = time.time() - start_time
        if elapsed_time < 1:
            time.sleep(1 - elapsed_time)
        return answer, retrieved_docs
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return "Lỗi khi tạo câu trả lời.", []

# Chatbot RAG
def chatbot_rag(query_text: str = None, language: str = "vi") -> Tuple[str, str]:
    try:
        if not query_text:
            return "Vui lòng nhập câu hỏi.", ""
        answer, sources = answer_question(query_text, st.session_state.chunks, st.session_state.embeddings, st.session_state.metadata, st.session_state.model, language)
        source_text = sources[0][0] if sources else "Không tìm thấy nguồn."
        return answer, source_text
    except Exception as e:
        logger.error(f"Error in chatbot_rag: {e}")
        return "Lỗi khi xử lý yêu cầu.", ""

# Giao diện Streamlit
def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
    
    # Thêm CSS tùy chỉnh
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .stExpander {
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Chatbot")
    st.write("Hỏi bất kỳ câu hỏi nào!")

    # Khởi tạo dữ liệu và mô hình nếu chưa có
    if 'initialized' not in st.session_state:
        with st.spinner("Đang tải dữ liệu và mô hình..."):
            try:
                pdf_path = './file.pdf'
                contexts = load_documents(pdf_path)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
                st.session_state.chunks, st.session_state.metadata = chunk_documents(contexts, text_splitter)

                model_path = 'halong_embedding'
                st.session_state.embeddings = create_embeddings(st.session_state.chunks, model_path)
                st.session_state.model = SentenceTransformer(model_path, device='cpu')
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"Khởi tạo thất bại: {e}")
                logger.error(f"Initialization error: {e}")
                return

    # Khởi tạo lịch sử trò chuyện
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Nút xóa lịch sử trò chuyện
    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.chat_history = []
        st.success("Đã xóa lịch sử trò chuyện.")

    # Form nhập câu hỏi
    with st.form(key='question_form'):
        query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Nội dung chính của tài liệu là gì?")
        submit_button = st.form_submit_button(label="Gửi câu hỏi")

    if submit_button and query:
        with st.spinner("Đang xử lý câu hỏi..."):
            answer, source = chatbot_rag(query)
            st.session_state.chat_history.append({"question": query, "answer": answer, "source": source})
            st.subheader("Câu trả lời:")
            st.markdown(f"**{answer}**")
            st.subheader("Nguồn:")
            st.write(source)

    # Hiển thị lịch sử trò chuyện
    if st.session_state.chat_history:
        st.subheader("Lịch sử trò chuyện")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Câu hỏi {len(st.session_state.chat_history) - i}: {chat['question']}"):
                st.markdown(f"**Trả lời**: {chat['answer']}")
                st.write(f"**Nguồn**: {chat['source']}")

if __name__ == "__main__":
    main()
