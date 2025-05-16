import streamlit as st
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import gc
import faiss
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import re
import logging
import torch
from typing import List, Tuple
import time

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hàm xử lý văn bản
def preprocess_text(text: str) -> str:
    try:
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return text

# Tải dữ liệu với cache
@st.cache_data
def load_documents(contexts_path: str, questions_path: str, answers_path: str) -> Tuple[List[str], List[str], List[str]]:
    try:
        contexts, questions, answers = [], [], []
        for path, output_list in [(contexts_path, contexts), (questions_path, questions), (answers_path, answers)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()[:100]
                output_list.extend(preprocess_text(line.strip()) for line in lines if line.strip())
        if not (len(contexts) == len(questions) == len(answers)):
            raise ValueError("Mismatch in number of contexts, questions, and answers")
        return contexts, questions, answers
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise

# Chia nhỏ văn bản với cache
@st.cache_data
def chunk_documents(contexts: List[str], questions: List[str], answers: List[str], _text_splitter) -> Tuple[List[str], List[dict]]:
    try:
        chunks = []
        metadata = []
        for context, question, answer in tqdm(zip(contexts, questions, answers), total=len(contexts), desc="Chunking documents"):
            doc_chunks = _text_splitter.split_text(context)
            for chunk in doc_chunks:
                chunks.append(chunk)
                metadata.append({"context": context, "question": question, "answer": answer})
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
        chunk_size = 256  # Giảm chunk_size để tiết kiệm bộ nhớ
        batch_size = 4    # Giảm batch_size để tối ưu CPU
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
def answer_question(query: str, documents: List[str], doc_embeddings: np.ndarray, metadata: List[dict], model, generator, language: str = "vi") -> Tuple[str, List[Tuple[str, dict]]]:
    try:
        start_time = time.time()
        retrieved_docs = custom_retrieval(query, documents, doc_embeddings, metadata, model)
        if not retrieved_docs:
            return "Không tìm thấy tài liệu liên quan.", []

        context = " ".join([doc for doc, _ in retrieved_docs[:3]])
        example_prompt = ""
        for _, meta in retrieved_docs[:2]:
            example_prompt += f"Ví dụ: Câu hỏi: {meta['question']} Trả lời: {meta['answer']}\n"

        prompt = (
            f"Ngữ cảnh: {context}\n"
            f"{example_prompt}"
            f"Câu hỏi: {query}\n"
            f"Trả lời: Chỉ cung cấp câu trả lời chính xác, không thêm gì khác."
        )

        outputs = generator(prompt, max_new_tokens=50, temperature=0.3, top_p=0.9, do_sample=True, num_beams=2)
        answer = outputs[0]["generated_text"].strip()

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
        answer, sources = answer_question(query_text, st.session_state.chunks, st.session_state.embeddings, st.session_state.metadata, st.session_state.model, st.session_state.generator, language)
        source_text = sources[0][0] if sources else "Không tìm thấy nguồn."
        return answer, source_text
    except Exception as e:
        logger.error(f"Error in chatbot_rag: {e}")
        return "Lỗi khi xử lý yêu cầu.", ""

# Giao diện Streamlit
def main():
    st.set_page_config(page_title="hatbot", page_icon="🤖", layout="wide")
    
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
                contexts_path = './data/viquad.contexts'
                questions_path = './data/viquad.questions'
                answers_path = './data/viquad.answers'
                contexts, questions, answers = load_documents(contexts_path, questions_path, answers_path)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
                st.session_state.chunks, st.session_state.metadata = chunk_documents(contexts, questions, answers, text_splitter)
                # st.success(f"Đã chia nhỏ thành {len(st.session_state.chunks)} đoạn văn bản.")

                model_path = 'halong_embedding'
                st.session_state.embeddings = create_embeddings(st.session_state.chunks, model_path)

                model_name = "distilgpt2"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_llm = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
                st.session_state.generator = pipeline(
                    "text-generation",
                    model=model_llm,
                    tokenizer=tokenizer,
                    max_new_tokens=50,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=2,
                    return_full_text=False
                )
                st.session_state.model = SentenceTransformer(model_path, device='cpu')
                st.session_state.initialized = True
                # st.success("Khởi tạo mô hình và dữ liệu thành công!")
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
        query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?")
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