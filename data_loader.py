import streamlit as st
import os
from typing import List, Tuple
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from config import logger, CONTEXTS_PATH, QUESTIONS_PATH, ANSWERS_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def preprocess_text(text: str) -> str:
    """Xử lý trước văn bản để loại bỏ khoảng trắng thừa."""
    try:
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return text

@st.cache_data
def load_documents(contexts_path: str = CONTEXTS_PATH, questions_path: str = QUESTIONS_PATH, answers_path: str = ANSWERS_PATH) -> Tuple[List[str], List[str], List[str]]:
    """Tải dữ liệu từ các file văn bản."""
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

@st.cache_data
def chunk_documents(contexts: List[str], questions: List[str], answers: List[str], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> Tuple[List[str], List[dict]]:
    """Chia nhỏ văn bản thành các chunk."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        metadata = []
        for context, question, answer in tqdm(zip(contexts, questions, answers), total=len(contexts), desc="Chunking documents"):
            doc_chunks = text_splitter.split_text(context)
            for chunk in doc_chunks:
                chunks.append(chunk)
                metadata.append({"context": context, "question": question, "answer": answer})
        return chunks, metadata
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise