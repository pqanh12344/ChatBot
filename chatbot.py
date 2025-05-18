import time
import faiss
import numpy as np
from typing import List, Tuple
import requests
from sentence_transformers import SentenceTransformer
from config import logger, API_URL, API_KEY, TOP_K

def query(payload):
    """Gửi yêu cầu đến API Hugging Face."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_answer(prompt: str) -> str:
    """Tạo câu trả lời bằng cách gọi API Mixtral-8x7B."""
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

def custom_retrieval(query: str, documents: List[str], doc_embeddings: np.ndarray, metadata: List[dict], model: SentenceTransformer, top_k: int = TOP_K) -> List[Tuple[str, dict]]:
    """Truy xuất tài liệu liên quan dựa trên câu hỏi."""
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

def answer_question(query: str, documents: List[str], doc_embeddings: np.ndarray, metadata: List[dict], model: SentenceTransformer, language: str = "vi") -> Tuple[str, List[Tuple[str, dict]]]:
    """Trả lời câu hỏi dựa trên tài liệu truy xuất."""
    try:
        start_time = time.time()
        retrieved_docs = custom_retrieval(query, documents, doc_embeddings, metadata, model)
        if not retrieved_docs:
            return "Không tìm thấy tài liệu liên quan.", []

        context = " ".join([doc for doc, _ in retrieved_docs[:10]])
        example_prompt = ""
        for _, meta in retrieved_docs[:2]:
            example_prompt += f"Ví dụ: Câu hỏi: {meta['question']} Trả lời: {meta['answer']}\n"

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

def chatbot_rag(query_text: str, documents: List[str], embeddings: np.ndarray, metadata: List[dict], model: SentenceTransformer, language: str = "vi") -> Tuple[str, str]:
    """Xử lý câu hỏi và trả về câu trả lời cùng nguồn."""
    try:
        if not query_text:
            return "Vui lòng nhập câu hỏi.", ""
        answer, sources = answer_question(query_text, documents, embeddings, metadata, model, language)
        source_text = sources[0][0] if sources else "Không tìm thấy nguồn."
        return answer, source_text
    except Exception as e:
        logger.error(f"Error in chatbot_rag: {e}")
        return "Lỗi khi xử lý yêu cầu.", ""