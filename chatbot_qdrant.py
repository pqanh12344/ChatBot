import time
import numpy as np
from typing import List, Tuple
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import VectorParams, Distance
from config import logger, API_URL, API_KEY, TOP_K, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, VECTOR_SIZE

# ================== KẾT NỐI QDRANT ==================
def connect_qdrant(url: str, api_key: str) -> QdrantClient:
    client = QdrantClient(url=url, api_key=api_key)
    return client

def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int):
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name not in existing_collections:
        print(f"Collection '{collection_name}' chưa tồn tại, tạo mới...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print("Tạo collection thành công!")
    else:
        print(f"Collection '{collection_name}' đã tồn tại.")

# ================== HUGGINGFACE API ==================
def query_api(payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(API_URL, headers=headers, json=payload)
    return resp.json()

def generate_answer(prompt: str) -> str:
    payload = {
        "inputs": prompt + "\nTrả lời:",
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "num_beams": 2
        }
    }
    output = query_api(payload)
    if isinstance(output, dict) and "error" in output:
        return f"API error: {output['error']}"
    elif isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
        generated_text = output[0]["generated_text"].strip()
        idx = generated_text.find("\nTrả lời:")
        if idx != -1:
            return generated_text[idx + len("\nTrả lời:"):].strip()
        return generated_text
    else:
        return "Error: Unexpected API response."

# ================== UPLOAD DOCUMENTS (KHÔNG METADATA) ==================
def upload_documents_to_qdrant(client: QdrantClient, collection_name: str, chunks: List[str], embeddings: np.ndarray):
    points = []
    for i, vector in enumerate(embeddings):
        points.append(rest.PointStruct(id=i, vector=vector.tolist(), payload={"text": chunks[i]}))
    client.upsert(collection_name=collection_name, points=points)
    print(f"Uploaded {len(points)} documents to Qdrant Cloud.")

# ================== CUSTOM RETRIEVAL ==================
def custom_retrieval(client: QdrantClient, query: str, model: SentenceTransformer, top_k: int = TOP_K) -> List[str]:
    try:
        query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        results = client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k)
        return [r.payload.get("text", "") for r in results]
    except Exception as e:
        logger.error(f"Error in custom_retrieval: {e}")
        return []

# ================== ANSWER QUESTION ==================
def answer_question(client: QdrantClient, query: str, model: SentenceTransformer) -> str:
    retrieved_docs = custom_retrieval(client, query, model, top_k=TOP_K)
    if not retrieved_docs:
        return "Không tìm thấy tài liệu liên quan."
    context = " ".join(retrieved_docs[:10])
    prompt = (
        f"Bạn là một chuyên gia thông tin, luôn cung cấp câu trả lời chính xác.\n"
        f"Dựa trên thông tin sau: {context}\n"
        f"Câu hỏi: {query}\n"
    )
    return generate_answer(prompt)

# ================== CHATBOT RAG ==================
def chatbot_rag(client: QdrantClient, query: str, model: SentenceTransformer) -> Tuple[str, str]:
    if not query:
        return "Vui lòng nhập câu hỏi.", ""
    answer = answer_question(client, query, model)
    sources = custom_retrieval(client, query, model, top_k=1)
    source_text = sources[0] if sources else "Không tìm thấy nguồn."
    return answer, source_text

# ================== EXAMPLE USAGE ==================
if __name__ == "__main__":
    client = connect_qdrant(QDRANT_URL, QDRANT_API_KEY)
    create_collection_if_not_exists(client, COLLECTION_NAME, VECTOR_SIZE)

    # Load embedding model
    model = SentenceTransformer("halong_embedding")

    # Ví dụ dữ liệu
    chunks = [
        "Hà Nội là thủ đô của Việt Nam.",
        "TP. Hồ Chí Minh nổi tiếng với chợ Bến Thành."
    ]
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    # Upload lên Qdrant (không metadata)
    upload_documents_to_qdrant(client, COLLECTION_NAME, chunks, embeddings)

    # Chatbot RAG
    query = "Hà Nội là thủ đô của nước nào?"
    answer, source = chatbot_rag(client, query, model)
    print("Answer:", answer)
    print("Source:", source)
