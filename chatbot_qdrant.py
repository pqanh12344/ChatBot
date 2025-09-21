import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import VectorParams, Distance
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import logger, TOP_K, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, VECTOR_SIZE

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

# ================== LOAD LOCAL QWEN2 LLM ==================
tokenizer = AutoTokenizer.from_pretrained("Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen2-0.5B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
qwen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1000,
    temperature=0.3,
    top_p=0.9
)

# ================== GENERATE ANSWER ==================
def generate_answer(prompt: str) -> str:
    try:
        output = qwen_pipeline(
            prompt,
            max_new_tokens=50,  # chỉ cần đủ để trả lời ngắn
            do_sample=False      # deterministic
        )[0]["generated_text"]

        # Tách câu trả lời sau marker "Answer:"
        idx = output.find("Answer:")
        if idx != -1:
            answer = output[idx + len("Answer:"):].strip()
        else:
            answer = output.strip()

        # Loại bỏ prompt lặp lại nếu model chèn vào
        answer = answer.replace(prompt, "").strip()
        return answer
    except Exception as e:
        return f"LLM error: {e}"

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
        f"Dựa trên thông tin sau: {context}\n"
        f"Câu hỏi: {query}\n"
        f"Vui lòng chỉ trả lời một câu ngắn gọn, tuyệt đối không giải thích gì thêm.\n"
        f"Answer:"
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
    # Kết nối Qdrant
    client = connect_qdrant(QDRANT_URL, QDRANT_API_KEY)
    create_collection_if_not_exists(client, COLLECTION_NAME, VECTOR_SIZE)

    # Load embedding model
    embedding_model = SentenceTransformer("halong_embedding")

    # Ví dụ dữ liệu
    chunks = [
        "Hà Nội là thủ đô của Việt Nam.",
        "TP. Hồ Chí Minh nổi tiếng với chợ Bến Thành."
    ]
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    # Upload lên Qdrant (không metadata)
    upload_documents_to_qdrant(client, COLLECTION_NAME, chunks, embeddings)

    # Chatbot RAG
    query = "Hà Nội là thủ đô của nước nào?"
    answer, source = chatbot_rag(client, query, embedding_model)
    print("Answer:", answer)
    # print("Source:", source)
