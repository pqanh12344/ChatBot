import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import gc
from tqdm import tqdm
from config import logger, MODEL_PATH, CHUNK_SIZE, BATCH_SIZE
from typing import List, Tuple


@st.cache_data
def create_embeddings(chunks: List[str], model_path: str = MODEL_PATH, chunk_size: int = CHUNK_SIZE, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Tạo embeddings cho các chunk văn bản."""
    try:
        model = SentenceTransformer(model_path, device='cpu')
        gc.collect()
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

def load_model(model_path: str = MODEL_PATH) -> SentenceTransformer:
    """Tải mô hình SentenceTransformer."""
    try:
        model = SentenceTransformer(model_path, device='cpu')
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise