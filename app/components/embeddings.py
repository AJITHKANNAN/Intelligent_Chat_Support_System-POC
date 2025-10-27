# app/components/embeddings.py
from sentence_transformers import SentenceTransformer
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight Transformer

def get_embedding_model():
    try:
        logger.info(f"Loading Transformer embedding model: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
        return model
    except Exception as e:
        error_message = CustomException("Failed to load embedding model", e)
        logger.error(str(error_message))
        raise error_message
