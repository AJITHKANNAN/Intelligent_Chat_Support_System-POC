# app/components/vector_store.py
import os
import faiss
import numpy as np
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

# ------------------------
# Save vector store
# ------------------------
def save_vector_store(documents):
    try:
        if not documents:
            raise CustomException("No documents to index")

        embedding_model = get_embedding_model()
        logger.info("Generating embeddings for vector store...")

        # Include keywords in embedding for better semantic matches
        texts = [
            f"{doc.page_content} {' '.join(doc.metadata.get('keywords', []))}"
            for doc in documents
        ]

        # Normalize for cosine similarity
        embeddings = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Ensure folder exists
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

        # Save index + metadata
        metadata = [
            {
                "source": doc.metadata.get("source", ""),
                "answer": doc.metadata.get("answer", ""),
                "content": doc.page_content,
                **doc.metadata
            }
            for doc in documents
        ]

        faiss.write_index(index, DB_FAISS_PATH + ".index")
        np.save(DB_FAISS_PATH + "_metadata.npy", metadata, allow_pickle=True)

        logger.info(f"Vector store saved successfully at {DB_FAISS_PATH} with {len(documents)} documents.")
        return index, metadata

    except Exception as e:
        error_message = CustomException("Failed to create vector store", e)
        logger.error(str(error_message))
        raise error_message


# ------------------------
# Load vector store
# ------------------------
def load_vector_store():
    try:
        if not os.path.exists(DB_FAISS_PATH + ".index") or not os.path.exists(DB_FAISS_PATH + "_metadata.npy"):
            logger.warning("Vector store not found.")
            return None, None

        index = faiss.read_index(DB_FAISS_PATH + ".index")
        metadata = np.load(DB_FAISS_PATH + "_metadata.npy", allow_pickle=True).tolist()
        logger.info(f"Vector store loaded successfully from {DB_FAISS_PATH} with {len(metadata)} documents.")
        return index, metadata

    except Exception as e:
        error_message = CustomException("Failed to load vector store", e)
        logger.error(str(error_message))
        return None, None


# app/components/vector_store.py (or separate helper)
def get_relevant_answer(query, index, metadata, embedding_model):
    """
    Retrieve the single most relevant answer for a query.
    """
    try:
        # Embed user query
        query_vector = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Search top-1 vector
        distances, indices = index.search(query_vector, k=1)
        top_idx = indices[0][0]

        # Fetch corresponding answer
        top_doc = metadata[top_idx]
        return top_doc.get("answer") or "Sorry, I couldn't find an answer."

    except Exception as e:
        return f"Error retrieving answer: {e}"
