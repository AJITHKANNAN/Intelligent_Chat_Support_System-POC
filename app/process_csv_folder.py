import os
import pandas as pd
from app.components.vector_store import save_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

logger = get_logger(__name__)

DATA_FOLDER = r"E:\Anubavam_task\24Oct\data"  



def detect_text_column(df):
    """Detect the most likely text column (question, query_text, etc.)."""
    candidates = [
        "question", "query_text", "query", "description", "product_name", "review", "comment", "text"
    ]
    for col in df.columns:
        if any(keyword in col for keyword in candidates):
            return col
    # fallback: pick first object/text column
    text_cols = df.select_dtypes(include='object').columns
    return text_cols[0] if len(text_cols) > 0 else None


def detect_answer_column(df):
    """Detect an answer/response column if available."""
    candidates = ["answer", "response", "resolution_status", "category", "product_category"]
    for col in df.columns:
        if any(keyword in col for keyword in candidates):
            return col
    return None


def load_csv_files():
    all_docs = []
    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(DATA_FOLDER, file)
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.lower()

            text_col = detect_text_column(df)
            answer_col = detect_answer_column(df)

            if not text_col:
                logger.warning(f"No text-like column found in {file}. Skipping.")
                continue

            for _, row in df.iterrows():
                content = str(row.get(text_col, "")).strip()
                if not content:
                    continue

                answer_text = str(row.get(answer_col, "")) if answer_col else ""
                metadata = {
                    "source": file,
                    "answer": answer_text,
                    "extra_info": {
                        col: str(row[col]) for col in df.columns if col not in [text_col, answer_col]
                    }
                }

                all_docs.append(Document(page_content=content, metadata=metadata))

            logger.info(f"✅ Processed {len(df)} rows from {file}")

        except Exception as e:
            logger.error(f"❌ Failed to read {file}: {e}")

    return all_docs


def create_text_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks."""
    chunks = []
    for doc in documents:
        text = doc.page_content
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
            start += chunk_size - chunk_overlap
    return chunks


def process_and_store_csvs():
    try:
        logger.info("Processing CSV files into vector store...")
        documents = load_csv_files()
        if not documents:
            raise CustomException("No CSV documents found in the folder.")

        text_chunks = create_text_chunks(documents)
        save_vector_store(text_chunks)
        logger.info("✅ Vector store created successfully.")

    except Exception as e:
        error_message = CustomException("Failed to process CSV folder", e)
        logger.error(str(error_message))


if __name__ == "__main__":
    process_and_store_csvs()
    documents = load_csv_files()
    print(f"Documents loaded: {len(documents)}")

    text_chunks = create_text_chunks(documents)
    print(f"Chunks created: {len(text_chunks)}")
