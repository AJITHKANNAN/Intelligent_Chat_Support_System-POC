import os
import pandas as pd
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

logger = get_logger(__name__)
DATA_PATH = os.path.join(os.getcwd(), "data")

def load_csv_files():
    all_docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            path = os.path.join(DATA_PATH, file)
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip().str.lower()

                # Decide which column to use as the "searchable text"
                if "question" in df.columns:
                    text_column = "question"
                    answer_column = "answer"
                elif "query_text" in df.columns:
                    text_column = "query_text"
                    answer_column = None  # no explicit answer, we can store the query itself
                elif "product_name" in df.columns:
                    text_column = "product_name"
                    answer_column = None
                else:
                    # fallback: combine all text columns
                    text_columns = df.select_dtypes(include='object').columns
                    text_column = None

                for _, row in df.iterrows():
                    if text_column:
                        content = str(row[text_column])
                    else:
                        content = " ".join(str(row[col]) for col in text_columns)

                    metadata = {
                        "source": file,
                        "answer": str(row[answer_column]) if answer_column else content,
                        "category": str(row.get("category", "")),
                        "additional_info": row.to_dict()
                    }
                    all_docs.append(Document(page_content=content, metadata=metadata))

            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")
    return all_docs

