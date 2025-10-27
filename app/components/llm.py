# app/components/llm.py
from langchain_groq import ChatGroq
import os
from app.common.logger import get_logger

logger = get_logger(__name__)

def groq_load_llm(model_name="llama-3.1-8b-instant", groq_api_key=None):
    try:
        key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not key:
            logger.error("❌ No GROQ_API_KEY provided or found in environment.")
            return None

        os.environ["GROQ_API_KEY"] = key
        logger.info(f"🔑 Using GROQ_API_KEY (first 6 chars): {key[:6]}******")

        llm = ChatGroq(
            model_name=model_name,
            temperature=0.3,
            groq_api_key=key
        )

        # quick sanity check
        try:
            _ = llm.invoke("ping")
        except Exception as inner_e:
            if "invalid_api_key" in str(inner_e).lower():
                logger.error("🔒 GROQ_API_KEY invalid or expired.")
                return None

        logger.info(f"✅ Groq model loaded successfully ({model_name})")
        return llm

    except Exception as e:
        logger.exception(f"❌ Failed to load Groq LLM: {e}")
        return None


