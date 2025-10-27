
# app/streamlit_chatbot.py
import os
import streamlit as st
import faiss
import numpy as np
import ast, pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from app.config.config import DB_FAISS_PATH
from app.components.llm import groq_load_llm
from app.common.logger import get_logger
from app.evaluation.metrics import evaluate_response


# Explicitly load your intended .env file and override any existing key
dotenv_path = r"E:\Anubavam_task\24Oct\.env"
load_dotenv(dotenv_path=dotenv_path, override=True)

# Force set the key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
print("✅ Using GROQ_API_KEY:", GROQ_API_KEY[:6], "******") 


# -----------------------------
# Initialize logger
# -----------------------------
logger = get_logger(__name__)
logger.info("🚀 Starting Streamlit chatbot...")

# -----------------------------
# Load environment variables
# -----------------------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please check your .env file.")
    logger.error("❌ GROQ_API_KEY not found. Check your .env file.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
logger.info(f"✅ GROQ_API_KEY loaded (first 6 chars): {GROQ_API_KEY[:6]}******")
st.success("✅ GROQ_API_KEY loaded successfully.")

from transformers import pipeline
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
logger.info("✅ DistilBERT QA model loaded.")


# def small_model_response(user_input: str, retrieved_docs: list) -> str:
#     context_combined = " ".join([doc.get("answer") or doc.get("content") for doc in retrieved_docs])
#     result = qa_model(question=user_input, context=context_combined)
#     return result["answer"]


# -----------------------------
# Query classification and keyword ranking

# Load the FAQ CSV once globally
FAQ_DF = pd.read_csv(r"E:\Anubavam_task\24Oct\data\faq_knowledge_base.csv")

def classify_query(user_input: str) -> str:
    """
    Classify the user query into a category based on keywords from the FAQ.
    Returns 'General' if no match is found.
    """
    for _, row in FAQ_DF.iterrows():
        keywords = row["keywords"]
        # If keywords is a string (e.g., "['word1','word2']"), convert to list
        if isinstance(keywords, str):
            try:
                keywords = ast.literal_eval(keywords)
            except Exception:
                # fallback: split by comma if literal_eval fails
                keywords = [k.strip().strip("'\"") for k in keywords.strip("[]").split(",")]

        # Check if any keyword is in user input (case-insensitive)
        if any(kw.lower() in user_input.lower() for kw in keywords):
            return row["category"]

    # Default category if no keyword matched
    return "General"


def rank_docs_by_keywords(user_input: str, docs: list) -> list:
    """Rank retrieved docs by number of keyword matches."""
    user_tokens = set(user_input.lower().split())
    scored_docs = []
    for d in docs:
        kw_set = set([k.lower() for k in d.get("keywords", [])])
        score = len(user_tokens & kw_set)
        scored_docs.append((score, d))
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [d for score, d in scored_docs if score > 0]

# -----------------------------
# Load FAISS index and metadata
# -----------------------------
try:
    index = faiss.read_index(f"{DB_FAISS_PATH}.index")
    metadata = np.load(f"{DB_FAISS_PATH}_metadata.npy", allow_pickle=True).tolist()
    logger.info(f"✅ FAISS database loaded from {DB_FAISS_PATH}")
except Exception as e:
    st.error(f"❌ Failed to load FAISS database: {e}")
    logger.exception("❌ Failed to load FAISS database.")
    st.stop()

# -----------------------------
# Load embedding model
# -----------------------------
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("✅ Embedding model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load embedding model: {e}")
    logger.exception("❌ Failed to load embedding model.")
    st.stop()

# -----------------------------
# Initialize LLM once (no session memory)
# -----------------------------
llm = groq_load_llm(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
if not llm:
    st.error("❌ Failed to load Groq LLM. Chatbot will only use raw data retrieval.")
    logger.error("❌ LLM initialization returned None.")
else:
    logger.info("✅ LLM loaded successfully.")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🛍️ Customer Support Chatbot", layout="centered")
st.title("🛍️ Customer Support Assistant")
st.write("Hi there! I'm your virtual assistant. Ask me anything about your orders, shipping, products, or policies.")

# -----------------------------
# Session state for chat history
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Retrieve top-k relevant documents from FAISS
# -----------------------------
def retrieve_relevant_docs(query: str, top_k: int = 3):
    try:
        query_vector = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = index.search(query_vector, k=top_k)
        results = [metadata[i] for i in indices[0] if i < len(metadata)]
        return results
    except Exception as e:
        logger.exception("Error retrieving vectors")
        return []



# -----------------------------
# Generate human-like response
# -----------------------------

def generate_response(user_input: str) -> dict:
    logger.info(f"🧠 User input: {user_input}")

    response_text = ""
    eval_metrics = None
    relevant_docs = retrieve_relevant_docs(user_input, top_k=2)
    q_category = classify_query(user_input)

    retrieved_info = "\n".join([f"- {doc.get('answer') or doc.get('content')}" for doc in relevant_docs if doc])
    if not retrieved_info:
        retrieved_info = "No matching FAQ found."

    logger.info(f"📄 Retrieved info from FAISS:\n{retrieved_info}")

    context = "\n".join([f"User: {c['user']}\nBot: {c.get('bot', '')}" for c in st.session_state.chat_history[-3:]])
    prompt = f"""
            You are a friendly, human-like customer support assistant.
            Respond naturally using simple language, not robotic.

            Relevant Info from Documents:
            {retrieved_info}

            Recent Conversation:
            {context}

            User: {user_input}
            Bot:
"""

    try:
        if not llm:
            logger.warning("❌ LLM not loaded. Returning retrieved info only.")
            response_text = retrieved_info
        else:
            logger.info("🚀 Sending prompt to Groq...")
            llm_response = llm.invoke(prompt)
            if hasattr(llm_response, "content"):
                response_text = llm_response.content
            elif isinstance(llm_response, str):
                response_text = llm_response
            else:
                response_text = str(llm_response)

            # Evaluate response
            if relevant_docs and relevant_docs[0].get("answer"):
                reference_answer = relevant_docs[0]["answer"]
                eval_metrics = evaluate_response(response_text, reference_answer)
                logger.info(f"📊 Evaluation: {eval_metrics}")

    except Exception as e:
        logger.exception(f"❌ LLM error: {e}")
        response_text = f"Sorry, something went wrong while generating a response: {e}"

    # ✅ Always build output below (outside try/except)
    formatted_output = f"""
            💬 **Chatbot Reply (Customer View)**

            **Bot:**  
            {response_text.strip()}

            ---

            **🗂️ Category:** {q_category}
            """

    if retrieved_info != "No matching FAQ found.":
        formatted_output += f"""**📘 Based on our policy:**  
                    {relevant_docs[0]['answer']}

                    ---
"""

    if eval_metrics:
        formatted_output += f"""**📊 Response Quality (Internal Use):**  
            - ROUGE-L: {eval_metrics.get('ROUGE-L', 0):.2f}  
            - BLEU: {eval_metrics.get('BLEU', 0):.2f}  
            - Semantic Similarity: {eval_metrics.get('Semantic Similarity', 0):.2f}
            """

    return {
        "response": response_text.strip(),
        "category": q_category,
        "top_docs": relevant_docs[:3],
        "eval_metrics": eval_metrics,
        "formatted": formatted_output.strip()
    }



def small_model_response(user_input: str, retrieved_docs: list, few_shot: bool = False) -> dict:
    # Prepare context
    context_docs = " ".join([doc.get("answer") or doc.get("content") for doc in retrieved_docs])
    
    if few_shot:
        few_shot_examples = """
         You are a friendly, human-like customer support assistant.
         Respond naturally using simple language, not robotic.

Q: What is the return policy for electronics?
A: Electronics can be returned within 30 days with receipt.

Q: Can I get a refund if I received a defective product?
A: Yes, you can request a refund within 14 days of delivery.
"""
        context_combined = few_shot_examples + "\n" + context_docs
    else:
        context_combined = context_docs
    
    result = qa_model(question=user_input, context=context_combined)
    answer_text = result["answer"]

    # Evaluate against first retrieved doc
    eval_metrics = None
    if retrieved_docs and retrieved_docs[0].get("answer"):
        eval_metrics = evaluate_response(answer_text, retrieved_docs[0]["answer"])

    # Prepare formatted output
    policy_text = retrieved_docs[0]["answer"] if retrieved_docs else ""
    formatted_output = f"""
💬 **Chatbot Reply (Customer View)**

**Bot:**  
{answer_text.strip() if answer_text else policy_text.strip()}

---  
**🗂️ Category:** {classify_query(user_input)}
"""
    if policy_text:
        formatted_output += f"\n **📘 Based on our policy:** \n{policy_text}\n---\n"

    if eval_metrics:
        formatted_output += f"""**📊 Response Quality (Internal Use):**  
- ROUGE-L: {eval_metrics.get('ROUGE-L', 0):.2f}  
- BLEU: {eval_metrics.get('BLEU', 0):.2f}  
- Semantic Similarity: {eval_metrics.get('Semantic Similarity', 0):.2f}
"""

    return {
        "response": answer_text.strip() if answer_text else policy_text.strip(),
        "category": classify_query(user_input),
        "top_docs": retrieved_docs[:3],
        "eval_metrics": eval_metrics,
        "formatted": formatted_output.strip(),
    }




def respond(user_input: str, use_llm: bool = True, few_shot: bool = False):
    docs = retrieve_relevant_docs(user_input, top_k=3)
    q_category = classify_query(user_input)

    if use_llm:
        return generate_response(user_input)
    else:
        return small_model_response(user_input, docs, few_shot=few_shot)





# -----------------------------
# Model Selection Toggle
# -----------------------------
st.sidebar.title("⚙️ Chatbot Settings")

use_small_model = st.sidebar.toggle("Use lightweight model (DistilBERT)", value=False)
st.sidebar.caption("Switch between Groq LLM and small QA model.")

# Zero-shot / Few-shot toggle (only applies if small model is selected)
if use_small_model:
    few_shot_mode = st.sidebar.radio(
        "Prompting Mode:",
        options=["Zero-shot", "Few-shot"],
        index=0
    )
    st.sidebar.caption("Few-shot mode uses example Q&A in context for better responses.")
else:
    few_shot_mode = False  # Not used for Groq LLM

# -----------------------------
# Chat input area
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Generate model response
    result = respond(
        user_input, 
        use_llm=not use_small_model, 
        few_shot=(few_shot_mode == "Few-shot") if use_small_model else False
    )

    # ✅ Save to session history
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": result.get("formatted", result.get("response", ""))
    })


# -----------------------------
# Display chat history
# -----------------------------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['user']}")
    with st.chat_message("assistant"):
        bot_msg = chat.get("bot", "")
        st.markdown(bot_msg.strip() if isinstance(bot_msg, str) else "_No response generated._")

