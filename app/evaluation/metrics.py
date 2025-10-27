import evaluate
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load metric modules
# -----------------------------
rouge = evaluate.load("rouge")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_rouge(predictions, references):
    """Compute ROUGE-L score between predicted and reference answers."""
    results = rouge.compute(predictions=predictions, references=references)
    return float(results["rougeL"])  # <-- convert NumPy scalar to float

def compute_bleu(predictions, references):
    """Compute BLEU score between predicted and reference answers."""
    smoothie = SmoothingFunction().method4
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie))
    return float(np.mean(scores))

def compute_semantic_similarity(predictions, references):
    """Compute cosine similarity between sentence embeddings."""
    pred_embeddings = embedding_model.encode(predictions, convert_to_numpy=True, normalize_embeddings=True)
    ref_embeddings = embedding_model.encode(references, convert_to_numpy=True, normalize_embeddings=True)
    sims = [cosine_similarity([p], [r])[0][0] for p, r in zip(pred_embeddings, ref_embeddings)]
    return float(np.mean(sims))

def evaluate_response(prediction, reference):
    """Evaluate a single chatbot response."""
    rouge_score = compute_rouge([prediction], [reference])
    bleu_score = compute_bleu([prediction], [reference])
    semantic_sim = compute_semantic_similarity([prediction], [reference])

    return {
        "ROUGE-L": round(rouge_score, 4),
        "BLEU": round(bleu_score, 4),
        "Semantic Similarity": round(semantic_sim, 4),
    }
