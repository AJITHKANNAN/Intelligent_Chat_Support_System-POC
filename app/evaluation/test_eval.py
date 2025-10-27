
from app.evaluation.metrics import evaluate_response
# Example
pred = "Your order will be delivered in 3 to 5 business days."
ref = "The estimated delivery time is between 3 to 5 working days."

results = evaluate_response(pred, ref)
print("Evaluation Results:", results)
