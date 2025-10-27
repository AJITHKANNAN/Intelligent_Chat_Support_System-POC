import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Generate FAQ/Knowledge Base for RAG (Part 3)
faq_data = [
    {
        "question": "How do I track my order?",
        "answer": "You can track your order by logging into your account and visiting the 'My Orders' section. You'll find tracking information and estimated delivery dates there.",
        "category": "Shipping",
        "keywords": ["track", "order", "delivery", "shipping"]
    },
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy for most items. Items must be in original condition with tags attached. Electronics have a 15-day return window.",
        "category": "Returns",
        "keywords": ["return", "policy", "refund", "exchange"]
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. Free shipping is available on orders over $50.",
        "category": "Shipping",
        "keywords": ["shipping", "delivery", "time", "express", "standard"]
    },
    {
        "question": "Can I cancel my order?",
        "answer": "Orders can be cancelled within 1 hour of placement. After that, the order is processed and cannot be cancelled, but you can return items once delivered.",
        "category": "Billing",
        "keywords": ["cancel", "order", "refund"]
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, Google Pay, and gift cards.",
        "category": "Billing",
        "keywords": ["payment", "credit card", "paypal", "billing"]
    },
    {
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page, enter your email address, and we'll send you a reset link. The link expires in 24 hours.",
        "category": "Technical Support",
        "keywords": ["password", "reset", "login", "account"]
    },
    {
        "question": "Do you offer warranty on electronics?",
        "answer": "Yes, all electronics come with manufacturer warranty. Extended warranty options are available at checkout for most electronic items.",
        "category": "Product Info",
        "keywords": ["warranty", "electronics", "protection", "coverage"]
    },
    {
        "question": "How do I contact customer support?",
        "answer": "You can reach us via live chat (available 24/7), email at support@company.com, or phone at 1-800-SUPPORT during business hours (9 AM - 6 PM EST).",
        "category": "Product Info",
        "keywords": ["contact", "support", "help", "phone", "email", "chat"]
    },
    {
        "question": "Can I change my shipping address?",
        "answer": "You can change your shipping address within 1 hour of placing the order. After that, contact customer support - we may be able to help if the order hasn't shipped yet.",
        "category": "Shipping",
        "keywords": ["change", "address", "shipping", "delivery"]
    },
    {
        "question": "What if my item arrives damaged?",
        "answer": "If your item arrives damaged, please contact us within 48 hours with photos of the damage. We'll provide a prepaid return label and send a replacement immediately.",
        "category": "Returns",
        "keywords": ["damaged", "broken", "defective", "replacement"]
    },
    {
        "question": "Do you price match?",
        "answer": "Yes, we offer price matching for identical items from major competitors. The item must be in stock and the competitor's price must be verifiable.",
        "category": "Billing",
        "keywords": ["price", "match", "competitor", "discount"]
    },
    {
        "question": "How do I apply a discount code?",
        "answer": "Enter your discount code in the 'Promo Code' field during checkout. The discount will be applied before you complete your purchase.",
        "category": "Billing",
        "keywords": ["discount", "promo", "code", "coupon"]
    },
    {
        "question": "What are your store hours?",
        "answer": "Our online store is available 24/7. Physical store locations vary - please check the store locator on our website for specific hours.",
        "category": "Product Info",
        "keywords": ["hours", "store", "location", "open"]
    },
    {
        "question": "Do you ship internationally?",
        "answer": "We currently ship to the US, Canada, and select European countries. International shipping rates and delivery times vary by destination.",
        "category": "Shipping",
        "keywords": ["international", "shipping", "global", "worldwide"]
    },
    {
        "question": "How do I leave a product review?",
        "answer": "After your order is delivered, you'll receive an email invitation to review your purchased items. You can also leave reviews by visiting the product page.",
        "category": "Product Info",
        "keywords": ["review", "rating", "feedback", "product"]
    }
]

faq_df = pd.DataFrame(faq_data)
faq_df.to_csv('faq_knowledge_base.csv', index=False)

# Generate historical interaction data
np.random.seed(42)
historical_interactions = []

for i in range(2000):  # Generate 2000 historical interactions
    customer_id = f'CUST_{np.random.randint(1, 1001):04d}'
    interaction_date = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
    
    historical_interactions.append({
        'interaction_id': f'INT_{i+1:06d}',
        'customer_id': customer_id,
        'interaction_date': interaction_date,
        'channel': np.random.choice(['Email', 'Chat', 'Phone'], p=[0.4, 0.4, 0.2]),
        'resolution_status': np.random.choice(['Resolved', 'Escalated', 'Pending'], p=[0.8, 0.15, 0.05]),
        'agent_id': f'AGENT_{np.random.randint(1, 21):02d}',
        'interaction_duration_minutes': np.random.exponential(15),
        'follow_up_required': np.random.choice([True, False], p=[0.2, 0.8])
    })

historical_df = pd.DataFrame(historical_interactions)
historical_df.to_csv('historical_interactions.csv', index=False)

# Generate product catalog for context
products_data = []
product_id = 1

for category, products in {
    'Electronics': ['iPhone 15', 'MacBook Pro', 'Samsung 65" TV', 'Sony Headphones', 'PS5 Console', 'iPad Air', 'Apple Watch'],
    'Clothing': ['Levi\'s Jeans', 'Nike T-shirt', 'Summer Dress', 'Adidas Sneakers', 'Winter Jacket', 'Wool Sweater', 'Hiking Boots'],
    'Home & Garden': ['Keurig Coffee Maker', 'Dyson Vacuum', 'Garden Hose 50ft', 'Electric Lawn Mower', 'Dining Table Set', 'Sectional Sofa'],
    'Books': ['Python Programming Guide', 'Mystery Novel', 'Italian Cookbook', 'Steve Jobs Biography', 'Data Science Textbook', 'Marvel Comics'],
    'Sports': ['Running Shoes', 'Yoga Mat Premium', 'Wilson Basketball', 'Tennis Racket Pro', 'Adjustable Dumbbells', 'Bike Helmet']
}.items():
    for product in products:
        products_data.append({
            'product_id': f'PROD_{product_id:05d}',
            'product_name': product,
            'category': category,
            'price': np.random.uniform(25, 1200),
            'rating': np.random.uniform(3.5, 5.0),
            'num_reviews': np.random.randint(10, 1000),
            'in_stock': np.random.choice([True, False], p=[0.9, 0.1]),
            'warranty_months': np.random.choice([0, 12, 24, 36], p=[0.3, 0.4, 0.2, 0.1])
        })
        product_id += 1

products_df = pd.DataFrame(products_data)
products_df.to_csv('product_catalog.csv', index=False)

# Generate sample responses for LLM evaluation
sample_responses = [
    {
        'query': "My order hasn't arrived and it's been 5 days",
        'human_response': "I apologize for the delay with your order. Let me track that for you right away. I can see your order is currently with our shipping partner and should arrive within the next 24 hours. I'll also apply a $10 credit to your account for the inconvenience.",
        'llm_response_1': "I understand your concern about the delayed delivery. Please provide your order number so I can track its current status and provide you with an accurate update.",
        'llm_response_2': "Sorry for the delay! Your package might be stuck in transit. Here's what we can do: 1) Track your order, 2) Contact shipping partner, 3) Provide compensation if needed. Can you share your order number?"
    },
    {
        'query': "I want to return this jacket, it doesn't fit",
        'human_response': "I'd be happy to help you with that return. Since clothing items can be returned within 30 days, you're well within our return window. I'll email you a prepaid return label, and once we receive the item, we'll process your refund within 3-5 business days.",
        'llm_response_1': "No problem! Our return policy allows 30 days for clothing returns. I can generate a return label for you - would you prefer a refund or exchange for a different size?",
        'llm_response_2': "Returns are easy! Just pack the jacket in its original packaging, use our prepaid label, and drop it off at any authorized location. Refund typically takes 5-7 days."
    }
]

with open('sample_responses.json', 'w') as f:
    json.dump(sample_responses, f, indent=2)

print("Supporting datasets generated:")
print(f"1. FAQ Knowledge Base: {len(faq_df)} entries")
print(f"2. Historical Interactions: {len(historical_df)} records")
print(f"3. Product Catalog: {len(products_df)} products")
print(f"4. Sample Responses: {len(sample_responses)} examples")

print("\nFAQ Categories:")
print(faq_df['category'].value_counts())
print("\nProduct Categories:")
print(products_df['category'].value_counts())
print("\nHistorical Interaction Channels:")
print(historical_df['channel'].value_counts())