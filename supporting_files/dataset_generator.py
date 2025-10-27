import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker # type:ignore
import re

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

fake = Faker()
Faker.seed(42)

# Configuration
n_samples = 5000
n_customers = 1000

# Define categories and their characteristics
product_categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports']
query_types = ['Billing', 'Shipping', 'Returns', 'Product Info', 'Technical Support']

# Query templates for realistic text generation
query_templates = {
    'Billing': [
        "I was charged twice for my order #{order_id}. Can you help me get a refund?",
        "My credit card shows a charge of ${amount} but I only ordered items worth ${real_amount}",
        "I need to update my billing address for order #{order_id}",
        "Why was I charged tax on my order when I live in {state}?",
        "I need a receipt for my purchase #{order_id} for tax purposes"
    ],
    'Shipping': [
        "My order #{order_id} was supposed to arrive yesterday but I haven't received it",
        "Can you expedite shipping for order #{order_id}? I need it urgently",
        "The tracking shows my package is stuck in {city} for 3 days",
        "I need to change the delivery address for order #{order_id}",
        "My package was delivered but I wasn't home. Where is it now?"
    ],
    'Returns': [
        "I want to return the {product} I ordered. It doesn't fit properly",
        "The {product} arrived damaged. How do I return it?",
        "I ordered {product} but received {wrong_product} instead",
        "Can I return this {product} even though it's been 40 days?",
        "I lost my receipt but want to return this {product}"
    ],
    'Product Info': [
        "What are the dimensions of the {product}?",
        "Is the {product} compatible with {other_product}?",
        "When will the {product} be back in stock?",
        "Can you tell me more about the warranty for {product}?",
        "What's the difference between {product} and {similar_product}?"
    ],
    'Technical Support': [
        "My {product} stopped working after 2 weeks. Can you help?",
        "I can't connect my {product} to WiFi. What should I do?",
        "The {product} makes a strange noise when I use it",
        "How do I reset my {product} to factory settings?",
        "The {product} app keeps crashing on my phone"
    ]
}

# Product examples by category
products_by_category = {
    'Electronics': ['iPhone', 'MacBook', 'Samsung TV', 'Headphones', 'Gaming Console', 'Tablet', 'Smart Watch'],
    'Clothing': ['Jeans', 'T-shirt', 'Dress', 'Sneakers', 'Jacket', 'Sweater', 'Boots'],
    'Home & Garden': ['Coffee Maker', 'Vacuum Cleaner', 'Garden Hose', 'Lawn Mower', 'Dining Table', 'Couch'],
    'Books': ['Programming Book', 'Novel', 'Cookbook', 'Biography', 'Textbook', 'Comic Book'],
    'Sports': ['Running Shoes', 'Yoga Mat', 'Basketball', 'Tennis Racket', 'Dumbbells', 'Bike Helmet']
}

def generate_customer_query(query_type, product_category):
    """Generate realistic customer query text"""
    template = random.choice(query_templates[query_type])
    product = random.choice(products_by_category[product_category])
    
    # Replace placeholders
    template = template.replace('{product}', product)
    template = template.replace('{order_id}', f"ORD{random.randint(100000, 999999)}")
    template = template.replace('{amount}', f"{random.randint(50, 500):.2f}")
    template = template.replace('{real_amount}', f"{random.randint(25, 250):.2f}")
    template = template.replace('{state}', fake.state())
    template = template.replace('{city}', fake.city())
    template = template.replace('{other_product}', random.choice(products_by_category[product_category]))
    template = template.replace('{wrong_product}', random.choice([item for cat in products_by_category.values() for item in cat]))
    template = template.replace('{similar_product}', random.choice(products_by_category[product_category]))
    
    return template

def calculate_resolution_time(query_type, product_category, customer_tier, query_complexity):
    """Calculate resolution time based on various factors"""
    base_times = {
        'Billing': 2.5,
        'Shipping': 1.8,
        'Returns': 3.2,
        'Product Info': 1.2,
        'Technical Support': 4.5
    }
    
    category_multipliers = {
        'Electronics': 1.3,
        'Clothing': 0.8,
        'Home & Garden': 1.1,
        'Books': 0.7,
        'Sports': 0.9
    }
    
    tier_multipliers = {
        'Bronze': 1.2,
        'Silver': 1.0,
        'Gold': 0.7,
        'Platinum': 0.5
    }
    
    base_time = base_times[query_type]
    time = base_time * category_multipliers[product_category] * tier_multipliers[customer_tier]
    
    # Add complexity factor
    time *= (1 + query_complexity * 0.5)
    
    # Add some randomness
    time *= np.random.normal(1, 0.3)
    
    return max(0.5, time)  # Minimum 30 minutes

def calculate_satisfaction_score(resolution_time, query_type, customer_tier):
    """Calculate customer satisfaction based on resolution time and other factors"""
    # Base satisfaction scores
    base_scores = {
        'Billing': 3.5,
        'Shipping': 4.0,
        'Returns': 3.2,
        'Product Info': 4.2,
        'Technical Support': 3.0
    }
    
    tier_bonus = {
        'Bronze': 0,
        'Silver': 0.2,
        'Gold': 0.4,
        'Platinum': 0.6
    }
    
    base_score = base_scores[query_type] + tier_bonus[customer_tier]
    
    # Penalize long resolution times
    if resolution_time > 6:
        base_score -= 1.0
    elif resolution_time > 3:
        base_score -= 0.5
    elif resolution_time < 1:
        base_score += 0.5
    
    # Add randomness
    score = base_score + np.random.normal(0, 0.5)
    
    return max(1, min(5, round(score, 1)))

# Generate customer data
customers = []
for i in range(n_customers):
    customers.append({
        'customer_id': f'CUST_{i+1:04d}',
        'customer_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                        p=[0.4, 0.35, 0.2, 0.05]),
        'tenure_months': np.random.exponential(24),
        'total_orders': np.random.poisson(12),
        'avg_order_value': np.random.normal(150, 50)
    })

customer_df = pd.DataFrame(customers)

# Generate main dataset
data = []
for i in range(n_samples):
    # Select random customer
    customer = customer_df.iloc[i % len(customer_df)]
    
    # Generate query characteristics
    product_category = np.random.choice(product_categories)
    query_type = np.random.choice(query_types)
    
    # Query complexity (0-1 scale)
    query_complexity = np.random.beta(2, 5)  # Skewed towards simpler queries
    
    # Generate query text
    query_text = generate_customer_query(query_type, product_category)
    
    # Calculate resolution time
    resolution_time = calculate_resolution_time(
        query_type, product_category, customer['customer_tier'], query_complexity
    )
    
    # Calculate satisfaction score
    satisfaction_score = calculate_satisfaction_score(
        resolution_time, query_type, customer['customer_tier']
    )
    
    # Additional features
    query_length = len(query_text.split())
    has_order_id = 'ORD' in query_text
    urgency_keywords = ['urgent', 'asap', 'immediately', 'quickly', 'fast']
    is_urgent = any(keyword in query_text.lower() for keyword in urgency_keywords)
    
    # Time-based features
    created_date = fake.date_time_between(start_date='-1y', end_date='now')
    hour_of_day = created_date.hour
    day_of_week = created_date.weekday()
    is_weekend = day_of_week >= 5
    
    data.append({
        'query_id': f'QRY_{i+1:06d}',
        'customer_id': customer['customer_id'],
        'query_text': query_text,
        'product_category': product_category,
        'query_type': query_type,
        'resolution_time_hours': round(resolution_time, 2),
        'customer_satisfaction': satisfaction_score,
        'customer_tier': customer['customer_tier'],
        'customer_tenure_months': round(customer['tenure_months'], 1),
        'customer_total_orders': customer['total_orders'],
        'customer_avg_order_value': round(customer['avg_order_value'], 2),
        'query_length_words': query_length,
        'has_order_reference': has_order_id,
        'is_urgent': is_urgent,
        'created_datetime': created_date,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'query_complexity_score': round(query_complexity, 3)
    })

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values to make it realistic
missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[missing_indices[:len(missing_indices)//2], 'customer_avg_order_value'] = np.nan
df.loc[missing_indices[len(missing_indices)//2:], 'customer_satisfaction'] = np.nan

# Save to CSV
df.to_csv('customer_support_dataset.csv', index=False)

print(f"Dataset generated with {len(df)} records")
print(f"Columns: {list(df.columns)}")
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget variable (resolution_time_hours) statistics:")
print(df['resolution_time_hours'].describe())
print("\nQuery type distribution:")
print(df['query_type'].value_counts())
print("\nProduct category distribution:")
print(df['product_category'].value_counts())