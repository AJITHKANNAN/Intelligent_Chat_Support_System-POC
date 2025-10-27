Problem Statement: Intelligent Customer Support System 
You are tasked with designing and implementing an end-to-end intelligent customer support system for an e-commerce platform. The system should automatically categorize customer queries, predict resolution time, and generate contextual responses. 
Dataset Description 
You will work with a simulated customer support dataset. First, run the provided data generation scripts to create your datasets: 
Data Generation 
1. Main Dataset: Run dataset_generator.py to create customer_support_dataset.csv (5,000 records) 2. Supporting Data: Run supporting_data_generator.py to create additional files 
Generated Datasets 
● customer_support_dataset.csv: Main dataset with customer queries, resolution times, satisfaction scores 
● faq_knowledge_base.csv: FAQ entries for RAG implementation 
● historical_interactions.csv: Customer interaction history 
● product_catalog.csv: Product information 
● sample_responses.json: Example responses for LLM evaluation 
Main Dataset Features 
● Customer queries (text): Raw customer messages/complaints 
● Product categories: Electronics, Clothing, Home & Garden, Books, Sports 
● Query types: Billing, Shipping, Returns, Product Info, Technical Support 
● Resolution time (continuous): Time taken to resolve in hours 
● Customer satisfaction score (1-5): Post-resolution rating 
● Customer features: Tier, tenure, order history 
● Temporal features: Time of day, day of week, urgency indicators
Multi-Part Challenge 
Part 1: Classical Machine Learning (30 points) 
Objective: Build a predictive model for resolution time estimation 
Tasks: 
1. Perform comprehensive EDA on numerical and categorical features 
2. Engineer features from text data (TF-IDF, word counts, sentiment scores) 3. Build and compare at least 3 different ML models (e.g., Random Forest, XGBoost, Linear Regression) 
4. Implement proper cross-validation and hyperparameter tuning 
5. Provide feature importance analysis and business insights 
Deliverables: 
● Feature engineering pipeline 
● Model comparison with metrics (MAE, RMSE, R²) 
● Business recommendations based on feature importance 
Part 2: Deep Learning (35 points) 
Objective: Develop a multi-class text classifier for query categorization 
Tasks: 
1. Preprocess text data (tokenization, padding, handling class imbalance) 2. Implement at least 2 different neural network architectures: 
○ CNN-based text classifier 
○ LSTM/GRU-based classifier 
3. Compare with a pre-trained embedding approach (Word2Vec/GloVe) 
4. Implement proper regularization techniques (dropout, early stopping) 
5. Analyze misclassifications and model behavior 
Deliverables: 
● Neural network architectures with justification 
● Training curves and performance metrics 
● Confusion matrix analysis with actionable insights 
Part 3: Large Language Models (35 points) 
Objective: Create an intelligent response generation system
Tasks: 
1. Prompt Engineering: Design prompts for query classification and response generation 2. RAG Implementation: Build a retrieval-augmented generation system using: ○ Vector database for storing FAQ/knowledge base 
○ Embedding-based similarity search 
○ Context-aware response generation 
3. Model Comparison: Compare at least 2 approaches: 
○ Fine-tuned smaller model (e.g., DistilBERT) vs API-based LLM 
○ Zero-shot vs few-shot prompting strategies 
4. Evaluation: Implement both automated metrics (BLEU, ROUGE) and design human evaluation criteria 
Deliverables: 
● RAG system architecture diagram 
● Prompt templates with examples 
● Evaluation framework with sample results 
● Discussion on LLM limitations and mitigation strategies 
Additional Requirements 
Technical Implementation 
● Use Python with appropriate libraries (scikit-learn, TensorFlow/PyTorch, transformers) ● Implement proper logging and error handling 
● Write modular, reusable code with documentation 
● Include unit tests for critical functions 
Business Context 
● Consider scalability for handling 10,000+ daily queries 
● Address potential biases in automated responses 
● Discuss privacy and data security implications 
● Propose A/B testing framework for system evaluation 
Presentation Component 
Prepare a 15-minute presentation covering: 
1. Problem approach and methodology 
2. Key findings and model performance 
3. Business impact and recommendations 
4. Production deployment considerations
5. Future improvements and research directions 
Evaluation Criteria 
Technical Excellence : 
● Code quality and best practices 
● Model performance and methodology 
● Proper use of evaluation metrics 
● Innovation in approach 
Business Acumen : 
● Understanding of business context 
● Practical recommendations 
● Consideration of operational constraints 
● ROI and impact assessment 
Communication 
● Clear documentation and comments 
● Effective visualization of results 
● Quality of presentation 
● Ability to explain complex concepts simply 
Timeline and Deliverables 
Total Time: 5-7 days 
Submit: 
1. Jupyter notebooks with complete analysis 
2. Python scripts for production-ready code 
3. Requirements.txt and setup instructions 
4. Written report (3-5 pages) summarizing findings 
5. Presentation slides 
Bonus Challenges (Optional) 
1. Multi-modal Analysis: Incorporate image data from product-related queries 2. Real-time Processing: Design streaming pipeline for live query processing 3. Explainable AI: Implement LIME/SHAP for model interpretability 4. Active Learning: Propose strategy for continuous model improvement
Success Metrics 
A successful submission should demonstrate: 
● Strong foundation in all three domains (ML/DL/LLM) ● Ability to work with messy, real-world data ● Business-oriented thinking and practical solutions ● Clear communication of technical concepts ● Consideration of production deployment challenges
