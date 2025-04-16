# Fake News Detection Model

## Overview

This repository contains an optimized fake news detection model that combines advanced natural language processing (NLP) techniques and state-of-the-art transformer models to accurately classify news articles as real or fake.

The model leverages a hybrid architecture combining:

- Traditional machine learning ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Transformer-based deep learning (BERT fine-tuning)
- Novel feature engineering including sentiment analysis, named entity recognition, and text complexity metrics

The model achieves high accuracy and provides explainability using SHAP and LIME.

---

## Dataset

The dataset used in this project is sourced from Kaggle:

**Fake News Detection Dataset**  
[https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

- Contains approximately 45,000 news articles
- Includes two CSV files: `True.csv` (real news) and `Fake.csv` (fake news)
- Each article has columns such as `title`, `text`, `subject`, `date`
- Labels are assigned as `1` for real news and `0` for fake news

---

## Data Preprocessing

- Text cleaning: Lowercasing, URL removal, HTML tag stripping, special character removal
- Tokenization and lemmatization using NLTK
- Stopword removal and filtering out short tokens
- Feature extraction:
  - Sentiment polarity using TextBlob
  - Named entity count using spaCy
  - Text complexity features: average sentence length and lexical diversity
- TF-IDF vectorization with unigrams and bigrams

---

## Model Architecture

- **Ensemble Learning**: Combines predictions from multiple traditional ML models (Random Forest, Gradient Boosting, XGBoost, LightGBM) using a logistic regression meta-learner
- **Transformer Model**: Fine-tuned BERT (`bert-base-uncased`) for contextual text understanding
- **Hybrid Model**: Weighted combination of ensemble and BERT predictions for improved accuracy

---

## Training Details

- Dataset split into training and testing sets with stratification
- Class imbalance handled using SMOTE oversampling
- BERT fine-tuned for 4 epochs on an NVIDIA A100 GPU
- Optimizer: AdamW with learning rate scheduling
- Gradient clipping applied to stabilize training

---

## Evaluation

- Metrics reported: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix and ROC curve visualizations
- Explainability using:
  - SHAP for global feature importance
  - LIME for local instance-level explanations

---

## Usage

- Predict fake or real news for new articles using the `predict_news()` function, which preprocesses input text and combines ensemble and BERT model predictions
- Visualize model decisions and explanations with SHAP and LIME

---

## Requirements

- Python 3.7+
- Libraries: pandas, numpy, scikit-learn, nltk, spacy, textblob, transformers, torch, shap, lime, imblearn, matplotlib, seaborn

---

## How to Run

1. Clone the repository
2. Download the dataset from Kaggle and place `True.csv` and `Fake.csv` in the `data/` directory
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the notebook or script to preprocess data, train models, and evaluate performance

---

## References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Fake News Detection Dataset on Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- SHAP and LIME documentation for model explainability

