# Sentiment_Analysis_on_Customer_Reviews_Using_NLP
Natural Language Processing (NLP) Text Preprocessing Feature Extraction (TF-IDF, Word2Vec, GloVe) Deep Learning for NLP (RNNs, LSTMs, Transformers) Model Evaluation and Optimization
# Sentiment Analysis on Customer Reviews (Amazon)

Project: NLP pipeline to classify Amazon customer reviews into **Positive / Neutral / Negative**.

**Contents**
- `data/` — raw / processed datasets
- `scripts/`
  - `1_data_preprocessing.py`
  - `2_feature_extraction.py`
  - `3_model_training.py`
  - `4_prediction.py`
  - `5_eval_report.md`
- `notebooks/` — Colab notebook (end-to-end)
- `reports/` — evaluation_report.md, slides.md, confusion matrices images

## Quick summary
We trained and compared:
- Logistic Regression (TF-IDF) — baseline
- Random Forest, Linear SVM — classical baselines
- LSTM (Keras) — deep learning baseline
- BERT (Hugging Face) — transformer fine-tuning

**Best results (high-level):**
- Accuracy: ~86–87% for top models
- Neutral class remains challenging due to dataset imbalance
- Recommendations: oversampling, class weighting, more neutral data, or targeted fine-tuning

## How to run
1. Open `notebooks/amazon_sentiment_colab.ipynb` in Colab.
2. Follow top-to-bottom cells: install deps → download dataset → preprocess → train models → evaluate.
3. Scripts in `scripts/` replicate the same steps for running locally.

## License
CC0-1.0 (dataset license, check dataset source)
