# 🎮 Game Review Sentiment Analysis

A Natural Language Processing (NLP) project exploring whether the language used in video game reviews can be used to predict whether a player recommends a game.

The project analyses over **17,000 user reviews** and compares two different approaches to text classification: a traditional **Naive Bayes model** and a **Transformer-based model using semantic embeddings and Logistic Regression**.

## Project Overview

Online game reviews contain large amounts of unstructured text that can provide insight into player experiences and opinions. This project frames the problem as a binary classification task:

- **1 — Recommended**
- **0 — Not Recommended**

The aim was to investigate how effectively review text alone could predict a player's recommendation and to compare traditional NLP techniques with a more contextual Transformer-based approach.

## Exploratory Analysis

Before modelling, the dataset was explored to understand:

- The distribution of recommended and non-recommended reviews
- The number of unique games represented
- Differences in recommendation behaviour between games
- Common language associated with recommended and non-recommended reviews

This helped establish the structure of the classification problem and identify patterns within the review data.

## Model 1 — Naive Bayes

The first approach used **TF-IDF text representation with a Multinomial Naive Bayes classifier**.

TF-IDF converts review text into numerical features based on the importance of words across the dataset, allowing Naive Bayes to learn which language is associated with recommended and non-recommended reviews.

The model achieved approximately **84% accuracy** on the holdout data.

Analysis of influential features also highlighted words and terms strongly associated with each recommendation class.

## Model 2 — Transformer + Logistic Regression

The second approach used a pretrained **MiniLM Transformer** to convert each review into a contextual semantic embedding.

Unlike TF-IDF, which primarily represents individual word importance, Transformer embeddings capture information about the broader meaning and context of the review.

These embeddings were then used to train a **Logistic Regression classifier** to predict player recommendations.

The model achieved approximately **82% accuracy** on its holdout data and produced comparatively balanced performance across the two recommendation classes.

## Model Evaluation

Both approaches were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

The project demonstrates that increased model complexity does not automatically result in higher classification accuracy. The two approaches capture text differently, making comparison of their predictions and errors particularly useful when investigating sentiment in user-generated reviews.

## Unseen Review Prediction

Both trained approaches were also applied to a separate, unlabeled test dataset.

Because the test data does not contain the true recommendation labels, it cannot be used to calculate model accuracy. Instead, it provides an opportunity to compare how the two models classify previously unseen reviews and investigate cases where their predictions differ.

## Technologies

**Python** • **Pandas** • **NLTK** • **Scikit-learn** • **Sentence Transformers** • **Matplotlib** • **Jupyter Notebook**

## Repository Structure

```text
sentiment-analysis-game/
│
├── notebooks/
│   ├── game_sentiment_analysis.ipynb
│   └── transformer_sa.ipynb
├──data/
│   ├── train.csv
│   ├── test.csv
│
└── README.md
