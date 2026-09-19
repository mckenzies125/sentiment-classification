# Sentiment Classification

Implementation of linear and neural approaches to binary sentiment classification, developed as part of NYU's DS-GA 1011 course.

## Overview

This project explores two approaches to classifying text as having positive or negative sentiment:

- **Logistic Regression:** Implemented from scratch using NumPy with sparse bag-of-words features.
- **Deep Averaging Network (DAN):** Implemented in PyTorch using pretrained GloVe word embeddings and a feedforward neural network.

## Methods

### Logistic Regression
The linear classifier uses unigram bag-of-words features and a sparse feature representation. Model training was implemented from scratch, including:

- Feature extraction and vocabulary indexing
- Sparse weight calculations
- Sigmoid-based binary classification
- Negative log-likelihood optimization
- Stochastic gradient descent

### Deep Averaging Network
The neural classifier represents sentences by averaging pretrained GloVe word embeddings before passing them through a feedforward neural network.

The architecture consists of:

1. Pretrained GloVe embeddings
2. Average pooling over word embeddings
3. Fully connected hidden layer
4. ReLU activation
5. Output layer for binary sentiment classification

The model was trained using PyTorch and the Adam optimizer.

## Technologies

Python · NumPy · PyTorch · GloVe · NLP · Machine Learning

## Attribution

This project originated from an assignment for **DS-GA 1011 at New York University**. The assignment specification and supporting infrastructure were provided by the course staff.

The implementation included in this repositorym, contained in `models.py`, is my work.
