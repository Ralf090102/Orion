# Machine Learning Fundamentals

## Introduction to ML

Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### 1. Supervised Learning
- **Classification**: Predicting discrete labels (e.g., spam detection)
- **Regression**: Predicting continuous values (e.g., house prices)

### 2. Unsupervised Learning
- **Clustering**: Grouping similar data points (e.g., customer segmentation)
- **Dimensionality Reduction**: Reducing feature space (e.g., PCA)

### 3. Reinforcement Learning
Learning through trial and error with rewards and penalties.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Features | Input variables used for prediction |
| Labels | Output variables we're trying to predict |
| Training Set | Data used to train the model |
| Test Set | Data used to evaluate model performance |

## Common Algorithms

```python
# Simple example of a decision tree classifier
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Best Practices

1. Always split your data into training, validation, and test sets
2. Use cross-validation to prevent overfitting
3. Normalize/standardize features when necessary
4. Monitor both training and validation metrics
5. Start simple and gradually increase complexity

> "Machine learning is the science of getting computers to learn without being explicitly programmed." - Andrew Ng
