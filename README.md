# Query Matcher

ML-based customer query classification system that matches natural language queries to categories and topics.

## Quick Start

```bash
python play.py
```

That's it! The interactive menu will start automatically.

## What This Does

This system uses machine learning to automatically classify customer service queries into appropriate categories (like "billing", "technical_support") and specific topics within those categories (like "payment_failed", "password_reset"). It learns patterns from training examples to accurately route new queries without any hardcoded rules.

## Requirements
- Python 3.x
- scikit-learn
- pyyaml
- rich (for CLI interface)

## Features

- **Data-Driven**: No hardcoded logic - learns entirely from training data
- **Generalizable**: Works with any domain by changing the YAML training file
- **Simple ML**: Uses TF-IDF vectorization with Logistic Regression
- **Beautiful CLI**: Interactive menu with color-coded results
- **Batch Processing**: Process multiple queries at once
- **BDD Tested**: Full test coverage with Behave framework

## How It Works

1. Loads hierarchical training data from `data/training_data.yaml`
2. Trains a classifier on example customer queries
3. Matches new queries to the most likely category+topic combination
4. Returns confidence scores for each match

## Testing

Run the BDD tests:
```bash
behave
```

## Training Data

The system learns from examples in `data/training_data.yaml`. To adapt it for your domain:
1. Edit the YAML file with your categories and topics
2. Provide at least 3-5 example queries per topic
3. Restart the application to retrain automatically