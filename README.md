# Query Matcher - AI Customer Service Router

An ML-based system that intelligently classifies and routes customer service queries to the correct category and topic.

## 🎯 Mission

To build a simple, generalizable, and non-LLM-based system that matches customer queries to predefined categories and topics. The system learns from example data to recognize patterns, aiming for over 80% accuracy without hardcoded logic.

## 🚀 Quick Start

```bash
python menu.py
```

Select option 1 (`🚀 Quick Start`) and the system will initialize automatically and be ready for queries.

## The Journey to the Final Architecture

This project evolved through several experimental iterations to find the most effective classification model. The journey is a key part of understanding the final design.

### Iteration 1: Classic Machine Learning (TF-IDF + Naive Bayes)

*   **Approach:** Used `scikit-learn` to convert queries into TF-IDF vectors and classify them with a Naive Bayes model.
*   **Result:** Achieved ~76% accuracy. It was a solid baseline but struggled with queries that used different wording than the training examples.

### Iteration 2: Direct Vector Similarity (Sentence Transformers)

*   **Approach:** Encoded all training examples into vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2` model). A new query was then vectorized and compared against all training examples using cosine similarity.
*   **Result:** This approach was fast and transparent, showing the exact example that matched. However, accuracy was lower (~69%) because it was poor at generalizing.

### The Breakthrough: `play2.C3.py`

The most successful experiment was **`play2.C3.py`**, which refined the Sentence Transformer approach and achieved **82.5% accuracy**. It became the foundation for the final, production-ready architecture.

## 🤖 How It *Really* Works: The Final Architecture

The current system (`query_classifier.py`) is a hybrid model that automatically uses the best available algorithm, inspired by the project's experimental findings:

1.  **Primary Method (Sentence Transformers):** If `sentence-transformers` is installed, the system uses the `all-MiniLM-L6-v2` model. It encodes the user's query and finds the training example with the highest cosine similarity. This method, perfected in `play2.C3.py`, offers the best balance of accuracy and performance for this task.
2.  **Fallback Method (TF-IDF + Naive Bayes):** If Sentence Transformers is not available, the system gracefully falls back to the classic TF-IDF and Naive Bayes classifier from Iteration 1.

This multi-tiered approach ensures the system is both robust and makes use of the best-performing model discovered during the experimental phase.

## ✨ Features

- **Intelligent Classification**: Uses modern Sentence Transformers for high accuracy.
- **Graceful Fallback**: Defaults to a reliable TF-IDF/Naive Bayes model if advanced libraries are unavailable.
- **Data-Driven**: Learns entirely from the `data/training_data.yaml` file. Change the data to adapt to any domain.
- **Transparent**: Can show which training example was matched to a query.
- **Beautiful & Interactive CLI**: A rich command-line interface makes the system easy and pleasant to use.
- **BDD Tested**: Full test coverage with the Behave framework ensures reliability.

## ✅ Requirements
- Python 3.x
- `pyyaml`
- `rich` (for CLI interface)
- `scikit-learn`
- `sentence-transformers` (Recommended for best performance)

## 🧪 Testing

Run the full suite of behavior-driven development tests:

```bash
behave
```

## 📚 Training Data

The system learns from examples in `data/training_data.yaml`. To adapt it for your domain:

1.  Edit the YAML file with your categories and topics.
2.  Provide at least 3-5 example queries per topic.
3.  Restart the application to retrain the model automatically.