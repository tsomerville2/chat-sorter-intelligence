#!/usr/bin/env python3
"""
Query Classifier - BDD-Driven Implementation
Builds on insights from play2.C3 (82.5% accuracy) and play4 (75.8% accuracy)
Designed for BDD testing with clean interfaces
"""

import yaml
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Try to import advanced ML models (fallback to scikit-learn)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

try:
    from setfit import SetFitModel
    SETFIT_AVAILABLE = True
except ImportError:
    SETFIT_AVAILABLE = False


@dataclass
class QueryResult:
    """Result of query classification"""
    category: str
    topic: str
    confidence: float
    matched_example: Optional[str] = None
    similarity_score: Optional[float] = None


class DataManager:
    """Manages training data from YAML files"""
    
    def __init__(self):
        self.categories = {}
        self.training_examples = []
        self.training_labels = []
        self._example_metadata = []

    def load_yaml(self, filepath: str) -> bool:
        """Load training data from YAML file"""
        try:
            with open(filepath, 'r') as f:
                yaml_data = yaml.safe_load(f)
            self.categories = yaml_data.get('categories', {})
            return True
        except Exception as e:
            print(f"Error loading YAML: {e}")
            return False

    def prepare_training_data(self):
        """Prepare training examples and labels from categories"""
        self.training_examples.clear()
        self.training_labels.clear()
        self._example_metadata.clear()
        
        for category_name, category_data in self.categories.items():
            topics = category_data.get('topics', {})
            for topic_name, topic_data in topics.items():
                label = f"{category_name}:{topic_name}"
                examples = topic_data.get('examples', [])
                
                for example in examples:
                    self.training_examples.append(example)
                    self.training_labels.append(label)
                    self._example_metadata.append({
                        'category': category_name,
                        'topic': topic_name,
                        'example': example
                    })

    def get_training_data(self) -> Tuple[List[str], List[str]]:
        """Get training examples and labels"""
        return self.training_examples, self.training_labels

    def get_categories(self) -> Dict:
        """Get all categories"""
        return self.categories

    def get_example_metadata(self) -> List[Dict]:
        """Get metadata for all examples"""
        return self._example_metadata


class MLClassifier:
    """Machine Learning classifier with multiple algorithm support"""
    
    def __init__(self):
        self.is_trained = False
        self.algorithm = 'auto'  # Will choose best available
        
        # Initialize based on available libraries
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.algorithm = 'sentence_transformers'
            self.training_embeddings = None
        elif SETFIT_AVAILABLE:
            self.model = SetFitModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.algorithm = 'setfit'
        else:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.classifier = MultinomialNB()
            self.algorithm = 'tfidf_nb'
            
        self.training_examples = []
        self.training_labels = []
        self.example_metadata = []

    def train_model(self, examples: List[str], labels: List[str], metadata: List[Dict] = None):
        """Train the ML model"""
        if len(examples) < 10:
            raise ValueError("Need at least 10 training examples")
            
        self.training_examples = examples
        self.training_labels = labels
        self.example_metadata = metadata or []
        
        if self.algorithm == 'sentence_transformers':
            # Use sentence transformers for embeddings
            self.training_embeddings = self.model.encode(examples)
            
        elif self.algorithm == 'setfit':
            # Use SetFit for few-shot learning
            train_texts = []
            train_labels_encoded = []
            unique_labels = list(set(labels))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            
            for text, label in zip(examples, labels):
                train_texts.append(text)
                train_labels_encoded.append(label_to_id[label])
                
            self.model.fit(train_texts, train_labels_encoded)
            self.unique_labels = unique_labels
            self.label_to_id = label_to_id
            
        else:
            # Use TF-IDF + Naive Bayes
            X = self.vectorizer.fit_transform(examples)
            self.classifier.fit(X, labels)
            
        self.is_trained = True

    def predict_query(self, query: str) -> Tuple[str, float, Optional[str]]:
        """Predict category:topic for a query"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
            
        if self.algorithm == 'sentence_transformers':
            return self._predict_with_sentence_transformers(query)
        elif self.algorithm == 'setfit':
            return self._predict_with_setfit(query)
        else:
            return self._predict_with_tfidf_nb(query)
    
    def _predict_with_sentence_transformers(self, query: str) -> Tuple[str, float, Optional[str]]:
        """Predict using sentence transformers similarity"""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.training_embeddings)[0]
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        predicted_label = self.training_labels[best_idx]
        
        # Get the matched example
        matched_example = self.training_examples[best_idx] if best_idx < len(self.training_examples) else None
        
        return predicted_label, float(best_similarity), matched_example
    
    def _predict_with_setfit(self, query: str) -> Tuple[str, float, Optional[str]]:
        """Predict using SetFit"""
        predictions = self.model.predict_proba([query])[0]
        best_idx = np.argmax(predictions)
        confidence = float(predictions[best_idx])
        predicted_label = self.unique_labels[best_idx]
        
        return predicted_label, confidence, None
    
    def _predict_with_tfidf_nb(self, query: str) -> Tuple[str, float, Optional[str]]:
        """Predict using TF-IDF + Naive Bayes"""
        X = self.vectorizer.transform([query])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        return prediction, confidence, None
    
    def split_label(self, label: str) -> Tuple[str, str]:
        """Split label into category and topic"""
        if ':' in label:
            return label.split(':', 1)
        else:
            return label, 'unknown'


class QueryClassifier:
    """Main Query Classifier - BDD Interface"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_classifier = MLClassifier()
        self.is_ready = False

    def load_data(self, filepath: str) -> bool:
        """Load training data from YAML file"""
        success = self.data_manager.load_yaml(filepath)
        if not success:
            return False
            
        self.data_manager.prepare_training_data()
        return True

    def train(self) -> bool:
        """Train the classifier"""
        examples, labels = self.data_manager.get_training_data()
        metadata = self.data_manager.get_example_metadata()
        
        if len(examples) < 10:
            raise ValueError("Not enough training data - need at least 10 examples")
            
        self.ml_classifier.train_model(examples, labels, metadata)
        self.is_ready = True
        return True

    def predict(self, query: str) -> QueryResult:
        """Predict category and topic for a query"""
        if not self.is_ready:
            raise RuntimeError("Classifier not ready - call train() first")
            
        label, confidence, matched_example = self.ml_classifier.predict_query(query)
        category, topic = self.ml_classifier.split_label(label)
        
        # Calculate similarity score if available
        similarity_score = confidence if self.ml_classifier.algorithm == 'sentence_transformers' else None
        
        return QueryResult(
            category=category,
            topic=topic,
            confidence=confidence,
            matched_example=matched_example,
            similarity_score=similarity_score
        )

    def get_algorithm_info(self) -> Dict[str, str]:
        """Get information about the active algorithm"""
        return {
            'algorithm': self.ml_classifier.algorithm,
            'sentence_transformers_available': str(SENTENCE_TRANSFORMERS_AVAILABLE),
            'setfit_available': str(SETFIT_AVAILABLE)
        }


if __name__ == '__main__':
    # Quick test
    classifier = QueryClassifier()
    print(f"Algorithm info: {classifier.get_algorithm_info()}")
    
    # Test with training data
    if classifier.load_data('data/training_data.yaml'):
        classifier.train()
        
        # Test a few queries
        test_queries = [
            "my card was declined yesterday",
            "I can't remember my password", 
            "I need to know where my package is"
        ]
        
        for query in test_queries:
            result = classifier.predict(query)
            print(f"Query: '{query}'")
            print(f"Result: {result.category}:{result.topic} (confidence: {result.confidence:.3f})")
            if result.matched_example:
                print(f"Matched: {result.matched_example}")
            print()
    else:
        print("Failed to load training data")
