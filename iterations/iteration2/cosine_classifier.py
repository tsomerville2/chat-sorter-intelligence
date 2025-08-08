"""
Iteration 2: Cosine Similarity Approach
Direct comparison using dot product against all training examples
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict
import yaml
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    category: str
    topic: str
    confidence: float
    matched_example: str
    similarity_score: float


class CosineClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.training_vectors = None
        self.training_labels = []
        self.training_examples = []
        self.categories = {}
        
    def load_data(self, filepath: str) -> bool:
        """Load training data from YAML"""
        try:
            with open(filepath, 'r') as file:
                yaml_data = yaml.safe_load(file)
                self.categories = yaml_data.get('categories', {})
                
                # Prepare flat lists for vectorization
                for category_name, category_data in self.categories.items():
                    topics = category_data.get('topics', {})
                    
                    for topic_name, topic_data in topics.items():
                        examples = topic_data.get('examples', [])
                        
                        for example in examples:
                            self.training_examples.append(example)
                            self.training_labels.append(f"{category_name}:{topic_name}")
                
                # Vectorize all training examples
                self.training_vectors = self.vectorizer.fit_transform(self.training_examples)
                return True
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def predict(self, query: str) -> SimilarityResult:
        """Find most similar example using cosine similarity"""
        if self.training_vectors is None:
            raise ValueError("Classifier not trained")
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity with all training examples
        # This is essentially a dot product since TF-IDF vectors are normalized
        similarities = (self.training_vectors * query_vector.T).toarray().flatten()
        
        # Find the most similar example
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Get the label and example
        best_label = self.training_labels[best_idx]
        best_example = self.training_examples[best_idx]
        
        # Split label into category and topic
        category, topic = best_label.split(':')
        
        # Calculate confidence based on similarity score
        # Cosine similarity ranges from -1 to 1, but TF-IDF usually gives 0 to 1
        confidence = float(best_similarity)
        
        return SimilarityResult(
            category=category,
            topic=topic,
            confidence=confidence,
            matched_example=best_example,
            similarity_score=float(best_similarity)
        )
    
    def predict_top_k(self, query: str, k: int = 3) -> List[SimilarityResult]:
        """Get top K most similar matches"""
        if self.training_vectors is None:
            raise ValueError("Classifier not trained")
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = (self.training_vectors * query_vector.T).toarray().flatten()
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            label = self.training_labels[idx]
            category, topic = label.split(':')
            
            results.append(SimilarityResult(
                category=category,
                topic=topic,
                confidence=float(similarities[idx]),
                matched_example=self.training_examples[idx],
                similarity_score=float(similarities[idx])
            ))
        
        return results