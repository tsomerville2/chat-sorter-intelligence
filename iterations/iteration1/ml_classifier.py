from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Tuple, List


class MLClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=300,
            ngram_range=(1, 2),
            min_df=1
        )
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            class_weight='balanced'  # Balance class weights
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train_model(self, examples: List[str], labels: List[str]) -> bool:
        """Train the classifier on examples"""
        try:
            # Augment training data by duplicating examples
            augmented_examples = []
            augmented_labels = []
            
            for example, label in zip(examples, labels):
                augmented_examples.append(example)
                augmented_labels.append(label)
                # Add lowercase version
                augmented_examples.append(example.lower())
                augmented_labels.append(label)
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(augmented_labels)
            
            # Convert text to numerical vectors
            vectors = self.vectorizer.fit_transform(augmented_examples)
            
            # Train the classifier
            self.model.fit(vectors, encoded_labels)
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_query(self, query: str) -> Tuple[str, float]:
        """Predict category:topic for a query"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Transform query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Get prediction and confidence
        encoded_prediction = self.model.predict(query_vector)[0]
        prediction = self.label_encoder.inverse_transform([encoded_prediction])[0]
        probabilities = self.model.predict_proba(query_vector)[0]
        confidence = float(np.max(probabilities))
        
        return prediction, confidence
    
    def split_label(self, label: str) -> Tuple[str, str]:
        """Split label into category and topic"""
        parts = label.split(':')
        if len(parts) == 2:
            return parts[0], parts[1]
        return '', ''