from dataclasses import dataclass
from data_manager import DataManager
from ml_classifier import MLClassifier


@dataclass
class MatchResult:
    category: str
    topic: str
    confidence: float


class QueryClassifier:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_classifier = MLClassifier()
        self.is_ready = False
    
    def load_data(self, filepath: str) -> bool:
        """Load training data from YAML file"""
        success = self.data_manager.load_yaml(filepath)
        if not success:
            raise ValueError("Failed to load data")
        
        self.data_manager.prepare_training_data()
        return True
    
    def train(self) -> bool:
        """Train the ML model on loaded data"""
        examples, labels = self.data_manager.get_training_data()
        
        if len(examples) < 10:
            raise ValueError("Not enough training data")
        
        success = self.ml_classifier.train_model(examples, labels)
        if success:
            self.is_ready = True
        return success
    
    def predict(self, query: str) -> MatchResult:
        """Predict category and topic for a query"""
        if not self.is_ready:
            raise ValueError("Classifier not ready")
        
        label, confidence = self.ml_classifier.predict_query(query)
        category, topic = self.ml_classifier.split_label(label)
        
        return MatchResult(
            category=category,
            topic=topic,
            confidence=confidence
        )