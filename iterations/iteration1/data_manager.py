import yaml
from typing import Dict, List, Tuple


class DataManager:
    def __init__(self):
        self.categories = {}
        self.training_examples = []
        self.training_labels = []
    
    def load_yaml(self, filepath: str) -> bool:
        """Load training data from YAML file"""
        try:
            with open(filepath, 'r') as file:
                yaml_data = yaml.safe_load(file)
                self.categories = yaml_data.get('categories', {})
                return True
        except Exception as e:
            print(f"Error loading YAML: {e}")
            return False
    
    def prepare_training_data(self) -> bool:
        """Convert hierarchical data to flat lists for training"""
        self.training_examples = []
        self.training_labels = []
        
        for category_name, category_data in self.categories.items():
            topics = category_data.get('topics', {})
            
            for topic_name, topic_data in topics.items():
                label = f"{category_name}:{topic_name}"
                examples = topic_data.get('examples', [])
                
                for example in examples:
                    self.training_examples.append(example)
                    self.training_labels.append(label)
        
        return True
    
    def get_training_data(self) -> Tuple[List[str], List[str]]:
        """Return training examples and labels"""
        return self.training_examples, self.training_labels
    
    def get_categories(self) -> Dict:
        """Return categories dictionary"""
        return self.categories