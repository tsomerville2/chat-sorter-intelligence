#!/usr/bin/env python3
"""
Test different sentence transformer models to find the best one
"""

import sys
sys.path.append('shared')
from load_test_data import load_test_data_sample
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import yaml

def test_model(model_name, test_cases, training_data, training_labels):
    """Test a specific model"""
    print(f"\nTesting {model_name}...")
    
    try:
        model = SentenceTransformer(f'sentence-transformers/{model_name}')
        
        # Create embeddings for training data
        train_embeddings = model.encode(training_data, convert_to_numpy=True)
        
        # Test on sample
        correct = 0
        for query, expected_cat, expected_topic in test_cases:
            query_embedding = model.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(query_embedding, train_embeddings)[0]
            
            best_idx = np.argmax(similarities)
            predicted = training_labels[best_idx]
            expected = f"{expected_cat}:{expected_topic}"
            
            if predicted == expected:
                correct += 1
        
        accuracy = (correct / len(test_cases)) * 100
        print(f"  Accuracy: {accuracy:.1f}%")
        return accuracy
        
    except Exception as e:
        print(f"  Failed: {e}")
        return 0

def main():
    # Load training data
    with open('data/training_data.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    training_data = []
    training_labels = []
    
    for category_name, category_data in yaml_data['categories'].items():
        for topic_name, topic_data in category_data['topics'].items():
            for example in topic_data['examples']:
                training_data.append(example)
                training_labels.append(f"{category_name}:{topic_name}")
    
    # Load test cases
    test_cases = load_test_data_sample(100)
    
    # Test different models
    models = [
        'all-MiniLM-L6-v2',      # play2.C3 uses this
        'all-MiniLM-L12-v2',     # Larger version
        'all-mpnet-base-v2',     # Often better
        'all-distilroberta-v1',  # Different architecture
        'multi-qa-MiniLM-L6-cos-v1',  # Optimized for Q&A
        'paraphrase-MiniLM-L6-v2',    # Good for paraphrases
        'paraphrase-mpnet-base-v2',   # Paraphrase MPNet
    ]
    
    results = {}
    for model_name in models:
        accuracy = test_model(model_name, test_cases, training_data, training_labels)
        results[model_name] = accuracy
    
    # Show best model
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {acc:.1f}%")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\nBEST MODEL: {best_model[0]} with {best_model[1]:.1f}% accuracy")

if __name__ == "__main__":
    main()