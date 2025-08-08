#!/usr/bin/env python3
"""
Test the new query_classifier.py accuracy
"""

import sys
sys.path.append('shared')
from load_test_data import load_comprehensive_test_data, load_test_data_sample
from query_classifier import QueryClassifier

def test_accuracy():
    # Initialize classifier
    classifier = QueryClassifier()
    
    # Load training data
    if not classifier.load_data("data/training_data.yaml"):
        print("Failed to load training data")
        return
    
    # Train the classifier
    classifier.train()
    
    # Test on sample data first
    print("Testing on 100 samples...")
    test_cases = load_test_data_sample(100)
    
    correct = 0
    for query, expected_cat, expected_topic in test_cases:
        try:
            result = classifier.predict(query)
            if result.category == expected_cat and result.topic == expected_topic:
                correct += 1
        except Exception as e:
            pass
    
    sample_accuracy = (correct / len(test_cases)) * 100
    print(f"Sample accuracy (100): {sample_accuracy:.1f}%")
    
    # Test on full data
    print("\nTesting on full dataset...")
    test_cases = load_comprehensive_test_data()
    
    if not test_cases:
        print("Failed to load comprehensive test data")
        return
        
    correct = 0
    for query, expected_cat, expected_topic in test_cases:
        try:
            result = classifier.predict(query)
            if result.category == expected_cat and result.topic == expected_topic:
                correct += 1
        except Exception as e:
            pass
    
    full_accuracy = (correct / len(test_cases)) * 100
    print(f"Full accuracy ({len(test_cases)} queries): {full_accuracy:.1f}%")
    
    # Compare with play2.C3 (82.5%) and play4 (75.8%)
    print("\nComparison:")
    print(f"- query_classifier.py: {full_accuracy:.1f}%")
    print(f"- play2.C3.py: 82.5%")
    print(f"- play4.py: 75.8%")
    
    if full_accuracy > 82.5:
        print(f"\n✅ SUCCESS: Improved by {full_accuracy - 82.5:.1f}% over play2.C3!")
    else:
        print(f"\n❌ NOT YET: Still {82.5 - full_accuracy:.1f}% below play2.C3")
    
    return full_accuracy

if __name__ == "__main__":
    test_accuracy()