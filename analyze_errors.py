#!/usr/bin/env python3
"""
Analyze errors to understand what's going wrong
"""

import sys
sys.path.append('shared')
from load_test_data import load_comprehensive_test_data
from play2 import CosineQueryMatcher
from collections import defaultdict

def analyze():
    # Load test data
    test_cases = load_comprehensive_test_data()
    
    # Load classifier
    matcher = CosineQueryMatcher()
    matcher.load_data("data/training_data.yaml")
    
    # Track errors
    errors_by_category = defaultdict(list)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for query, expected_cat, expected_topic in test_cases[:200]:  # Sample
        try:
            result = matcher.predict_query(query)
            predicted_cat, predicted_topic = result[0].split(':')
            
            if predicted_cat != expected_cat or predicted_topic != expected_topic:
                errors_by_category[expected_cat].append({
                    'query': query,
                    'expected': f"{expected_cat}:{expected_topic}",
                    'predicted': result[0]
                })
                confusion_matrix[expected_cat][predicted_cat] += 1
        except:
            pass
    
    # Print analysis
    print("ERROR ANALYSIS")
    print("=" * 50)
    
    for category, errors in errors_by_category.items():
        print(f"\n{category}: {len(errors)} errors")
        if errors:
            print(f"  Sample: '{errors[0]['query'][:50]}...'")
            print(f"  Expected: {errors[0]['expected']}")
            print(f"  Got: {errors[0]['predicted']}")
    
    print("\nCONFUSION PATTERNS")
    print("=" * 50)
    for true_cat, predictions in confusion_matrix.items():
        for pred_cat, count in predictions.items():
            if count > 2:  # Show significant confusions
                print(f"{true_cat} -> {pred_cat}: {count} times")

if __name__ == "__main__":
    analyze()