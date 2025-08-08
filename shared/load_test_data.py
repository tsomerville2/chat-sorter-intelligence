"""
Shared utility to load the comprehensive 1000+ test dataset
"""

import yaml
from typing import List, Dict, Tuple

def load_comprehensive_test_data() -> List[Tuple[str, str, str]]:
    """
    Load all test cases from the comprehensive test files
    Returns: List of (query, expected_category, expected_topic) tuples
    """
    all_test_cases = []
    
    # Load first test file
    try:
        with open('shared/test_1000.yaml', 'r') as f:
            data1 = yaml.safe_load(f)
            
        # Process test cases from first file
        for section_name, section_data in data1.get('test_cases', {}).items():
            for case in section_data:
                # Handle both positive and negative tests
                if 'cat' in case and 'topic' in case:
                    query = case.get('q', '')
                    category = case.get('cat')
                    topic = case.get('topic')
                    # Skip null categories (out of domain)
                    if category is not None and topic is not None:
                        all_test_cases.append((query, category, topic))
    except Exception as e:
        print(f"Error loading test_1000.yaml: {e}")
    
    # Load second test file
    try:
        with open('shared/test_1000_complete.yaml', 'r') as f:
            data2 = yaml.safe_load(f)
            
        # Process test cases from second file
        for section_name, section_data in data2.get('additional_test_cases', {}).items():
            for case in section_data:
                if 'cat' in case and 'topic' in case:
                    query = case.get('q', '')
                    category = case.get('cat')
                    topic = case.get('topic')
                    # Skip null categories (out of domain)
                    if category is not None and topic is not None:
                        all_test_cases.append((query, category, topic))
    except Exception as e:
        print(f"Error loading test_1000_complete.yaml: {e}")
    
    return all_test_cases

def load_test_data_sample(size: int = 100) -> List[Tuple[str, str, str]]:
    """
    Load a sample of test cases
    """
    all_cases = load_comprehensive_test_data()
    
    # Return sample ensuring variety (take every Nth item)
    if len(all_cases) <= size:
        return all_cases
    
    step = len(all_cases) // size
    return [all_cases[i] for i in range(0, len(all_cases), step)][:size]

def get_test_stats() -> Dict:
    """
    Get statistics about the test dataset
    """
    all_cases = load_comprehensive_test_data()
    
    category_counts = {}
    topic_counts = {}
    
    for query, category, topic in all_cases:
        category_counts[category] = category_counts.get(category, 0) + 1
        topic_key = f"{category}:{topic}"
        topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
    
    return {
        'total_cases': len(all_cases),
        'unique_categories': len(category_counts),
        'unique_topics': len(topic_counts),
        'category_distribution': category_counts,
        'topic_distribution': topic_counts
    }