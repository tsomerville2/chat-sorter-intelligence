#!/usr/bin/env python3
"""
Main Controller - API Layer for Query Matcher
Provides clean API interface for CLI and potential web frontends
"""

import sys
import os
from typing import Dict, Any, Optional
from dataclasses import asdict
from query_classifier import QueryClassifier, QueryResult
import json
import time


class MainController:
    """Main controller providing API layer for query matching"""
    
    def __init__(self, data_path: str = "data/training_data.yaml"):
        self.data_path = data_path
        self.classifier = None
        self.is_initialized = False
        self.initialization_time = None
        
    def initialize(self) -> Dict[str, Any]:
        """Initialize the classifier with training data"""
        try:
            start_time = time.time()
            
            self.classifier = QueryClassifier()
            
            # Load training data
            if not self.classifier.load_data(self.data_path):
                return {
                    "success": False,
                    "error": f"Failed to load training data from {self.data_path}",
                    "error_type": "data_load_error"
                }
            
            # Train the classifier
            self.classifier.train()
            
            self.initialization_time = time.time() - start_time
            self.is_initialized = True
            
            # Get algorithm info
            algo_info = self.classifier.get_algorithm_info()
            
            return {
                "success": True,
                "message": "Query classifier initialized successfully",
                "initialization_time_seconds": round(self.initialization_time, 3),
                "algorithm_info": algo_info,
                "data_path": self.data_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "initialization_error"
            }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query and return classification result"""
        if not self.is_initialized or self.classifier is None:
            return {
                "success": False,
                "error": "Controller not initialized. Call initialize() first.",
                "error_type": "not_initialized"
            }
            
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Empty query provided",
                "error_type": "empty_query"
            }
            
        try:
            start_time = time.time()
            
            result = self.classifier.predict(query.strip())
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "query": query.strip(),
                "result": {
                    "category": result.category,
                    "topic": result.topic,
                    "confidence": round(result.confidence, 4),
                    "matched_example": result.matched_example,
                    "similarity_score": round(result.similarity_score, 4) if result.similarity_score else None
                },
                "processing_time_seconds": round(processing_time, 4),
                "classification": f"{result.category}:{result.topic}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "prediction_error",
                "query": query.strip()
            }
    
    def process_batch_queries(self, queries: list) -> Dict[str, Any]:
        """Process multiple queries in batch"""
        if not self.is_initialized:
            return {
                "success": False,
                "error": "Controller not initialized",
                "error_type": "not_initialized"
            }
            
        results = []
        errors = []
        
        start_time = time.time()
        
        for i, query in enumerate(queries):
            result = self.process_query(query)
            if result["success"]:
                results.append(result)
            else:
                errors.append({
                    "index": i,
                    "query": query,
                    "error": result["error"]
                })
        
        total_time = time.time() - start_time
        
        return {
            "success": len(results) > 0,
            "total_queries": len(queries),
            "successful_results": len(results),
            "errors": len(errors),
            "results": results,
            "error_details": errors if errors else None,
            "total_processing_time_seconds": round(total_time, 4),
            "average_time_per_query": round(total_time / len(queries), 4) if queries else 0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "initialized": self.is_initialized,
            "data_path": self.data_path,
            "initialization_time_seconds": self.initialization_time,
            "algorithm_info": self.classifier.get_algorithm_info() if self.classifier else None
        }
    
    def get_categories_info(self) -> Dict[str, Any]:
        """Get information about available categories and topics"""
        if not self.is_initialized:
            return {
                "success": False,
                "error": "Controller not initialized"
            }
            
        try:
            categories = self.classifier.data_manager.get_categories()
            
            # Build summary
            category_summary = {}
            total_examples = 0
            
            for cat_name, cat_data in categories.items():
                topics = cat_data.get('topics', {})
                topic_summary = {}
                
                for topic_name, topic_data in topics.items():
                    examples = topic_data.get('examples', [])
                    topic_summary[topic_name] = {
                        "example_count": len(examples),
                        "sample_examples": examples[:3] if len(examples) > 3 else examples
                    }
                    total_examples += len(examples)
                
                category_summary[cat_name] = {
                    "topic_count": len(topics),
                    "topics": topic_summary
                }
            
            return {
                "success": True,
                "total_categories": len(categories),
                "total_examples": total_examples,
                "categories": category_summary
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "categories_error"
            }


def main():
    """Simple CLI interface for testing"""
    controller = MainController()
    
    print("Initializing Query Matcher...")
    init_result = controller.initialize()
    
    if not init_result["success"]:
        print(f"❌ Initialization failed: {init_result['error']}")
        return
    
    print(f"✅ Initialized successfully in {init_result['initialization_time_seconds']}s")
    print(f"📊 Algorithm: {init_result['algorithm_info']['algorithm']}")
    print()
    
    # Interactive loop
    print("Enter queries to classify (type 'quit' to exit, 'status' for info, 'categories' for data info):")
    
    while True:
        try:
            query = input("\n> ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            elif query.lower() == 'status':
                status = controller.get_status()
                print(json.dumps(status, indent=2))
            elif query.lower() == 'categories':
                categories = controller.get_categories_info()
                print(json.dumps(categories, indent=2))
            elif query:
                result = controller.process_query(query)
                if result["success"]:
                    r = result["result"]
                    print(f"📂 Category: {r['category']}")
                    print(f"🏷️  Topic: {r['topic']}")
                    print(f"🎯 Confidence: {r['confidence']:.1%}")
                    if r['matched_example']:
                        print(f"📝 Matched: {r['matched_example']}")
                    print(f"⏱️  Time: {result['processing_time_seconds']}s")
                else:
                    print(f"❌ Error: {result['error']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == '__main__':
    main()
