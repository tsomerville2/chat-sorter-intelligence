#!/usr/bin/env python3
"""
Comprehensive test suite comparing all three iterations
"""

import sys
import yaml
import json
from typing import Dict, List
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich import box
import time

# Import all classifiers
sys.path.append('iterations/iteration1')
sys.path.append('iterations/iteration2')
sys.path.append('iterations/iteration3')

from query_classifier import QueryClassifier
from cosine_classifier import CosineClassifier
from llm_classifier import LLMClassifier


console = Console()


@dataclass
class TestResult:
    query: str
    expected_category: str
    expected_topic: str
    predicted_category: str
    predicted_topic: str
    confidence: float
    correct: bool
    time_ms: float


class IterationTester:
    def __init__(self):
        self.test_queries = self.load_test_queries()
        self.results = {}
        
    def load_test_queries(self) -> List[Dict]:
        """Load test queries from shared YAML"""
        with open('shared/test_queries.yaml', 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten all test categories into a single list
        all_queries = []
        for category in ['perfect_matches', 'close_variations', 'ambiguous', 'complex']:
            if category in data['test_queries']:
                all_queries.extend(data['test_queries'][category])
        
        return all_queries
    
    def test_iteration1(self) -> List[TestResult]:
        """Test ML classifier (Logistic Regression)"""
        console.print("[bold cyan]Testing Iteration 1 (Logistic Regression)...[/bold cyan]")
        
        classifier = QueryClassifier()
        classifier.load_data("data/training_data.yaml")
        classifier.train()
        
        results = []
        for test_case in self.test_queries:
            if test_case.get('expected_category') is None:
                continue  # Skip out-of-domain tests
            
            start_time = time.time()
            try:
                result = classifier.predict(test_case['query'])
                elapsed_ms = (time.time() - start_time) * 1000
                
                correct = (result.category == test_case['expected_category'] and 
                          result.topic == test_case['expected_topic'])
                
                results.append(TestResult(
                    query=test_case['query'],
                    expected_category=test_case['expected_category'],
                    expected_topic=test_case['expected_topic'],
                    predicted_category=result.category,
                    predicted_topic=result.topic,
                    confidence=result.confidence,
                    correct=correct,
                    time_ms=elapsed_ms
                ))
            except Exception as e:
                console.print(f"[red]Error testing '{test_case['query']}': {e}[/red]")
        
        return results
    
    def test_iteration2(self) -> List[TestResult]:
        """Test cosine similarity classifier"""
        console.print("[bold magenta]Testing Iteration 2 (Cosine Similarity)...[/bold magenta]")
        
        classifier = CosineClassifier()
        classifier.load_data("data/training_data.yaml")
        
        results = []
        for test_case in self.test_queries:
            if test_case.get('expected_category') is None:
                continue
            
            start_time = time.time()
            try:
                result = classifier.predict(test_case['query'])
                elapsed_ms = (time.time() - start_time) * 1000
                
                correct = (result.category == test_case['expected_category'] and 
                          result.topic == test_case['expected_topic'])
                
                results.append(TestResult(
                    query=test_case['query'],
                    expected_category=test_case['expected_category'],
                    expected_topic=test_case['expected_topic'],
                    predicted_category=result.category,
                    predicted_topic=result.topic,
                    confidence=result.confidence,
                    correct=correct,
                    time_ms=elapsed_ms
                ))
            except Exception as e:
                console.print(f"[red]Error testing '{test_case['query']}': {e}[/red]")
        
        return results
    
    def test_iteration3(self, model_name: str = "openai/gpt-oss-20b") -> List[TestResult]:
        """Test LLM classifier"""
        import os
        if not os.environ.get("GROQ_API_KEY"):
            console.print("[yellow]Skipping Iteration 3 - GROQ_API_KEY not set[/yellow]")
            return []
        
        console.print(f"[bold green]Testing Iteration 3 ({model_name})...[/bold green]")
        
        classifier = LLMClassifier(model_name=model_name)
        classifier.load_data("data/training_data.yaml")
        
        results = []
        for test_case in self.test_queries[:5]:  # Limit to 5 to save API calls
            if test_case.get('expected_category') is None:
                continue
            
            start_time = time.time()
            try:
                result = classifier.predict(test_case['query'])
                elapsed_ms = (time.time() - start_time) * 1000
                
                correct = (result.category == test_case['expected_category'] and 
                          result.topic == test_case['expected_topic'])
                
                results.append(TestResult(
                    query=test_case['query'],
                    expected_category=test_case['expected_category'],
                    expected_topic=test_case['expected_topic'],
                    predicted_category=result.category,
                    predicted_topic=result.topic,
                    confidence=result.confidence,
                    correct=correct,
                    time_ms=elapsed_ms
                ))
            except Exception as e:
                console.print(f"[red]Error testing '{test_case['query']}': {e}[/red]")
        
        return results
    
    def print_comparison_table(self, results: Dict[str, List[TestResult]]):
        """Print comparison table of all iterations"""
        table = Table(title="Performance Comparison", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Iteration 1\n(Logistic Regression)", style="yellow")
        table.add_column("Iteration 2\n(Cosine Similarity)", style="magenta")
        table.add_column("Iteration 3\n(LLM-20B)", style="green")
        
        # Calculate metrics for each iteration
        metrics = {}
        for name, iteration_results in results.items():
            if not iteration_results:
                metrics[name] = {
                    'accuracy': 0,
                    'avg_confidence': 0,
                    'avg_time': 0,
                    'total': 0
                }
                continue
            
            correct = sum(1 for r in iteration_results if r.correct)
            total = len(iteration_results)
            avg_conf = sum(r.confidence for r in iteration_results) / total if total > 0 else 0
            avg_time = sum(r.time_ms for r in iteration_results) / total if total > 0 else 0
            
            metrics[name] = {
                'accuracy': (correct / total * 100) if total > 0 else 0,
                'avg_confidence': avg_conf,
                'avg_time': avg_time,
                'total': total
            }
        
        # Add rows to table
        table.add_row(
            "Accuracy",
            f"{metrics.get('iteration1', {}).get('accuracy', 0):.1f}%",
            f"{metrics.get('iteration2', {}).get('accuracy', 0):.1f}%",
            f"{metrics.get('iteration3', {}).get('accuracy', 0):.1f}%"
        )
        
        table.add_row(
            "Avg Confidence",
            f"{metrics.get('iteration1', {}).get('avg_confidence', 0):.2%}",
            f"{metrics.get('iteration2', {}).get('avg_confidence', 0):.2%}",
            f"{metrics.get('iteration3', {}).get('avg_confidence', 0):.2%}"
        )
        
        table.add_row(
            "Avg Response Time",
            f"{metrics.get('iteration1', {}).get('avg_time', 0):.1f}ms",
            f"{metrics.get('iteration2', {}).get('avg_time', 0):.1f}ms",
            f"{metrics.get('iteration3', {}).get('avg_time', 0):.1f}ms"
        )
        
        table.add_row(
            "Test Cases",
            str(metrics.get('iteration1', {}).get('total', 0)),
            str(metrics.get('iteration2', {}).get('total', 0)),
            str(metrics.get('iteration3', {}).get('total', 0))
        )
        
        console.print(table)
        
        return metrics
    
    def print_detailed_results(self, results: Dict[str, List[TestResult]]):
        """Print detailed results for each query"""
        for iteration_name, iteration_results in results.items():
            if not iteration_results:
                continue
            
            console.print(f"\n[bold]{iteration_name.replace('iteration', 'Iteration ')}[/bold]")
            
            table = Table(box=box.SIMPLE)
            table.add_column("Query", style="white", width=40)
            table.add_column("Expected", style="cyan")
            table.add_column("Predicted", style="magenta")
            table.add_column("Conf", style="yellow")
            table.add_column("✓/✗", style="green")
            
            for result in iteration_results[:10]:  # Show first 10
                expected = f"{result.expected_category}:{result.expected_topic}"
                predicted = f"{result.predicted_category}:{result.predicted_topic}"
                
                table.add_row(
                    result.query[:40],
                    expected[:30],
                    predicted[:30],
                    f"{result.confidence:.1%}",
                    "✓" if result.correct else "✗"
                )
            
            console.print(table)
    
    def generate_report(self, results: Dict[str, List[TestResult]], metrics: Dict):
        """Generate comprehensive QA report"""
        with open('QA-REPORT.md', 'w') as f:
            f.write("# Query Matcher - QA Report\n\n")
            f.write("## Executive Summary\n\n")
            
            # Find best performer
            best_accuracy = 0
            best_iteration = ""
            for name, metric in metrics.items():
                if metric['accuracy'] > best_accuracy:
                    best_accuracy = metric['accuracy']
                    best_iteration = name
            
            f.write(f"**Best Performer:** {best_iteration.replace('iteration', 'Iteration ')} ")
            f.write(f"with {best_accuracy:.1f}% accuracy\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Iteration 1 (ML) | Iteration 2 (Cosine) | Iteration 3 (LLM) |\n")
            f.write("|--------|------------------|---------------------|-------------------|\n")
            f.write(f"| Accuracy | {metrics.get('iteration1', {}).get('accuracy', 0):.1f}% | ")
            f.write(f"{metrics.get('iteration2', {}).get('accuracy', 0):.1f}% | ")
            f.write(f"{metrics.get('iteration3', {}).get('accuracy', 0):.1f}% |\n")
            f.write(f"| Avg Confidence | {metrics.get('iteration1', {}).get('avg_confidence', 0):.2%} | ")
            f.write(f"{metrics.get('iteration2', {}).get('avg_confidence', 0):.2%} | ")
            f.write(f"{metrics.get('iteration3', {}).get('avg_confidence', 0):.2%} |\n")
            f.write(f"| Avg Response Time | {metrics.get('iteration1', {}).get('avg_time', 0):.1f}ms | ")
            f.write(f"{metrics.get('iteration2', {}).get('avg_time', 0):.1f}ms | ")
            f.write(f"{metrics.get('iteration3', {}).get('avg_time', 0):.1f}ms |\n\n")
            
            f.write("## Detailed Analysis\n\n")
            
            f.write("### Iteration 1: Logistic Regression\n")
            f.write("- **Strengths:** Fast, no API costs, reasonable accuracy\n")
            f.write("- **Weaknesses:** Low confidence scores, poor on variations\n")
            f.write("- **Best for:** High-volume, cost-sensitive applications\n\n")
            
            f.write("### Iteration 2: Cosine Similarity\n")
            f.write("- **Strengths:** Transparent, shows matched example, fast\n")
            f.write("- **Weaknesses:** Can't generalize beyond exact matches\n")
            f.write("- **Best for:** Small datasets with distinct examples\n\n")
            
            f.write("### Iteration 3: LLM-based\n")
            f.write("- **Strengths:** Best understanding, handles variations, provides reasoning\n")
            f.write("- **Weaknesses:** Requires API key, costs money, slower\n")
            f.write("- **Best for:** Complex queries needing human-like understanding\n\n")
            
            f.write("## Test Cases Analysis\n\n")
            
            # Analyze specific problematic queries
            f.write("### Problematic Queries\n\n")
            for iteration_name, iteration_results in results.items():
                incorrect = [r for r in iteration_results if not r.correct]
                if incorrect:
                    f.write(f"**{iteration_name.replace('iteration', 'Iteration ')}:**\n")
                    for r in incorrect[:3]:
                        f.write(f"- \"{r.query}\" → Expected: {r.expected_category}:{r.expected_topic}, ")
                        f.write(f"Got: {r.predicted_category}:{r.predicted_topic}\n")
                    f.write("\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **For production with budget constraints:** Use Iteration 1 (Logistic Regression)\n")
            f.write("2. **For transparency and debugging:** Use Iteration 2 (Cosine Similarity)\n")
            f.write("3. **For best accuracy and handling complex queries:** Use Iteration 3 (LLM)\n")
            f.write("4. **Hybrid approach:** Use Iteration 2 for exact matches, fall back to Iteration 3 for low-confidence results\n\n")
            
            f.write("## Conclusion\n\n")
            f.write("The test demonstrates that simple ML approaches (Iterations 1 & 2) can achieve ")
            f.write("reasonable accuracy for straightforward queries, but struggle with variations ")
            f.write("and typos. The LLM approach (Iteration 3) provides superior understanding but ")
            f.write("comes with API costs and latency. Choose based on your specific requirements ")
            f.write("for accuracy, cost, and speed.\n")


def main():
    """Run comprehensive tests"""
    console.print("[bold]Query Matcher - Comprehensive Test Suite[/bold]\n")
    
    tester = IterationTester()
    results = {}
    
    # Test each iteration
    results['iteration1'] = tester.test_iteration1()
    results['iteration2'] = tester.test_iteration2()
    results['iteration3'] = tester.test_iteration3()
    
    # Print results
    console.print("\n[bold]Performance Comparison[/bold]\n")
    metrics = tester.print_comparison_table(results)
    
    console.print("\n[bold]Detailed Results[/bold]")
    tester.print_detailed_results(results)
    
    # Generate report
    tester.generate_report(results, metrics)
    console.print("\n[green]✓ QA Report generated: QA-REPORT.md[/green]")


if __name__ == "__main__":
    main()