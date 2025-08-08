#!/usr/bin/env python3
"""
Query Matcher - Iteration 2.B (Different Similarity Metrics)
Experimental: Trying Jaccard, Dice, and character n-grams
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
import sys
import os
import time
sys.path.append('shared')

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Tuple, Dict
import yaml
from dataclasses import dataclass
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats


console = Console()


@dataclass
class SimilarityResult:
    category: str
    topic: str
    confidence: float
    matched_example: str
    similarity_score: float
    metric_used: str


class MultiMetricClassifier:
    def __init__(self, metric='jaccard'):
        self.metric = metric
        
        if metric in ['jaccard', 'dice']:
            # Use binary count vectorizer for set-based metrics
            self.vectorizer = CountVectorizer(
                lowercase=True,
                binary=True,  # Binary for set operations
                ngram_range=(1, 2),
                token_pattern=r'\b\w+\b|[^\w\s]'  # Include punctuation
            )
        elif metric == 'char_ngram':
            # Character n-grams can capture partial matches
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                analyzer='char',
                ngram_range=(2, 4),  # Character bigrams to 4-grams
            )
        else:  # weighted_cosine
            # Custom weighted TF-IDF
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 3),  # Include trigrams
                min_df=1,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
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
    
    def calculate_similarity(self, vec1, vec2):
        """Calculate similarity based on selected metric"""
        if self.metric == 'jaccard':
            # Jaccard similarity: |A ∩ B| / |A ∪ B|
            intersection = np.minimum(vec1.toarray(), vec2.toarray()).sum()
            union = np.maximum(vec1.toarray(), vec2.toarray()).sum()
            return intersection / union if union > 0 else 0
            
        elif self.metric == 'dice':
            # Dice coefficient: 2|A ∩ B| / (|A| + |B|)
            intersection = np.minimum(vec1.toarray(), vec2.toarray()).sum()
            sum_sizes = vec1.toarray().sum() + vec2.toarray().sum()
            return 2 * intersection / sum_sizes if sum_sizes > 0 else 0
            
        elif self.metric == 'char_ngram':
            # Standard cosine for char n-grams
            return (vec1 * vec2.T).toarray()[0, 0]
            
        else:  # weighted_cosine
            # Enhanced cosine with better normalization
            dot_product = (vec1 * vec2.T).toarray()[0, 0]
            norm1 = np.sqrt((vec1 * vec1.T).toarray()[0, 0])
            norm2 = np.sqrt((vec2 * vec2.T).toarray()[0, 0])
            if norm1 * norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)
    
    def predict(self, query: str) -> SimilarityResult:
        """Find most similar example using selected metric"""
        if self.training_vectors is None:
            raise ValueError("Classifier not trained")
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with all training examples
        similarities = []
        for i in range(self.training_vectors.shape[0]):
            train_vec = self.training_vectors[i]
            sim = self.calculate_similarity(query_vector, train_vec)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Find the most similar example
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Get the label and example
        best_label = self.training_labels[best_idx]
        best_example = self.training_examples[best_idx]
        
        # Split label into category and topic
        category, topic = best_label.split(':')
        
        return SimilarityResult(
            category=category,
            topic=topic,
            confidence=float(best_similarity),
            matched_example=best_example,
            similarity_score=float(best_similarity),
            metric_used=self.metric
        )
    
    def predict_top_k(self, query: str, k: int = 3) -> List[SimilarityResult]:
        """Get top K most similar matches"""
        if self.training_vectors is None:
            raise ValueError("Classifier not trained")
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = []
        for i in range(self.training_vectors.shape[0]):
            train_vec = self.training_vectors[i]
            sim = self.calculate_similarity(query_vector, train_vec)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
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
                similarity_score=float(similarities[idx]),
                metric_used=self.metric
            ))
        
        return results


def display_welcome(metric):
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🎯 ", style="bold magenta")
    welcome_text.append("QUERY MATCHER v2.B", style="bold yellow")
    welcome_text.append(f" - {metric.title()} Similarity", style="bold cyan")
    
    metric_desc = {
        'jaccard': "Set-based similarity: |A ∩ B| / |A ∪ B|",
        'dice': "Dice coefficient: 2|A ∩ B| / (|A| + |B|)",
        'char_ngram': "Character n-gram matching (captures typos)",
        'weighted_cosine': "Enhanced cosine with trigrams"
    }
    
    panel = Panel(
        f"[magenta]Experimental: {metric_desc.get(metric, 'Unknown metric')}[/magenta]\n"
        "[dim]Testing different similarity metrics for better matching[/dim]",
        title=welcome_text,
        border_style="bright_magenta",
        padding=(1, 2),
        box=box.DOUBLE
    )
    console.print(panel)
    console.print()


def display_menu():
    """Display main menu"""
    table = Table(show_header=False, box=box.ROUNDED, border_style="magenta")
    table.add_column("Option", style="bold yellow", width=3)
    table.add_column("Action", style="white")
    
    table.add_row("1", "🎯 Quick Match (with similarity details)")
    table.add_row("2", "🔍 Top 3 Matches")
    table.add_row("3", "🔄 Switch Metric")
    table.add_row("4", "🧪 Run Quick Test (5 queries)")
    table.add_row("5", "🔬 Run Sample Test (100 queries)")
    table.add_row("6", "⚡ Run Full Test Suite (1000+ queries)")
    table.add_row("7", "⚖️ Compare All Metrics")
    table.add_row("8", "📈 Show Test Statistics")
    table.add_row("9", "ℹ️  About These Metrics")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def quick_match(classifier):
    """Quick match showing the most similar example"""
    console.print(f"[bold magenta]Quick Match Mode - {classifier.metric.title()} Metric[/bold magenta]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status(f"[magenta]Finding match using {classifier.metric}...[/magenta]"):
            try:
                result = classifier.predict(query)
                
                # Create detailed result panel
                result_table = Table(show_header=False, box=box.SIMPLE)
                result_table.add_column("Field", style="magenta")
                result_table.add_column("Value", style="green")
                
                result_table.add_row("Category", f"[bold]{result.category}[/bold]")
                result_table.add_row("Topic", f"[bold]{result.topic}[/bold]")
                result_table.add_row("Similarity Score", f"{result.similarity_score:.4f}")
                result_table.add_row("Metric", result.metric_used)
                result_table.add_row("Matched Example", f'"{result.matched_example}"')
                
                # Warn if zero similarity
                if result.similarity_score == 0:
                    console.print("[bold red]⚠️  WARNING: Zero similarity - defaulting to first training example![/bold red]")
                
                console.print(Panel(result_table, title="[green]✓ Best Match Found[/green]", 
                                  border_style="green"))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def switch_metric(classifier):
    """Switch between different similarity metrics"""
    console.print("[bold magenta]Switch Similarity Metric[/bold magenta]\n")
    console.print(f"Current metric: [cyan]{classifier.metric}[/cyan]\n")
    
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Option", style="yellow")
    table.add_column("Metric", style="white")
    table.add_column("Description", style="dim")
    
    table.add_row("1", "jaccard", "Set-based: Good for exact word matches")
    table.add_row("2", "dice", "Similar to Jaccard but gives more weight to overlap")
    table.add_row("3", "char_ngram", "Character n-grams: Handles typos well")
    table.add_row("4", "weighted_cosine", "Enhanced cosine with trigrams")
    
    console.print(table)
    
    choice = Prompt.ask("\n[bold]Select metric[/bold]", choices=["1", "2", "3", "4"])
    
    metrics = ["jaccard", "dice", "char_ngram", "weighted_cosine"]
    new_metric = metrics[int(choice) - 1]
    
    console.print(f"\n[yellow]Switching to {new_metric}... Reloading classifier...[/yellow]")
    
    # Create new classifier with different metric
    new_classifier = MultiMetricClassifier(metric=new_metric)
    new_classifier.load_data("data/training_data.yaml")
    
    console.print(f"[green]✓ Switched to {new_metric} metric![/green]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
    
    return new_classifier


def run_quick_test(classifier):
    """Run the original 5-query test"""
    console.print(f"[bold magenta]Running Quick Test Suite - {classifier.metric.title()} Metric[/bold magenta]\n")
    
    test_queries = [
        ("I can't login", "technical_support", "password_reset"),
        ("my payment didn't work", "billing", "payment_failed"),
        ("where's my stuff", "shipping", "track_order"),
        ("I forgot my password", "technical_support", "password_reset"),
        ("card declined", "billing", "payment_failed"),
    ]
    
    results_table = Table(title=f"Test Results - {classifier.metric}", box=box.ROUNDED)
    results_table.add_column("Query", style="white")
    results_table.add_column("Expected", style="cyan")
    results_table.add_column("Predicted", style="magenta")
    results_table.add_column("Similarity", style="yellow")
    results_table.add_column("✓/✗", style="green")
    
    correct = 0
    zero_sim_count = 0
    
    for query, expected_cat, expected_topic in test_queries:
        result = classifier.predict(query)
        expected = f"{expected_cat}:{expected_topic}"
        predicted = f"{result.category}:{result.topic}"
        is_correct = expected == predicted
        if is_correct:
            correct += 1
        if result.similarity_score == 0:
            zero_sim_count += 1
        
        results_table.add_row(
            query,
            expected,
            predicted,
            f"{result.similarity_score:.3f}",
            "✓" if is_correct else "✗"
        )
    
    console.print(results_table)
    console.print(f"\n[bold]Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.1f}%)[/bold]")
    console.print(f"[bold]Metric: {classifier.metric}[/bold]")
    if zero_sim_count > 0:
        console.print(f"[bold red]⚠️  {zero_sim_count} queries had ZERO similarity[/bold red]")
    else:
        console.print("[bold green]✓ No zero similarity issues![/bold green]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def compare_all_metrics():
    """Compare all metrics on the quick test set"""
    console.print("[bold magenta]Comparing All Similarity Metrics[/bold magenta]\n")
    
    test_queries = [
        ("I can't login", "technical_support", "password_reset"),
        ("my payment didn't work", "billing", "payment_failed"),
        ("where's my stuff", "shipping", "track_order"),
        ("I forgot my password", "technical_support", "password_reset"),
        ("card declined", "billing", "payment_failed"),
    ]
    
    metrics = ["jaccard", "dice", "char_ngram", "weighted_cosine"]
    results = {}
    
    for metric in metrics:
        console.print(f"[yellow]Testing {metric}...[/yellow]")
        classifier = MultiMetricClassifier(metric=metric)
        classifier.load_data("data/training_data.yaml")
        
        correct = 0
        zero_sim = 0
        avg_sim = 0
        
        for query, expected_cat, expected_topic in test_queries:
            result = classifier.predict(query)
            if result.category == expected_cat and result.topic == expected_topic:
                correct += 1
            if result.similarity_score == 0:
                zero_sim += 1
            avg_sim += result.similarity_score
        
        results[metric] = {
            'correct': correct,
            'zero_sim': zero_sim,
            'avg_sim': avg_sim / len(test_queries)
        }
    
    # Display comparison table
    comparison_table = Table(title="Metric Comparison", box=box.ROUNDED)
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Accuracy", style="green")
    comparison_table.add_column("Zero Sim", style="red")
    comparison_table.add_column("Avg Similarity", style="yellow")
    
    for metric, res in results.items():
        comparison_table.add_row(
            metric,
            f"{res['correct']}/5 ({res['correct']*20}%)",
            str(res['zero_sim']),
            f"{res['avg_sim']:.3f}"
        )
    
    console.print(comparison_table)
    
    # Test specific problem query
    console.print("\n[bold]Testing 'I can't login' across all metrics:[/bold]")
    problem_table = Table(box=box.SIMPLE)
    problem_table.add_column("Metric", style="cyan")
    problem_table.add_column("Prediction", style="magenta")
    problem_table.add_column("Similarity", style="yellow")
    
    for metric in metrics:
        classifier = MultiMetricClassifier(metric=metric)
        classifier.load_data("data/training_data.yaml")
        result = classifier.predict("I can't login")
        problem_table.add_row(
            metric,
            f"{result.category}:{result.topic}",
            f"{result.similarity_score:.3f}"
        )
    
    console.print(problem_table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries"""
    console.print(f"[bold magenta]Running Sample Test Suite (100 queries) - {classifier.metric.title()} Metric[/bold magenta]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    correct = 0
    zero_sim_wrong = 0
    zero_sim_right = 0
    category_accuracy = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"[magenta]Testing with {classifier.metric}...", total=len(test_cases))
        
        for query, expected_cat, expected_topic in test_cases:
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                    if result.similarity_score == 0:
                        zero_sim_right += 1
                else:
                    if result.similarity_score == 0:
                        zero_sim_wrong += 1
                
                # Track category accuracy
                if expected_cat not in category_accuracy:
                    category_accuracy[expected_cat] = {'correct': 0, 'total': 0}
                category_accuracy[expected_cat]['total'] += 1
                if is_correct:
                    category_accuracy[expected_cat]['correct'] += 1
                
            except Exception as e:
                pass
            
            progress.update(task, advance=1)
    
    # Display results
    accuracy = (correct / len(test_cases)) * 100
    
    console.print(f"\n[bold green]Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)[/bold green]")
    console.print(f"[bold]Metric: {classifier.metric}[/bold]")
    
    if zero_sim_wrong > 0 or zero_sim_right > 0:
        console.print(f"[bold yellow]Zero Similarity Cases:[/bold yellow]")
        console.print(f"  • Wrong (defaulted incorrectly): {zero_sim_wrong}")
        console.print(f"  • Right (lucky default): {zero_sim_right}\n")
    else:
        console.print("[bold green]✓ No zero similarity issues![/bold green]\n")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_about():
    """Display information about these metrics"""
    about_text = """
[bold magenta]Iteration 2.B: Alternative Similarity Metrics[/bold magenta]

[yellow]Metrics Implemented:[/yellow]

[cyan]1. Jaccard Similarity[/cyan]
• Formula: |A ∩ B| / |A ∪ B|
• Set-based metric treating documents as word sets
• Good for: Exact word matching, presence/absence
• Weakness: Ignores word frequency

[cyan]2. Dice Coefficient[/cyan]
• Formula: 2|A ∩ B| / (|A| + |B|)
• Similar to Jaccard but emphasizes overlap more
• Good for: When overlap is more important than union
• Weakness: Still ignores frequency

[cyan]3. Character N-grams[/cyan]
• Uses character sequences (2-4 chars) instead of words
• Good for: Handling typos, partial matches
• Example: "login" → ["lo", "og", "gi", "in"]
• Weakness: May match unrelated words with similar chars

[cyan]4. Weighted Cosine (Enhanced)[/cyan]
• TF-IDF with trigrams and better normalization
• Good for: Capturing phrases and context
• Uses sublinear TF and smooth IDF
• Weakness: Still affected by vocabulary mismatch

[yellow]Hypothesis:[/yellow]
Different metrics will handle the "I can't login" problem differently:
- Jaccard/Dice: May match on "login" alone
- Char n-grams: Should match "login" even with different forms
- Weighted cosine: Trigrams might capture "can't login" phrase
    """
    
    console.print(Panel(about_text, title="[green]About Alternative Metrics[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Initialize with Jaccard by default
    console.print("[magenta]Initializing Multi-Metric Classifier (Jaccard)...[/magenta]")
    classifier = MultiMetricClassifier(metric="jaccard")
    
    try:
        classifier.load_data("data/training_data.yaml")
        console.print(f"[green]✓ Loaded {len(classifier.training_examples)} training examples![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    # Main menu loop
    while True:
        display_welcome(classifier.metric)
        display_menu()
        
        choice = Prompt.ask("[bold]Select option[/bold]", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        
        console.clear()
        
        if choice == "0":
            console.print("[yellow]Thanks for using Query Matcher v2.B! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            quick_match(classifier)
        elif choice == "2":
            # Show top matches not implemented for brevity
            console.print("[yellow]Top matches feature coming soon![/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        elif choice == "3":
            classifier = switch_metric(classifier)
        elif choice == "4":
            run_quick_test(classifier)
        elif choice == "5":
            run_sample_test(classifier)
        elif choice == "6":
            console.print("[yellow]Full test suite would take time - use sample test for now[/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        elif choice == "7":
            compare_all_metrics()
        elif choice == "8":
            console.print("[yellow]Test statistics feature coming soon![/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        elif choice == "9":
            show_about()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        sys.exit(0)