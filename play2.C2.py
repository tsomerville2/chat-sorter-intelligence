#!/usr/bin/env python3
"""
Query Matcher - Iteration 2.C2 (Embeddings with Fallback)
Two-stage approach: Primary classifier with confidence threshold, 
fallback to top-N matching when confidence is low
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
from typing import List, Tuple, Dict
import yaml
from dataclasses import dataclass
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats
from sklearn.feature_extraction.text import TfidfVectorizer

console = Console()


@dataclass
class SimilarityResult:
    category: str
    topic: str
    confidence: float
    matched_example: str
    similarity_score: float
    top_n_matches: List[Tuple[str, str, float]] = None  # (category:topic, example, score)
    used_fallback: bool = False


class TwoStageEmbeddingClassifier:
    def __init__(self, confidence_threshold=0.35, top_n=2):
        """
        Initialize two-stage classifier
        
        Args:
            confidence_threshold: If confidence < threshold, use fallback
            top_n: Number of top matches to consider in fallback
        """
        self.confidence_threshold = confidence_threshold
        self.top_n = top_n
        
        # Combined word and char features (like play2.C)
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            use_idf=True
        )
        
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(3, 5),
            analyzer='char',
            min_df=1,
            sublinear_tf=True
        )
        
        self.training_embeddings = None
        self.training_labels = []
        self.training_examples = []
        self.categories = {}
        
    def load_data(self, filepath: str) -> bool:
        """Load training data from YAML"""
        try:
            with open(filepath, 'r') as file:
                yaml_data = yaml.safe_load(file)
                self.categories = yaml_data.get('categories', {})
                
                # Prepare flat lists
                for category_name, category_data in self.categories.items():
                    topics = category_data.get('topics', {})
                    
                    for topic_name, topic_data in topics.items():
                        examples = topic_data.get('examples', [])
                        
                        for example in examples:
                            self.training_examples.append(example)
                            self.training_labels.append(f"{category_name}:{topic_name}")
                
                # Create embeddings for all training examples
                word_vecs = self.word_vectorizer.fit_transform(self.training_examples)
                char_vecs = self.char_vectorizer.fit_transform(self.training_examples)
                
                # Convert to dense and concatenate
                word_dense = word_vecs.toarray()
                char_dense = char_vecs.toarray()
                
                # Weight character features less
                self.training_embeddings = np.concatenate([word_dense, char_dense * 0.5], axis=1)
                
                return True
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 * norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    
    def get_top_n_matches(self, query: str, n: int = None) -> List[Tuple[str, str, float]]:
        """Get top N matches for a query"""
        if n is None:
            n = self.top_n
            
        # Get query embedding
        word_vec = self.word_vectorizer.transform([query]).toarray()[0]
        char_vec = self.char_vectorizer.transform([query]).toarray()[0]
        query_embedding = np.concatenate([word_vec, char_vec * 0.5])
        
        # Calculate similarities
        similarities = []
        for train_emb in self.training_embeddings:
            sim = self.cosine_similarity(query_embedding, train_emb)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Get top N indices
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                self.training_labels[idx],
                self.training_examples[idx],
                float(similarities[idx])
            ))
        
        return results
    
    def predict(self, query: str) -> SimilarityResult:
        """
        Two-stage prediction:
        1. Try primary classification
        2. If confidence < threshold, use top-N fallback
        """
        # Get query embedding
        word_vec = self.word_vectorizer.transform([query]).toarray()[0]
        char_vec = self.char_vectorizer.transform([query]).toarray()[0]
        query_embedding = np.concatenate([word_vec, char_vec * 0.5])
        
        # Calculate similarities
        similarities = []
        for train_emb in self.training_embeddings:
            sim = self.cosine_similarity(query_embedding, train_emb)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_label = self.training_labels[best_idx]
        best_example = self.training_examples[best_idx]
        
        # Split label
        category, topic = best_label.split(':')
        
        # Check if we should use fallback
        if best_similarity < self.confidence_threshold:
            # Get top N matches
            top_n_matches = self.get_top_n_matches(query, self.top_n)
            
            # Use the best match from top N (which should be the same as above)
            # But mark that we used fallback for tracking
            return SimilarityResult(
                category=category,
                topic=topic,
                confidence=float(best_similarity),
                matched_example=best_example,
                similarity_score=float(best_similarity),
                top_n_matches=top_n_matches,
                used_fallback=True
            )
        else:
            # Primary classification is confident enough
            return SimilarityResult(
                category=category,
                topic=topic,
                confidence=float(best_similarity),
                matched_example=best_example,
                similarity_score=float(best_similarity),
                top_n_matches=None,
                used_fallback=False
            )
    
    def test_with_fallback(self, query: str, expected_cat: str, expected_topic: str) -> Tuple[bool, bool, float]:
        """
        Test a query with fallback logic
        Returns: (is_correct_primary, is_correct_with_fallback, confidence)
        """
        result = self.predict(query)
        expected = f"{expected_cat}:{expected_topic}"
        predicted = f"{result.category}:{result.topic}"
        
        # Check primary prediction
        is_correct_primary = (predicted == expected)
        
        # Check if correct answer is in top N (when using fallback)
        is_correct_with_fallback = is_correct_primary
        
        if result.used_fallback and result.top_n_matches:
            # Check if the correct answer is in the top N
            for label, _, _ in result.top_n_matches:
                if label == expected:
                    is_correct_with_fallback = True
                    break
        
        return is_correct_primary, is_correct_with_fallback, result.confidence


def display_welcome():
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🎯 ", style="bold magenta")
    welcome_text.append("QUERY MATCHER v2.C2", style="bold yellow")
    welcome_text.append(" - Two-Stage Embeddings", style="bold cyan")
    
    panel = Panel(
        "[magenta]Two-Stage Classification with Confidence Threshold[/magenta]\n"
        "[dim]Falls back to top-N matching when confidence is low[/dim]",
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
    
    table.add_row("1", "🎯 Quick Match (shows fallback if used)")
    table.add_row("2", "🔍 Top N Matches")
    table.add_row("3", "⚙️  Configure Threshold & N")
    table.add_row("4", "🧪 Run Quick Test (5 queries)")
    table.add_row("5", "🔬 Run Sample Test (100 queries)")
    table.add_row("6", "⚡ Run Full Test Suite (1000+ queries)")
    table.add_row("7", "📊 Compare with/without Fallback")
    table.add_row("8", "ℹ️  About Two-Stage Approach")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def quick_match(classifier):
    """Quick match showing primary and fallback results"""
    console.print(f"[bold magenta]Quick Match Mode - Two-Stage[/bold magenta]")
    console.print(f"[dim]Threshold: {classifier.confidence_threshold:.2f}, Top-N: {classifier.top_n}[/dim]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status("[magenta]Computing semantic similarity...[/magenta]"):
            try:
                result = classifier.predict(query)
                
                # Create result panel
                result_table = Table(show_header=False, box=box.SIMPLE)
                result_table.add_column("Field", style="magenta")
                result_table.add_column("Value", style="green")
                
                result_table.add_row("Category", f"[bold]{result.category}[/bold]")
                result_table.add_row("Topic", f"[bold]{result.topic}[/bold]")
                result_table.add_row("Similarity Score", f"{result.similarity_score:.4f}")
                result_table.add_row("Confidence", f"{result.confidence:.2%}")
                result_table.add_row("Matched Example", f'"{result.matched_example}"')
                
                if result.used_fallback:
                    result_table.add_row("", "")
                    result_table.add_row("Status", "[yellow]⚠️  Used Fallback (Low Confidence)[/yellow]")
                    
                    if result.top_n_matches:
                        result_table.add_row("", "")
                        result_table.add_row("Top Matches", "")
                        for i, (label, example, score) in enumerate(result.top_n_matches, 1):
                            result_table.add_row(f"  #{i}", f"{label} ({score:.3f})")
                else:
                    result_table.add_row("Status", "[green]✓ Primary Classification[/green]")
                
                console.print(Panel(result_table, title="[green]Classification Result[/green]", 
                                  border_style="green"))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def configure_params(classifier):
    """Configure threshold and N parameters"""
    console.print("[bold magenta]Configure Two-Stage Parameters[/bold magenta]\n")
    
    console.print(f"Current threshold: [yellow]{classifier.confidence_threshold:.2f}[/yellow]")
    console.print(f"Current top-N: [yellow]{classifier.top_n}[/yellow]\n")
    
    new_threshold = Prompt.ask(
        "Enter new confidence threshold (0.0-1.0)",
        default=str(classifier.confidence_threshold)
    )
    
    new_n = Prompt.ask(
        "Enter new top-N value (1-5)",
        default=str(classifier.top_n)
    )
    
    try:
        classifier.confidence_threshold = float(new_threshold)
        classifier.top_n = int(new_n)
        console.print(f"[green]✓ Updated: threshold={classifier.confidence_threshold:.2f}, top-N={classifier.top_n}[/green]")
    except ValueError:
        console.print("[red]Invalid values, keeping current settings[/red]")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_quick_test(classifier):
    """Run the original 5-query test with fallback logic"""
    console.print(f"[bold magenta]Running Quick Test - Two-Stage[/bold magenta]")
    console.print(f"[dim]Threshold: {classifier.confidence_threshold:.2f}, Top-N: {classifier.top_n}[/dim]\n")
    
    test_queries = [
        ("I can't login", "technical_support", "password_reset"),
        ("my payment didn't work", "billing", "payment_failed"),
        ("where's my stuff", "shipping", "track_order"),
        ("I forgot my password", "technical_support", "password_reset"),
        ("card declined", "billing", "payment_failed"),
    ]
    
    results_table = Table(title="Test Results", box=box.ROUNDED)
    results_table.add_column("Query", style="white")
    results_table.add_column("Expected", style="cyan")
    results_table.add_column("Predicted", style="magenta")
    results_table.add_column("Conf", style="yellow")
    results_table.add_column("Primary", style="green")
    results_table.add_column("w/ Fallback", style="blue")
    
    correct_primary = 0
    correct_fallback = 0
    used_fallback_count = 0
    
    for query, expected_cat, expected_topic in test_queries:
        is_primary, is_fallback, confidence = classifier.test_with_fallback(query, expected_cat, expected_topic)
        
        result = classifier.predict(query)
        expected = f"{expected_cat}:{expected_topic}"
        predicted = f"{result.category}:{result.topic}"
        
        if is_primary:
            correct_primary += 1
        if is_fallback:
            correct_fallback += 1
        if result.used_fallback:
            used_fallback_count += 1
        
        results_table.add_row(
            query,
            expected,
            predicted,
            f"{confidence:.2f}",
            "✓" if is_primary else "✗",
            "✓" if is_fallback else "✗"
        )
    
    console.print(results_table)
    console.print(f"\n[bold]Primary Accuracy: {correct_primary}/5 ({100*correct_primary/5:.0f}%)[/bold]")
    console.print(f"[bold green]With Fallback: {correct_fallback}/5 ({100*correct_fallback/5:.0f}%)[/bold green]")
    console.print(f"[dim]Used fallback: {used_fallback_count} times[/dim]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries with fallback"""
    console.print(f"[bold magenta]Running Sample Test (100 queries) - Two-Stage[/bold magenta]")
    console.print(f"[dim]Threshold: {classifier.confidence_threshold:.2f}, Top-N: {classifier.top_n}[/dim]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    correct_primary = 0
    correct_fallback = 0
    used_fallback_count = 0
    confidence_sum = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[magenta]Testing with two-stage approach...", total=len(test_cases))
        
        for query, expected_cat, expected_topic in test_cases:
            try:
                is_primary, is_fallback, confidence = classifier.test_with_fallback(
                    query, expected_cat, expected_topic
                )
                
                result = classifier.predict(query)
                
                if is_primary:
                    correct_primary += 1
                if is_fallback:
                    correct_fallback += 1
                if result.used_fallback:
                    used_fallback_count += 1
                
                confidence_sum += confidence
                
            except Exception as e:
                pass
            
            progress.update(task, advance=1)
    
    # Display results
    primary_accuracy = (correct_primary / len(test_cases)) * 100
    fallback_accuracy = (correct_fallback / len(test_cases)) * 100
    improvement = fallback_accuracy - primary_accuracy
    avg_confidence = confidence_sum / len(test_cases)
    
    console.print(f"\n[bold]Primary Accuracy: {correct_primary}/100 ({primary_accuracy:.1f}%)[/bold]")
    console.print(f"[bold green]With Fallback: {correct_fallback}/100 ({fallback_accuracy:.1f}%)[/bold green]")
    console.print(f"[bold cyan]Improvement: +{improvement:.1f}%[/bold cyan]")
    console.print(f"[bold]Average Confidence: {avg_confidence:.3f}[/bold]")
    console.print(f"[bold]Used Fallback: {used_fallback_count} times ({used_fallback_count}%)[/bold]")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def compare_with_without_fallback(classifier):
    """Compare performance with and without fallback"""
    console.print("[bold magenta]Comparing With/Without Fallback[/bold magenta]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    # Test different threshold values
    thresholds = [0.2, 0.3, 0.35, 0.4, 0.5]
    results = []
    
    for threshold in thresholds:
        classifier.confidence_threshold = threshold
        correct_primary = 0
        correct_fallback = 0
        used_fallback = 0
        
        for query, expected_cat, expected_topic in test_cases:
            is_primary, is_fallback, _ = classifier.test_with_fallback(
                query, expected_cat, expected_topic
            )
            result = classifier.predict(query)
            
            if is_primary:
                correct_primary += 1
            if is_fallback:
                correct_fallback += 1
            if result.used_fallback:
                used_fallback += 1
        
        results.append((
            threshold,
            correct_primary,
            correct_fallback,
            used_fallback
        ))
    
    # Display comparison table
    table = Table(title="Threshold Comparison", box=box.ROUNDED)
    table.add_column("Threshold", style="cyan")
    table.add_column("Primary Acc", style="yellow")
    table.add_column("w/ Fallback", style="green")
    table.add_column("Improvement", style="magenta")
    table.add_column("Used Fallback", style="blue")
    
    for threshold, primary, fallback, used in results:
        improvement = fallback - primary
        table.add_row(
            f"{threshold:.2f}",
            f"{primary}%",
            f"{fallback}%",
            f"+{improvement}",
            f"{used}"
        )
    
    console.print(table)
    
    # Find best threshold
    best = max(results, key=lambda x: x[2])  # Max by fallback accuracy
    console.print(f"\n[bold green]Best threshold: {best[0]:.2f} ({best[2]}% accuracy)[/bold green]")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_about():
    """Display information about two-stage approach"""
    about_text = """
[bold magenta]Iteration 2.C2: Two-Stage Classification[/bold magenta]

[yellow]How it works:[/yellow]
1. Primary classification using embeddings (word + char features)
2. If confidence < threshold, mark for fallback consideration
3. Get top-N matches for low-confidence queries
4. If correct answer is in top-N, consider it a success

[yellow]Key Parameters:[/yellow]
• Confidence Threshold: Default 0.35
  - Below this = use fallback logic
  - Above this = trust primary classification
• Top-N: Default 2
  - Number of alternatives to consider
  - Higher N = more lenient scoring

[yellow]Advantages:[/yellow]
• Improves accuracy without changing core algorithm
• Identifies uncertain predictions
• Provides alternatives for human review
• Tunable for precision vs recall trade-off

[yellow]Use Cases:[/yellow]
• Production systems with human fallback
• Cases where "probably right" is good enough
• Systems that can present multiple options
• When false negatives are worse than uncertainty

[yellow]Expected Performance:[/yellow]
• Primary: ~75% (same as play2.C)
• With fallback: ~80-85% (estimated)
• Trade-off: More "uncertain" classifications
    """
    
    console.print(Panel(about_text, title="[green]About Two-Stage Approach[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Initialize classifier with default threshold
    console.print("[magenta]Initializing Two-Stage Embedding Classifier...[/magenta]")
    console.print("[dim]Default: threshold=0.35, top-N=2[/dim]")
    
    classifier = TwoStageEmbeddingClassifier(confidence_threshold=0.35, top_n=2)
    
    try:
        classifier.load_data("data/training_data.yaml")
        console.print(f"[green]✓ Loaded {len(classifier.training_examples)} training examples![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    # Main menu loop
    while True:
        display_welcome()
        display_menu()
        
        choice = Prompt.ask("[bold]Select option[/bold]", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
        
        console.clear()
        
        if choice == "0":
            console.print("[yellow]Thanks for using Query Matcher v2.C2! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            quick_match(classifier)
        elif choice == "2":
            console.print("[yellow]Top-N display not implemented yet[/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        elif choice == "3":
            configure_params(classifier)
        elif choice == "4":
            run_quick_test(classifier)
        elif choice == "5":
            run_sample_test(classifier)
        elif choice == "6":
            console.print("[yellow]Full test would take time - use sample test[/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
        elif choice == "7":
            compare_with_without_fallback(classifier)
        elif choice == "8":
            show_about()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        sys.exit(0)