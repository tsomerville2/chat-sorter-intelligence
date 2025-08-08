#!/usr/bin/env python3
"""
Query Matcher - Iteration 2.C (Embedding Models)
Experimental: Using Sentence Transformers for semantic similarity
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

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    
    # Fallback to simple word embeddings
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

console = Console()


@dataclass
class SimilarityResult:
    category: str
    topic: str
    confidence: float
    matched_example: str
    similarity_score: float
    model_used: str


class EmbeddingClassifier:
    def __init__(self, model_name='simple'):
        self.model_name = model_name
        
        if EMBEDDINGS_AVAILABLE and model_name != 'simple':
            console.print(f"[yellow]Loading embedding model {model_name}... (first time may download ~90MB)[/yellow]")
            try:
                self.model = SentenceTransformer(model_name)
                self.use_embeddings = True
            except Exception as e:
                console.print(f"[red]Failed to load model: {e}[/red]")
                console.print("[yellow]Falling back to simple embeddings[/yellow]")
                self.use_embeddings = False
                self._init_simple_embeddings()
        else:
            self.use_embeddings = False
            self._init_simple_embeddings()
        
        self.training_embeddings = None
        self.training_labels = []
        self.training_examples = []
        self.categories = {}
    
    def _init_simple_embeddings(self):
        """Initialize simple word embeddings as fallback"""
        # Use character n-grams plus word n-grams for better coverage
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            analyzer='word',
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
                if self.use_embeddings:
                    self.training_embeddings = self.model.encode(self.training_examples)
                else:
                    # Combine word and char features
                    word_vecs = self.vectorizer.fit_transform(self.training_examples)
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
    
    def predict(self, query: str) -> SimilarityResult:
        """Find most similar example using embeddings"""
        if self.training_embeddings is None:
            raise ValueError("Classifier not trained")
        
        # Get query embedding
        if self.use_embeddings:
            query_embedding = self.model.encode([query])[0]
        else:
            # Combine word and char features for query
            word_vec = self.vectorizer.transform([query]).toarray()[0]
            char_vec = self.char_vectorizer.transform([query]).toarray()[0]
            query_embedding = np.concatenate([word_vec, char_vec * 0.5])
        
        # Calculate similarities
        similarities = []
        for train_emb in self.training_embeddings:
            sim = self.cosine_similarity(query_embedding, train_emb)
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
            model_used=self.model_name if self.use_embeddings else "simple_embeddings"
        )
    
    def predict_top_k(self, query: str, k: int = 3) -> List[SimilarityResult]:
        """Get top K most similar matches"""
        if self.training_embeddings is None:
            raise ValueError("Classifier not trained")
        
        # Get query embedding
        if self.use_embeddings:
            query_embedding = self.model.encode([query])[0]
        else:
            word_vec = self.vectorizer.transform([query]).toarray()[0]
            char_vec = self.char_vectorizer.transform([query]).toarray()[0]
            query_embedding = np.concatenate([word_vec, char_vec * 0.5])
        
        # Calculate similarities
        similarities = []
        for train_emb in self.training_embeddings:
            sim = self.cosine_similarity(query_embedding, train_emb)
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
                model_used=self.model_name if self.use_embeddings else "simple_embeddings"
            ))
        
        return results


def display_welcome(model_name):
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🎯 ", style="bold magenta")
    welcome_text.append("QUERY MATCHER v2.C", style="bold yellow")
    welcome_text.append(" - Semantic Embeddings", style="bold cyan")
    
    if not EMBEDDINGS_AVAILABLE:
        desc = "Using simple combined embeddings (sentence-transformers not available)"
    else:
        desc = f"Using {model_name} for semantic similarity"
    
    panel = Panel(
        f"[magenta]Experimental: {desc}[/magenta]\n"
        "[dim]Semantic matching understands meaning, not just words[/dim]",
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
    table.add_row("3", "🧪 Run Quick Test (5 queries)")
    table.add_row("4", "🔬 Run Sample Test (100 queries)")
    table.add_row("5", "⚡ Run Full Test Suite (1000+ queries)")
    table.add_row("6", "📈 Show Test Statistics")
    table.add_row("7", "ℹ️  About Embeddings")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def quick_match(classifier):
    """Quick match showing the most similar example"""
    model = classifier.model_name if classifier.use_embeddings else "simple"
    console.print(f"[bold magenta]Quick Match Mode - {model} embeddings[/bold magenta]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status("[magenta]Computing semantic similarity...[/magenta]"):
            try:
                result = classifier.predict(query)
                
                # Create detailed result panel
                result_table = Table(show_header=False, box=box.SIMPLE)
                result_table.add_column("Field", style="magenta")
                result_table.add_column("Value", style="green")
                
                result_table.add_row("Category", f"[bold]{result.category}[/bold]")
                result_table.add_row("Topic", f"[bold]{result.topic}[/bold]")
                result_table.add_row("Similarity Score", f"{result.similarity_score:.4f}")
                result_table.add_row("Confidence", f"{result.confidence:.2%}")
                result_table.add_row("Model", result.model_used)
                result_table.add_row("Matched Example", f'"{result.matched_example}"')
                
                # Warn if low similarity
                if result.similarity_score < 0.3:
                    console.print("[bold yellow]⚠️  Low similarity - uncertain match[/bold yellow]")
                
                console.print(Panel(result_table, title="[green]✓ Best Match Found[/green]", 
                                  border_style="green"))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def show_top_matches(classifier):
    """Show top 3 most similar matches"""
    console.print("[bold magenta]Top 3 Matches Mode[/bold magenta]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status("[magenta]Finding semantically similar examples...[/magenta]"):
            try:
                results = classifier.predict_top_k(query, k=3)
                
                console.print(f"\n[bold]Query:[/bold] '{query}'\n")
                
                for i, result in enumerate(results, 1):
                    # Create match panel
                    match_table = Table(show_header=False, box=box.SIMPLE)
                    match_table.add_column("", style="magenta", width=15)
                    match_table.add_column("", style="white")
                    
                    match_table.add_row("Category", result.category)
                    match_table.add_row("Topic", result.topic)
                    match_table.add_row("Similarity", f"{result.similarity_score:.4f}")
                    match_table.add_row("Example", f'"{result.matched_example[:50]}..."' 
                                      if len(result.matched_example) > 50 else f'"{result.matched_example}"')
                    
                    color = "green" if i == 1 else "yellow" if i == 2 else "cyan"
                    console.print(Panel(match_table, title=f"[{color}]Match #{i}[/{color}]", 
                                      border_style=color))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def run_quick_test(classifier):
    """Run the original 5-query test"""
    model = classifier.model_name if classifier.use_embeddings else "simple"
    console.print(f"[bold magenta]Running Quick Test Suite - {model} embeddings[/bold magenta]\n")
    
    test_queries = [
        ("I can't login", "technical_support", "password_reset"),
        ("my payment didn't work", "billing", "payment_failed"),
        ("where's my stuff", "shipping", "track_order"),
        ("I forgot my password", "technical_support", "password_reset"),
        ("card declined", "billing", "payment_failed"),
    ]
    
    results_table = Table(title=f"Test Results - {model}", box=box.ROUNDED)
    results_table.add_column("Query", style="white")
    results_table.add_column("Expected", style="cyan")
    results_table.add_column("Predicted", style="magenta")
    results_table.add_column("Similarity", style="yellow")
    results_table.add_column("✓/✗", style="green")
    
    correct = 0
    low_sim_count = 0
    
    for query, expected_cat, expected_topic in test_queries:
        result = classifier.predict(query)
        expected = f"{expected_cat}:{expected_topic}"
        predicted = f"{result.category}:{result.topic}"
        is_correct = expected == predicted
        if is_correct:
            correct += 1
        if result.similarity_score < 0.3:
            low_sim_count += 1
        
        results_table.add_row(
            query,
            expected,
            predicted,
            f"{result.similarity_score:.3f}",
            "✓" if is_correct else "✗"
        )
    
    console.print(results_table)
    console.print(f"\n[bold]Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.1f}%)[/bold]")
    console.print(f"[bold]Model: {model}[/bold]")
    if low_sim_count > 0:
        console.print(f"[bold yellow]⚠️  {low_sim_count} queries had low similarity (<0.3)[/bold yellow]")
    else:
        console.print("[bold green]✓ All queries had good similarity scores![/bold green]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries"""
    model = classifier.model_name if classifier.use_embeddings else "simple"
    console.print(f"[bold magenta]Running Sample Test Suite (100 queries) - {model} embeddings[/bold magenta]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    correct = 0
    category_accuracy = {}
    similarity_sum = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"[magenta]Testing with {model}...", total=len(test_cases))
        
        for query, expected_cat, expected_topic in test_cases:
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                
                similarity_sum += result.similarity_score
                
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
    avg_similarity = similarity_sum / len(test_cases)
    
    console.print(f"\n[bold green]Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)[/bold green]")
    console.print(f"[bold]Average Similarity: {avg_similarity:.3f}[/bold]")
    console.print(f"[bold]Model: {model}[/bold]\n")
    
    # Category breakdown
    cat_table = Table(title="Accuracy by Category", box=box.ROUNDED)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Correct", style="green")
    cat_table.add_column("Total", style="yellow")
    cat_table.add_column("Accuracy", style="magenta")
    
    for cat, stats in sorted(category_accuracy.items()):
        cat_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        cat_table.add_row(
            cat,
            str(stats['correct']),
            str(stats['total']),
            f"{cat_accuracy:.1f}%"
        )
    
    console.print(cat_table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_full_test(classifier):
    """Run test on all 1000+ queries"""
    model = classifier.model_name if classifier.use_embeddings else "simple"
    console.print(f"[bold magenta]Running Full Test Suite (1000+ queries) - {model} embeddings[/bold magenta]\n")
    console.print("[yellow]This may take a few minutes...[/yellow]\n")
    
    test_cases = load_comprehensive_test_data()
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    console.print(f"[dim]Loaded {len(test_cases)} test cases[/dim]\n")
    
    correct = 0
    category_accuracy = {}
    similarity_sum = 0
    errors = []
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console
    ) as progress:
        task = progress.add_task(f"[magenta]Testing with {model}...", total=len(test_cases))
        
        for i, (query, expected_cat, expected_topic) in enumerate(test_cases):
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                else:
                    if len(errors) < 10:
                        errors.append({
                            'query': query[:50],
                            'expected': f"{expected_cat}:{expected_topic}",
                            'predicted': f"{result.category}:{result.topic}",
                            'similarity': result.similarity_score
                        })
                
                similarity_sum += result.similarity_score
                
                # Track category accuracy
                if expected_cat not in category_accuracy:
                    category_accuracy[expected_cat] = {'correct': 0, 'total': 0}
                category_accuracy[expected_cat]['total'] += 1
                if result.category == expected_cat:
                    category_accuracy[expected_cat]['correct'] += 1
                
            except Exception as e:
                pass
            
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    # Display results
    accuracy = (correct / len(test_cases)) * 100
    avg_similarity = similarity_sum / len(test_cases)
    
    console.print(f"\n[bold green]═══ Final Results ═══[/bold green]")
    console.print(f"[bold]Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)[/bold]")
    console.print(f"[bold]Average Similarity: {avg_similarity:.3f}[/bold]")
    console.print(f"[bold]Model: {model}[/bold]")
    console.print(f"[bold]Time Taken: {elapsed_time:.2f} seconds[/bold]")
    console.print(f"[bold]Speed: {len(test_cases)/elapsed_time:.0f} queries/second[/bold]\n")
    
    # Category breakdown
    cat_table = Table(title="Accuracy by Category", box=box.ROUNDED)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Accuracy", style="magenta")
    cat_table.add_column("Correct/Total", style="yellow")
    
    for cat, stats in sorted(category_accuracy.items()):
        cat_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        cat_table.add_row(
            cat,
            f"{cat_accuracy:.1f}%",
            f"{stats['correct']}/{stats['total']}"
        )
    
    console.print(cat_table)
    
    # Show sample errors
    if errors:
        console.print("\n[bold red]Sample Errors (First 10):[/bold red]")
        error_table = Table(box=box.SIMPLE)
        error_table.add_column("Query", style="white", width=40)
        error_table.add_column("Expected", style="green")
        error_table.add_column("Got", style="red")
        error_table.add_column("Sim", style="yellow")
        
        for err in errors[:10]:
            error_table.add_row(
                err['query'],
                err['expected'],
                err['predicted'],
                f"{err['similarity']:.3f}"
            )
        
        console.print(error_table)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_test_stats():
    """Show statistics about the test dataset"""
    console.print("[bold magenta]Test Dataset Statistics[/bold magenta]\n")
    
    stats = get_test_stats()
    
    console.print(f"[bold]Total Test Cases:[/bold] {stats['total_cases']}")
    console.print(f"[bold]Unique Categories:[/bold] {stats['unique_categories']}")
    console.print(f"[bold]Unique Topics:[/bold] {stats['unique_topics']}\n")
    
    # Category distribution
    cat_table = Table(title="Test Cases by Category", box=box.ROUNDED)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", style="yellow")
    cat_table.add_column("Percentage", style="magenta")
    
    total = stats['total_cases']
    for cat, count in sorted(stats['category_distribution'].items()):
        percentage = (count / total) * 100
        cat_table.add_row(cat, str(count), f"{percentage:.1f}%")
    
    console.print(cat_table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_about():
    """Display information about embeddings"""
    about_text = """
[bold magenta]Iteration 2.C: Semantic Embeddings[/bold magenta]

[yellow]What are Embeddings?[/yellow]
• Dense vector representations of text
• Capture semantic meaning, not just words
• Similar meanings have similar vectors
• Can match "I can't login" with "unable to access account"

[yellow]Models Available:[/yellow]
• all-MiniLM-L6-v2: Fast, good quality (384 dims)
• all-mpnet-base-v2: Higher quality (768 dims)
• simple: Fallback using TF-IDF + char n-grams

[yellow]Advantages:[/yellow]
• Understands synonyms and paraphrases
• Handles typos and variations
• No zero similarity issues
• Works across languages (multilingual models)

[yellow]Disadvantages:[/yellow]
• Requires model download (~90MB)
• Slower than TF-IDF (but still fast)
• May need GPU for large datasets
• Can be fooled by semantically similar but wrong contexts

[yellow]Expected Performance:[/yellow]
• Should achieve 85-95% accuracy
• "I can't login" should correctly match password_reset
• Better handling of novel phrasings
• More consistent similarity scores
    """
    
    if not EMBEDDINGS_AVAILABLE:
        about_text += "\n[red]Note: sentence-transformers not installed[/red]\n"
        about_text += "[yellow]Install with: pip install sentence-transformers[/yellow]"
    
    console.print(Panel(about_text, title="[green]About Semantic Embeddings[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Initialize classifier
    if EMBEDDINGS_AVAILABLE:
        # Try to use a small, fast model
        model_name = "all-MiniLM-L6-v2"
        console.print(f"[magenta]Initializing Embedding Classifier ({model_name})...[/magenta]")
    else:
        model_name = "simple"
        console.print("[yellow]Sentence transformers not available. Using simple embeddings...[/yellow]")
    
    classifier = EmbeddingClassifier(model_name=model_name)
    
    try:
        classifier.load_data("data/training_data.yaml")
        console.print(f"[green]✓ Loaded {len(classifier.training_examples)} training examples![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    # Main menu loop
    while True:
        display_welcome(classifier.model_name)
        display_menu()
        
        choice = Prompt.ask("[bold]Select option[/bold]", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
        
        console.clear()
        
        if choice == "0":
            console.print("[yellow]Thanks for using Query Matcher v2.C! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            quick_match(classifier)
        elif choice == "2":
            show_top_matches(classifier)
        elif choice == "3":
            run_quick_test(classifier)
        elif choice == "4":
            run_sample_test(classifier)
        elif choice == "5":
            run_full_test(classifier)
        elif choice == "6":
            show_test_stats()
        elif choice == "7":
            show_about()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        sys.exit(0)