#!/usr/bin/env python3
"""
Query Matcher - Iteration 2.C3 (Production-Ready with Ambiguity Detection)
Uses sentence transformers for 78% base accuracy, returns Pydantic models
for single answers or multiple choices when confidence is low
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
from typing import List, Tuple, Dict, Union, Optional
import yaml
from pydantic import BaseModel, Field
from dataclasses import dataclass
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer

console = Console()


# Pydantic Models for API-ready responses
class MatchDetails(BaseModel):
    """Details about a single match"""
    category: str
    topic: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_example: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class SingleAnswer(BaseModel):
    """Response when we have high confidence in a single answer"""
    answer_type: str = "single"
    match: MatchDetails
    model_used: str


class MultipleAnswers(BaseModel):
    """Response when confidence is low and we present alternatives"""
    answer_type: str = "multiple"
    matches: List[MatchDetails]
    model_used: str
    reason: str = "Low confidence - presenting top alternatives"


# Type alias for the return type
ClassificationResult = Union[SingleAnswer, MultipleAnswers]


class ProductionEmbeddingClassifier:
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 confidence_threshold: float = 0.6,
                 alternatives_count: int = 3):
        """
        Initialize production classifier with ambiguity detection
        
        Args:
            model_name: Sentence transformer model to use
            confidence_threshold: Below this, return multiple alternatives
            alternatives_count: Number of alternatives to return when uncertain
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.alternatives_count = alternatives_count
        
        if EMBEDDINGS_AVAILABLE:
            try:
                console.print(f"[yellow]Loading {model_name}... (may download ~90MB first time)[/yellow]")
                self.model = SentenceTransformer(model_name)
                self.use_embeddings = True
            except Exception as e:
                console.print(f"[red]Failed to load model: {e}[/red]")
                console.print("[yellow]Falling back to TF-IDF + char n-grams[/yellow]")
                self.use_embeddings = False
                self._init_fallback_embeddings()
        else:
            console.print("[yellow]sentence-transformers not installed, using fallback[/yellow]")
            self.use_embeddings = False
            self._init_fallback_embeddings()
        
        self.training_embeddings = None
        self.training_labels = []
        self.training_examples = []
        self.categories = {}
    
    def _init_fallback_embeddings(self):
        """Initialize TF-IDF + char n-grams as fallback"""
        self.word_vectorizer = TfidfVectorizer(
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
                    # Fallback to TF-IDF + char n-grams
                    word_vecs = self.word_vectorizer.fit_transform(self.training_examples)
                    char_vecs = self.char_vectorizer.fit_transform(self.training_examples)
                    
                    word_dense = word_vecs.toarray()
                    char_dense = char_vecs.toarray()
                    
                    # Combine with char features weighted less
                    self.training_embeddings = np.concatenate([word_dense, char_dense * 0.5], axis=1)
                
                return True
                
        except Exception as e:
            console.print(f"[red]Error loading data: {e}[/red]")
            return False
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 * norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    
    def get_top_k_matches(self, query: str, k: int) -> List[MatchDetails]:
        """Get top K matches for a query"""
        # Get query embedding
        if self.use_embeddings:
            query_embedding = self.model.encode([query])[0]
        else:
            word_vec = self.word_vectorizer.transform([query]).toarray()[0]
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
        
        matches = []
        for idx in top_indices:
            label = self.training_labels[idx]
            category, topic = label.split(':')
            
            matches.append(MatchDetails(
                category=category,
                topic=topic,
                confidence=float(similarities[idx]),
                matched_example=self.training_examples[idx],
                similarity_score=float(similarities[idx])
            ))
        
        return matches
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query, returning either a single answer or multiple options
        
        Returns:
            SingleAnswer if confidence >= threshold
            MultipleAnswers if confidence < threshold (ambiguous)
        """
        if self.training_embeddings is None:
            raise ValueError("Classifier not trained")
        
        # Get top matches
        top_matches = self.get_top_k_matches(query, self.alternatives_count)
        
        if not top_matches:
            # Shouldn't happen, but handle gracefully
            return MultipleAnswers(
                matches=[],
                model_used=self.model_name if self.use_embeddings else "tfidf_char_ngrams",
                reason="No matches found"
            )
        
        best_match = top_matches[0]
        
        # Determine if we're confident enough for a single answer
        if best_match.confidence >= self.confidence_threshold:
            return SingleAnswer(
                match=best_match,
                model_used=self.model_name if self.use_embeddings else "tfidf_char_ngrams"
            )
        else:
            # Low confidence - return multiple options
            # Check if there's meaningful separation between top choices
            if len(top_matches) > 1:
                confidence_gap = best_match.confidence - top_matches[1].confidence
                if confidence_gap < 0.1:  # Very close scores
                    reason = f"Ambiguous query - top {len(top_matches)} matches have similar confidence"
                else:
                    reason = f"Low confidence ({best_match.confidence:.2f}) - presenting alternatives"
            else:
                reason = "Low confidence - only one match available"
            
            return MultipleAnswers(
                matches=top_matches,
                model_used=self.model_name if self.use_embeddings else "tfidf_char_ngrams",
                reason=reason
            )
    
    def evaluate_with_ambiguity(self, query: str, expected_cat: str, expected_topic: str) -> Dict:
        """
        Evaluate a query considering ambiguity detection
        
        Returns dict with:
        - is_correct: True if single answer is correct OR correct answer is in multiple choices
        - answer_type: 'single' or 'multiple'
        - confidence: Best match confidence
        - in_alternatives: True if correct answer was in alternatives
        """
        result = self.classify(query)
        expected = f"{expected_cat}:{expected_topic}"
        
        if result.answer_type == "single":
            predicted = f"{result.match.category}:{result.match.topic}"
            return {
                'is_correct': predicted == expected,
                'answer_type': 'single',
                'confidence': result.match.confidence,
                'in_alternatives': False
            }
        else:
            # Check if correct answer is in alternatives
            for match in result.matches:
                predicted = f"{match.category}:{match.topic}"
                if predicted == expected:
                    return {
                        'is_correct': True,
                        'answer_type': 'multiple',
                        'confidence': result.matches[0].confidence if result.matches else 0,
                        'in_alternatives': True
                    }
            
            return {
                'is_correct': False,
                'answer_type': 'multiple',
                'confidence': result.matches[0].confidence if result.matches else 0,
                'in_alternatives': False
            }


def display_welcome(classifier):
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🎯 ", style="bold magenta")
    welcome_text.append("QUERY MATCHER v2.C3", style="bold yellow")
    welcome_text.append(" - Production Ready", style="bold cyan")
    
    model_desc = classifier.model_name if classifier.use_embeddings else "TF-IDF + Char N-grams"
    
    panel = Panel(
        f"[magenta]Ambiguity-Aware Classification with Pydantic Models[/magenta]\n"
        f"[dim]Model: {model_desc} | Threshold: {classifier.confidence_threshold:.2f}[/dim]",
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
    
    table.add_row("1", "🎯 Interactive Query (shows API response)")
    table.add_row("2", "⚙️  Configure Threshold")
    table.add_row("3", "🧪 Run Quick Test (5 queries)")
    table.add_row("4", "🔬 Run Sample Test (100 queries)")
    table.add_row("5", "⚡ Run Full Test Suite (1000+ queries)")
    table.add_row("6", "📊 Analyze Ambiguity Distribution")
    table.add_row("7", "🔍 Test Specific Ambiguous Cases")
    table.add_row("8", "ℹ️  About Production Mode")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def interactive_query(classifier):
    """Interactive query mode showing API-ready responses"""
    console.print("[bold magenta]Interactive Query Mode[/bold magenta]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status("[magenta]Classifying...[/magenta]"):
            try:
                result = classifier.classify(query)
                
                # Display the Pydantic model response
                if isinstance(result, SingleAnswer):
                    # Single answer display
                    table = Table(show_header=False, box=box.SIMPLE)
                    table.add_column("Field", style="magenta")
                    table.add_column("Value", style="green")
                    
                    table.add_row("Response Type", "Single Answer")
                    table.add_row("Category", f"[bold]{result.match.category}[/bold]")
                    table.add_row("Topic", f"[bold]{result.match.topic}[/bold]")
                    table.add_row("Confidence", f"{result.match.confidence:.2%}")
                    table.add_row("Matched Example", f'"{result.match.matched_example}"')
                    table.add_row("Model", result.model_used)
                    
                    console.print(Panel(table, title="[green]✓ High Confidence Result[/green]", 
                                      border_style="green"))
                    
                    # Show the raw Pydantic output
                    console.print("\n[dim]API Response (Pydantic):[/dim]")
                    console.print(f"[dim]{result.model_dump_json(indent=2)}[/dim]")
                    
                else:  # MultipleAnswers
                    # Multiple answers display
                    console.print(f"[yellow]⚠️  {result.reason}[/yellow]\n")
                    
                    for i, match in enumerate(result.matches, 1):
                        table = Table(show_header=False, box=box.SIMPLE)
                        table.add_column("", style="magenta", width=15)
                        table.add_column("", style="white")
                        
                        table.add_row("Category", match.category)
                        table.add_row("Topic", match.topic)
                        table.add_row("Confidence", f"{match.confidence:.2%}")
                        table.add_row("Example", f'"{match.matched_example[:50]}..."' 
                                    if len(match.matched_example) > 50 else f'"{match.matched_example}"')
                        
                        color = "yellow" if i == 1 else "cyan"
                        console.print(Panel(table, title=f"[{color}]Option #{i}[/{color}]", 
                                          border_style=color))
                    
                    # Show the raw Pydantic output
                    console.print("\n[dim]API Response (Pydantic):[/dim]")
                    console.print(f"[dim]{result.model_dump_json(indent=2)}[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def configure_threshold(classifier):
    """Configure confidence threshold"""
    console.print("[bold magenta]Configure Confidence Threshold[/bold magenta]\n")
    
    console.print(f"Current threshold: [yellow]{classifier.confidence_threshold:.2f}[/yellow]")
    console.print(f"Current alternatives count: [yellow]{classifier.alternatives_count}[/yellow]\n")
    
    console.print("[dim]Higher threshold = more queries return multiple answers[/dim]")
    console.print("[dim]Lower threshold = more queries return single answer[/dim]\n")
    
    new_threshold = Prompt.ask(
        "Enter new confidence threshold (0.0-1.0)",
        default=str(classifier.confidence_threshold)
    )
    
    new_count = Prompt.ask(
        "Enter alternatives count (2-5)",
        default=str(classifier.alternatives_count)
    )
    
    try:
        classifier.confidence_threshold = float(new_threshold)
        classifier.alternatives_count = int(new_count)
        console.print(f"[green]✓ Updated: threshold={classifier.confidence_threshold:.2f}, "
                     f"alternatives={classifier.alternatives_count}[/green]")
    except ValueError:
        console.print("[red]Invalid values, keeping current settings[/red]")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_quick_test(classifier):
    """Run the 5-query test with ambiguity awareness"""
    console.print("[bold magenta]Running Quick Test - Ambiguity Aware[/bold magenta]\n")
    
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
    results_table.add_column("Type", style="magenta")
    results_table.add_column("Confidence", style="yellow")
    results_table.add_column("Result", style="green")
    
    single_correct = 0
    multiple_correct = 0
    single_count = 0
    multiple_count = 0
    
    for query, expected_cat, expected_topic in test_queries:
        eval_result = classifier.evaluate_with_ambiguity(query, expected_cat, expected_topic)
        result = classifier.classify(query)
        
        expected = f"{expected_cat}:{expected_topic}"
        
        if eval_result['answer_type'] == 'single':
            single_count += 1
            if eval_result['is_correct']:
                single_correct += 1
                status = "✓"
            else:
                status = "✗"
        else:
            multiple_count += 1
            if eval_result['is_correct']:
                multiple_correct += 1
                status = "✓ (in list)"
            else:
                status = "✗"
        
        results_table.add_row(
            query,
            expected,
            eval_result['answer_type'],
            f"{eval_result['confidence']:.2f}",
            status
        )
    
    console.print(results_table)
    
    total_correct = single_correct + multiple_correct
    console.print(f"\n[bold]Overall Success: {total_correct}/5 ({100*total_correct/5:.0f}%)[/bold]")
    console.print(f"[bold]Single answers: {single_count} ({single_correct} correct)[/bold]")
    console.print(f"[bold]Multiple answers: {multiple_count} ({multiple_correct} correct)[/bold]")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_full_test(classifier):
    """Run test on all 1000+ queries with comprehensive confusion analysis"""
    console.print("[bold magenta]Running Full Test Suite with Confusion Analysis (1000+ queries)[/bold magenta]\n")
    console.print("[yellow]This may take a few minutes...[/yellow]\n")
    
    test_cases = load_comprehensive_test_data()
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    console.print(f"[dim]Loaded {len(test_cases)} test cases[/dim]\n")
    
    single_correct = 0
    multiple_correct = 0
    single_count = 0
    multiple_count = 0
    category_stats = {}
    errors = []
    
    # Confusion tracking
    confusion_matrix = {}  # {(expected, predicted): count}
    topic_confusion = {}   # {(expected_topic, predicted_topic): count}
    confidence_buckets = {
        'correct_high': [],    # Correct with conf >= 0.7
        'correct_medium': [],  # Correct with 0.4 <= conf < 0.7
        'correct_low': [],     # Correct with conf < 0.4
        'incorrect_high': [],  # Wrong with conf >= 0.7 (concerning!)
        'incorrect_medium': [], # Wrong with 0.4 <= conf < 0.7
        'incorrect_low': []    # Wrong with conf < 0.4 (expected)
    }
    
    # Pattern tracking
    error_patterns = {
        'high_conf_errors': [],  # Confidently wrong predictions
        'ambiguous_queries': [], # Low confidence across the board
        'category_swaps': {}     # Track category confusion
    }
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console
    ) as progress:
        task = progress.add_task("[magenta]Testing with ambiguity detection...", total=len(test_cases))
        
        for i, (query, expected_cat, expected_topic) in enumerate(test_cases):
            try:
                eval_result = classifier.evaluate_with_ambiguity(query, expected_cat, expected_topic)
                result = classifier.classify(query)
                
                # Get predicted category and topic
                if isinstance(result, SingleAnswer):
                    pred_cat = result.match.category
                    pred_topic = result.match.topic
                    predicted_str = f"{pred_cat}:{pred_topic}"
                else:
                    # Use top prediction for confusion analysis
                    if result.matches:
                        pred_cat = result.matches[0].category
                        pred_topic = result.matches[0].topic
                    else:
                        pred_cat = pred_topic = "unknown"
                    predictions = []
                    for match in result.matches[:2]:
                        predictions.append(f"{match.category}:{match.topic}")
                    predicted_str = " | ".join(predictions)
                
                confidence = eval_result['confidence']
                is_correct = eval_result['is_correct']
                
                # Update confusion matrices
                expected_full = f"{expected_cat}:{expected_topic}"
                predicted_full = f"{pred_cat}:{pred_topic}"
                
                if not is_correct:
                    # Category confusion
                    cat_key = (expected_cat, pred_cat)
                    confusion_matrix[cat_key] = confusion_matrix.get(cat_key, 0) + 1
                    
                    # Topic confusion
                    topic_key = (expected_full, predicted_full)
                    topic_confusion[topic_key] = topic_confusion.get(topic_key, 0) + 1
                
                # Track confidence distributions
                if is_correct:
                    if confidence >= 0.7:
                        confidence_buckets['correct_high'].append(confidence)
                    elif confidence >= 0.4:
                        confidence_buckets['correct_medium'].append(confidence)
                    else:
                        confidence_buckets['correct_low'].append(confidence)
                else:
                    if confidence >= 0.7:
                        confidence_buckets['incorrect_high'].append({
                            'query': query, 'expected': expected_full, 
                            'predicted': predicted_full, 'confidence': confidence
                        })
                    elif confidence >= 0.4:
                        confidence_buckets['incorrect_medium'].append(confidence)
                    else:
                        confidence_buckets['incorrect_low'].append(confidence)
                
                # Track high confidence errors
                if not is_correct and confidence >= 0.7:
                    error_patterns['high_conf_errors'].append({
                        'query': query[:50], 'expected': expected_full,
                        'predicted': predicted_full, 'confidence': confidence
                    })
                
                # Original tracking code
                if eval_result['answer_type'] == 'single':
                    single_count += 1
                    if eval_result['is_correct']:
                        single_correct += 1
                else:
                    multiple_count += 1
                    if eval_result['is_correct']:
                        multiple_correct += 1
                    
                # Collect errors for first 10 incorrect
                if not eval_result['is_correct'] and len(errors) < 10:
                    errors.append({
                        'query': query[:50],
                        'expected': expected_full,
                        'predicted': predicted_str,
                        'type': eval_result['answer_type'],
                        'confidence': confidence
                    })
                
                # Track category stats
                if expected_cat not in category_stats:
                    category_stats[expected_cat] = {
                        'total': 0, 'single_correct': 0, 
                        'multiple_correct': 0, 'single_count': 0
                    }
                
                category_stats[expected_cat]['total'] += 1
                if eval_result['answer_type'] == 'single':
                    category_stats[expected_cat]['single_count'] += 1
                    if eval_result['is_correct']:
                        category_stats[expected_cat]['single_correct'] += 1
                elif eval_result['is_correct']:
                    category_stats[expected_cat]['multiple_correct'] += 1
                
            except Exception as e:
                pass
            
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    # Display results
    total_correct = single_correct + multiple_correct
    overall_accuracy = (total_correct / len(test_cases)) * 100
    
    console.print(f"\n[bold green]═══ Final Results ═══[/bold green]")
    console.print(f"[bold]Overall Success Rate: {total_correct}/{len(test_cases)} ({overall_accuracy:.1f}%)[/bold]")
    console.print(f"[bold]Time Taken: {elapsed_time:.2f} seconds[/bold]")
    console.print(f"[bold]Speed: {len(test_cases)/elapsed_time:.0f} queries/second[/bold]\n")
    
    console.print(f"[bold]Single Answer Cases: {single_count} ({single_correct} correct = "
                 f"{100*single_correct/single_count if single_count else 0:.1f}%)[/bold]")
    console.print(f"[bold]Multiple Answer Cases: {multiple_count} ({multiple_correct} correct = "
                 f"{100*multiple_correct/multiple_count if multiple_count else 0:.1f}%)[/bold]\n")
    
    # Category breakdown
    cat_table = Table(title="Performance by Category", box=box.ROUNDED)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Total", style="yellow")
    cat_table.add_column("Single/Correct", style="green")
    cat_table.add_column("Multiple/Correct", style="magenta")
    cat_table.add_column("Success Rate", style="bold")
    
    for cat, stats in sorted(category_stats.items()):
        total_cat_correct = stats['single_correct'] + stats['multiple_correct']
        success_rate = (total_cat_correct / stats['total']) * 100 if stats['total'] > 0 else 0
        multiple_count_cat = stats['total'] - stats['single_count']
        
        cat_table.add_row(
            cat,
            str(stats['total']),
            f"{stats['single_correct']}/{stats['single_count']}",
            f"{stats['multiple_correct']}/{multiple_count_cat}",
            f"{success_rate:.1f}%"
        )
    
    console.print(cat_table)
    
    # Show sample errors
    if errors:
        console.print("\n[bold red]Sample Errors (First 10):[/bold red]")
        error_table = Table(box=box.SIMPLE)
        error_table.add_column("Query", style="white", width=30)
        error_table.add_column("Expected", style="green", width=25)
        error_table.add_column("Predicted", style="red", width=40)
        error_table.add_column("Type", style="yellow")
        error_table.add_column("Conf", style="magenta")
        
        for err in errors[:10]:
            error_table.add_row(
                err['query'],
                err['expected'],
                err['predicted'],
                err['type'],
                f"{err['confidence']:.3f}"
            )
        
        console.print(error_table)
    
    # ========== CONFUSION ANALYSIS DASHBOARD ==========
    console.print("\n[bold magenta]═══ Confusion Analysis Dashboard ═══[/bold magenta]\n")
    
    # 1. Confidence Distribution Analysis
    console.print("[bold cyan]1. Confidence Distribution Analysis[/bold cyan]")
    
    # Calculate stats
    n_correct_high = len(confidence_buckets['correct_high'])
    n_correct_medium = len(confidence_buckets['correct_medium'])
    n_correct_low = len(confidence_buckets['correct_low'])
    n_incorrect_high = len(confidence_buckets['incorrect_high'])
    n_incorrect_medium = len(confidence_buckets['incorrect_medium'])
    n_incorrect_low = len(confidence_buckets['incorrect_low'])
    
    conf_table = Table(title="Confidence vs Correctness", box=box.ROUNDED)
    conf_table.add_column("Confidence Level", style="cyan")
    conf_table.add_column("Correct", style="green")
    conf_table.add_column("Incorrect", style="red")
    conf_table.add_column("Insight", style="yellow")
    
    conf_table.add_row(
        "High (≥0.7)",
        str(n_correct_high),
        str(n_incorrect_high),
        "⚠️ Check training data" if n_incorrect_high > 10 else "✓ Good"
    )
    conf_table.add_row(
        "Medium (0.4-0.7)",
        str(n_correct_medium),
        str(n_incorrect_medium),
        "Normal uncertainty"
    )
    conf_table.add_row(
        "Low (<0.4)",
        str(n_correct_low),
        str(n_incorrect_low),
        "Expected ambiguity"
    )
    
    console.print(conf_table)
    
    # Show high confidence errors if concerning
    if n_incorrect_high > 5:
        console.print(f"\n[bold red]⚠️  Found {n_incorrect_high} high-confidence errors![/bold red]")
        console.print("[dim]These suggest training data issues or category overlap:[/dim]")
        
        high_conf_table = Table(box=box.SIMPLE)
        high_conf_table.add_column("Query", style="white", width=30)
        high_conf_table.add_column("Expected", style="green")
        high_conf_table.add_column("Predicted", style="red")
        high_conf_table.add_column("Conf", style="yellow")
        
        for err in confidence_buckets['incorrect_high'][:5]:
            high_conf_table.add_row(
                err['query'][:30],
                err['expected'],
                err['predicted'],
                f"{err['confidence']:.3f}"
            )
        console.print(high_conf_table)
    
    # 2. Top Confusion Pairs
    console.print("\n[bold cyan]2. Most Common Confusions[/bold cyan]")
    
    # Category-level confusion
    if confusion_matrix:
        sorted_confusions = sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True)
        
        cat_conf_table = Table(title="Category Confusion (Top 5)", box=box.ROUNDED)
        cat_conf_table.add_column("Expected → Predicted", style="white")
        cat_conf_table.add_column("Count", style="red")
        cat_conf_table.add_column("Recommendation", style="yellow")
        
        for (expected, predicted), count in sorted_confusions[:5]:
            if expected == predicted:
                rec = "Check topic disambiguation"
            elif count > 20:
                rec = f"Major overlap - review both categories"
            elif count > 10:
                rec = f"Add examples distinguishing these"
            else:
                rec = "Minor confusion"
            
            cat_conf_table.add_row(
                f"{expected} → {predicted}",
                str(count),
                rec
            )
        console.print(cat_conf_table)
    
    # Topic-level confusion
    if topic_confusion:
        sorted_topic_conf = sorted(topic_confusion.items(), key=lambda x: x[1], reverse=True)
        
        topic_conf_table = Table(title="Topic Confusion (Top 10)", box=box.ROUNDED)
        topic_conf_table.add_column("Expected", style="green", width=25)
        topic_conf_table.add_column("Predicted", style="red", width=25)
        topic_conf_table.add_column("Count", style="yellow")
        
        for (expected, predicted), count in sorted_topic_conf[:10]:
            topic_conf_table.add_row(expected, predicted, str(count))
        console.print(topic_conf_table)
    
    # 3. Confusion Patterns Analysis
    console.print("\n[bold cyan]3. Confusion Pattern Insights[/bold cyan]")
    
    # Find bidirectional confusions
    bidirectional = {}
    for (exp, pred), count in confusion_matrix.items():
        reverse_key = (pred, exp)
        if reverse_key in confusion_matrix:
            pair = tuple(sorted([exp, pred]))
            if pair not in bidirectional:
                bidirectional[pair] = {
                    'forward': 0,
                    'reverse': 0,
                    'total': 0
                }
            if exp == pair[0]:
                bidirectional[pair]['forward'] = count
            else:
                bidirectional[pair]['reverse'] = count
            bidirectional[pair]['total'] = (bidirectional[pair]['forward'] + 
                                           bidirectional[pair]['reverse'])
    
    if bidirectional:
        console.print("\n[yellow]Bidirectional Confusions (categories confused both ways):[/yellow]")
        for (cat1, cat2), stats in sorted(bidirectional.items(), 
                                         key=lambda x: x[1]['total'], 
                                         reverse=True)[:3]:
            console.print(f"  • {cat1} ↔ {cat2}: {stats['total']} total confusions")
            console.print(f"    ({cat1}→{cat2}: {stats['forward']}, "
                         f"{cat2}→{cat1}: {stats['reverse']})")
            console.print(f"    [dim]→ Consider merging or clarifying distinction[/dim]")
    
    # 4. Actionable Recommendations
    console.print("\n[bold cyan]4. Actionable Recommendations[/bold cyan]")
    
    recommendations = []
    
    # Check for high confidence errors
    if n_incorrect_high > 10:
        recommendations.append({
            'priority': 'HIGH',
            'issue': f'Found {n_incorrect_high} high-confidence errors',
            'action': 'Review training data for mislabeled examples or overlapping categories'
        })
    
    # Check for systematic category confusion
    for (exp, pred), count in sorted_confusions[:3]:
        if count > 20:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'{exp} → {pred} confusion ({count} cases)',
                'action': f'Add more distinguishing examples between {exp} and {pred}'
            })
    
    # Check for low overall confidence
    total_low_conf = n_correct_low + n_incorrect_low
    if total_low_conf > len(test_cases) * 0.3:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': f'{total_low_conf} queries had low confidence',
            'action': 'Consider adding more training examples or adjusting threshold'
        })
    
    # Display recommendations
    if recommendations:
        rec_table = Table(title="Improvement Recommendations", box=box.ROUNDED)
        rec_table.add_column("Priority", style="bold")
        rec_table.add_column("Issue", style="yellow")
        rec_table.add_column("Recommended Action", style="green")
        
        for rec in sorted(recommendations, key=lambda x: x['priority']):
            color = "red" if rec['priority'] == "HIGH" else "yellow"
            rec_table.add_row(
                f"[{color}]{rec['priority']}[/{color}]",
                rec['issue'],
                rec['action']
            )
        console.print(rec_table)
    else:
        console.print("[green]✓ No major issues detected![/green]")
    
    # 5. Summary Statistics
    console.print("\n[bold cyan]5. Summary Statistics[/bold cyan]")
    
    total_confusions = sum(confusion_matrix.values())
    unique_confusion_pairs = len(confusion_matrix)
    
    summary_stats = {
        'Total Misclassifications': len(test_cases) - total_correct,
        'Unique Confusion Pairs': unique_confusion_pairs,
        'High Confidence Errors': n_incorrect_high,
        'Avg Confidence (Correct)': np.mean([c for bucket in ['correct_high', 'correct_medium', 'correct_low'] 
                                            for c in (confidence_buckets[bucket] if isinstance(confidence_buckets[bucket], list) and all(isinstance(x, (int, float)) for x in confidence_buckets[bucket]) else [])]) if any(confidence_buckets[b] for b in ['correct_high', 'correct_medium', 'correct_low']) else 0,
        'Most Confused Category': sorted_confusions[0][0][0] if sorted_confusions else "None"
    }
    
    for key, value in summary_stats.items():
        if isinstance(value, float):
            console.print(f"  • {key}: {value:.3f}")
        else:
            console.print(f"  • {key}: {value}")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries"""
    console.print("[bold magenta]Running Sample Test (100 queries)[/bold magenta]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    single_correct = 0
    multiple_correct = 0
    single_count = 0
    multiple_count = 0
    category_stats = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[magenta]Testing with ambiguity detection...", total=len(test_cases))
        
        for query, expected_cat, expected_topic in test_cases:
            try:
                eval_result = classifier.evaluate_with_ambiguity(query, expected_cat, expected_topic)
                
                if eval_result['answer_type'] == 'single':
                    single_count += 1
                    if eval_result['is_correct']:
                        single_correct += 1
                else:
                    multiple_count += 1
                    if eval_result['is_correct']:
                        multiple_correct += 1
                
                # Track category stats
                if expected_cat not in category_stats:
                    category_stats[expected_cat] = {
                        'total': 0, 'single_correct': 0, 
                        'multiple_correct': 0, 'single_count': 0
                    }
                
                category_stats[expected_cat]['total'] += 1
                if eval_result['answer_type'] == 'single':
                    category_stats[expected_cat]['single_count'] += 1
                    if eval_result['is_correct']:
                        category_stats[expected_cat]['single_correct'] += 1
                elif eval_result['is_correct']:
                    category_stats[expected_cat]['multiple_correct'] += 1
                
            except Exception as e:
                pass
            
            progress.update(task, advance=1)
    
    # Display results
    total_correct = single_correct + multiple_correct
    overall_accuracy = (total_correct / len(test_cases)) * 100
    
    console.print(f"\n[bold green]Overall Success Rate: {total_correct}/100 ({overall_accuracy:.1f}%)[/bold green]")
    console.print(f"[bold]Single Answer Cases: {single_count} ({single_correct} correct = "
                 f"{100*single_correct/single_count if single_count else 0:.1f}%)[/bold]")
    console.print(f"[bold]Multiple Answer Cases: {multiple_count} ({multiple_correct} correct = "
                 f"{100*multiple_correct/multiple_count if multiple_count else 0:.1f}%)[/bold]\n")
    
    # Category breakdown
    cat_table = Table(title="Performance by Category", box=box.ROUNDED)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Total", style="yellow")
    cat_table.add_column("Single/Correct", style="green")
    cat_table.add_column("Multiple/Correct", style="magenta")
    cat_table.add_column("Success Rate", style="bold")
    
    for cat, stats in sorted(category_stats.items()):
        total_cat_correct = stats['single_correct'] + stats['multiple_correct']
        success_rate = (total_cat_correct / stats['total']) * 100 if stats['total'] > 0 else 0
        multiple_count_cat = stats['total'] - stats['single_count']
        
        cat_table.add_row(
            cat,
            str(stats['total']),
            f"{stats['single_correct']}/{stats['single_count']}",
            f"{stats['multiple_correct']}/{multiple_count_cat}",
            f"{success_rate:.1f}%"
        )
    
    console.print(cat_table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def analyze_ambiguity(classifier):
    """Analyze ambiguity distribution in test set"""
    console.print("[bold magenta]Analyzing Ambiguity Distribution[/bold magenta]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    confidence_buckets = {
        '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, 
        '0.6-0.8': 0, '0.8-1.0': 0
    }
    ambiguous_examples = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[magenta]Analyzing queries...", total=len(test_cases))
        
        for query, expected_cat, expected_topic in test_cases:
            result = classifier.classify(query)
            
            if isinstance(result, SingleAnswer):
                conf = result.match.confidence
            else:
                conf = result.matches[0].confidence if result.matches else 0
                if conf < classifier.confidence_threshold:
                    ambiguous_examples.append((query, conf, expected_cat, expected_topic))
            
            # Bucket the confidence
            if conf < 0.2:
                confidence_buckets['0.0-0.2'] += 1
            elif conf < 0.4:
                confidence_buckets['0.2-0.4'] += 1
            elif conf < 0.6:
                confidence_buckets['0.4-0.6'] += 1
            elif conf < 0.8:
                confidence_buckets['0.6-0.8'] += 1
            else:
                confidence_buckets['0.8-1.0'] += 1
            
            progress.update(task, advance=1)
    
    # Display confidence distribution
    console.print("\n[bold]Confidence Distribution:[/bold]")
    for bucket, count in confidence_buckets.items():
        bar = "█" * int(count / 2)
        console.print(f"{bucket}: {bar} {count}")
    
    # Show some ambiguous examples
    if ambiguous_examples:
        console.print(f"\n[bold]Sample Ambiguous Queries (confidence < {classifier.confidence_threshold:.2f}):[/bold]")
        ambiguous_examples.sort(key=lambda x: x[1])  # Sort by confidence
        
        table = Table(box=box.SIMPLE)
        table.add_column("Query", style="white", width=40)
        table.add_column("Confidence", style="yellow")
        table.add_column("Expected", style="cyan")
        
        for query, conf, cat, topic in ambiguous_examples[:10]:
            table.add_row(
                query[:40],
                f"{conf:.3f}",
                f"{cat}:{topic}"
            )
        
        console.print(table)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def test_ambiguous_cases(classifier):
    """Test specific ambiguous cases"""
    console.print("[bold magenta]Testing Ambiguous Cases[/bold magenta]\n")
    
    # These are inherently ambiguous queries
    ambiguous_queries = [
        ("problem with my order", ["billing", "shipping", "technical_support"]),
        ("not working", ["technical_support", "billing"]),
        ("need help", ["technical_support", "account_management", "billing", "shipping"]),
        ("something is wrong", ["technical_support", "billing", "shipping"]),
        ("can't access", ["technical_support", "account_management"]),
        ("issue with payment", ["billing", "technical_support"]),
        ("update needed", ["account_management", "technical_support"]),
    ]
    
    for query, possible_categories in ambiguous_queries:
        console.print(f"\n[bold]Query:[/bold] '{query}'")
        console.print(f"[dim]Possible categories: {', '.join(possible_categories)}[/dim]\n")
        
        result = classifier.classify(query)
        
        if isinstance(result, SingleAnswer):
            console.print(f"[green]Single Answer (confidence {result.match.confidence:.2f}):[/green]")
            console.print(f"  → {result.match.category}:{result.match.topic}")
        else:
            console.print(f"[yellow]Multiple Answers ({result.reason}):[/yellow]")
            for i, match in enumerate(result.matches, 1):
                console.print(f"  {i}. {match.category}:{match.topic} ({match.confidence:.2f})")
        
        console.print()
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_about():
    """Display information about production mode"""
    about_text = """
[bold magenta]Iteration 2.C3: Production-Ready with Ambiguity Detection[/bold magenta]

[yellow]Key Features:[/yellow]
• Base accuracy: 78% with sentence transformers (all-MiniLM-L6-v2)
• Returns Pydantic models for API integration
• Distinguishes between confident and ambiguous queries
• No chatbot formatting - pure data models

[yellow]Response Types:[/yellow]
1. [green]SingleAnswer[/green] - High confidence (≥ threshold)
   - Returns single best match with confidence score
   - Used when model is confident in its prediction

2. [yellow]MultipleAnswers[/yellow] - Low confidence (< threshold)
   - Returns top N alternatives
   - Includes reason for ambiguity
   - Perfect for "Did you mean..." UX patterns

[yellow]Production Benefits:[/yellow]
• API-ready with Pydantic validation
• Type-safe responses with Union types
• Configurable confidence thresholds
• Handles genuine ambiguity gracefully
• Reduces false positives in production

[yellow]Use Cases:[/yellow]
• Chatbots that ask for clarification
• Support systems with escalation paths
• APIs that return confidence scores
• Systems requiring high precision

[yellow]Success Metrics:[/yellow]
• High confidence: Exact match required
• Low confidence: Success if correct answer in alternatives
• Overall: Better UX through transparency about uncertainty
    """
    
    console.print(Panel(about_text, title="[green]About Production Mode[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Initialize classifier
    console.print("[magenta]Initializing Production Classifier...[/magenta]")
    
    # Use sentence transformers if available, otherwise fallback
    if EMBEDDINGS_AVAILABLE:
        model_name = "all-MiniLM-L6-v2"
        console.print(f"[green]Using sentence transformers: {model_name}[/green]")
    else:
        model_name = "simple"
        console.print("[yellow]Using TF-IDF + char n-grams fallback[/yellow]")
    
    classifier = ProductionEmbeddingClassifier(
        model_name=model_name,
        confidence_threshold=0.6,
        alternatives_count=3
    )
    
    try:
        classifier.load_data("data/training_data.yaml")
        console.print(f"[green]✓ Loaded {len(classifier.training_examples)} training examples![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    # Main menu loop
    while True:
        display_welcome(classifier)
        display_menu()
        
        choice = Prompt.ask("[bold]Select option[/bold]", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
        
        console.clear()
        
        if choice == "0":
            console.print("[yellow]Thanks for using Query Matcher v2.C3! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            interactive_query(classifier)
        elif choice == "2":
            configure_threshold(classifier)
        elif choice == "3":
            run_quick_test(classifier)
        elif choice == "4":
            run_sample_test(classifier)
        elif choice == "5":
            run_full_test(classifier)
        elif choice == "6":
            analyze_ambiguity(classifier)
        elif choice == "7":
            test_ambiguous_cases(classifier)
        elif choice == "8":
            show_about()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        sys.exit(0)