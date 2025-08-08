#!/usr/bin/env python3
"""
Query Matcher - Iteration 1.A (Enhanced ML Features)
Experimental: Better feature engineering for ML classifier
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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import yaml
from dataclasses import dataclass
from typing import List, Dict, Tuple
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats
import re


console = Console()


@dataclass
class PredictionResult:
    category: str
    topic: str
    confidence: float
    features_used: str


class EnhancedMLClassifier:
    def __init__(self):
        # Multiple feature extractors
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),  # Unigrams to trigrams
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True
        )
        
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer='char_wb',  # Character n-grams with word boundaries
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.95
        )
        
        # POS tag patterns (simplified)
        self.pos_vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r'\b[A-Z]+\b',  # Capture patterns like VERB_NOUN
            ngram_range=(1, 2)
        )
        
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,  # Regularization
            class_weight='balanced',  # Handle imbalanced classes
            solver='lbfgs',
            multi_class='multinomial',
            n_jobs=-1
        )
        
        self.scaler = StandardScaler(with_mean=False)  # For sparse matrices
        self.categories = {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        
    def extract_custom_features(self, texts: List[str]) -> np.ndarray:
        """Extract custom features from text"""
        features = []
        
        for text in texts:
            text_lower = text.lower()
            feat = []
            
            # Length features
            feat.append(len(text))  # Total length
            feat.append(len(text.split()))  # Word count
            feat.append(len(text) / (len(text.split()) + 1))  # Avg word length
            
            # Punctuation features
            feat.append(text.count('?'))  # Questions
            feat.append(text.count('!'))  # Exclamations
            feat.append(text.count("'"))  # Contractions
            
            # Keyword indicators
            feat.append(1 if 'password' in text_lower else 0)
            feat.append(1 if 'login' in text_lower or 'log in' in text_lower else 0)
            feat.append(1 if 'payment' in text_lower or 'pay' in text_lower else 0)
            feat.append(1 if 'card' in text_lower else 0)
            feat.append(1 if 'ship' in text_lower or 'deliver' in text_lower else 0)
            feat.append(1 if 'order' in text_lower or 'package' in text_lower else 0)
            feat.append(1 if 'refund' in text_lower or 'money back' in text_lower else 0)
            feat.append(1 if 'account' in text_lower else 0)
            feat.append(1 if 'install' in text_lower else 0)
            feat.append(1 if 'cancel' in text_lower else 0)
            
            # Negation detection
            negations = ['not', "n't", 'no', 'never', 'cannot', "can't", "won't", "couldn't"]
            feat.append(1 if any(neg in text_lower for neg in negations) else 0)
            
            # Urgency indicators
            urgent = ['help', 'urgent', 'asap', 'immediately', 'now', 'please']
            feat.append(1 if any(u in text_lower for u in urgent) else 0)
            
            features.append(feat)
        
        return np.array(features)
    
    def create_pos_tags(self, texts: List[str]) -> List[str]:
        """Create simplified POS tag patterns"""
        pos_texts = []
        
        for text in texts:
            # Simple heuristic POS tagging
            words = text.split()
            tags = []
            
            for word in words:
                if word.lower() in ['i', 'my', 'me', 'you', 'your']:
                    tags.append('PRON')
                elif word.lower() in ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'can', "can't", "won't", "didn't"]:
                    tags.append('VERB')
                elif word.lower() in ['the', 'a', 'an']:
                    tags.append('DET')
                elif word.lower() in ['not', 'no', 'never']:
                    tags.append('NEG')
                elif word[0].isupper() and len(word) > 1:
                    tags.append('NOUN')
                elif word.endswith('ing') or word.endswith('ed'):
                    tags.append('VERB')
                elif word.endswith('ly'):
                    tags.append('ADV')
                else:
                    tags.append('WORD')
            
            # Create bigram patterns
            pos_text = ' '.join(tags)
            bigrams = [f"{tags[i]}_{tags[i+1]}" for i in range(len(tags)-1)]
            pos_text += ' ' + ' '.join(bigrams)
            
            pos_texts.append(pos_text)
        
        return pos_texts
    
    def load_data(self, filepath: str) -> bool:
        """Load training data from YAML"""
        try:
            with open(filepath, 'r') as file:
                yaml_data = yaml.safe_load(file)
                self.categories = yaml_data.get('categories', {})
                
                # Prepare training data
                texts = []
                labels = []
                
                for category_name, category_data in self.categories.items():
                    topics = category_data.get('topics', {})
                    
                    for topic_name, topic_data in topics.items():
                        examples = topic_data.get('examples', [])
                        label = f"{category_name}:{topic_name}"
                        
                        for example in examples:
                            texts.append(example)
                            labels.append(label)
                
                # Create label mappings
                unique_labels = sorted(set(labels))
                self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
                
                # Store for training
                self.training_texts = texts
                self.training_labels = [self.label_to_idx[label] for label in labels]
                
                return True
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def train(self):
        """Train the enhanced classifier"""
        if not hasattr(self, 'training_texts'):
            raise ValueError("No training data loaded")
        
        console.print("[yellow]Extracting enhanced features...[/yellow]")
        
        # Extract different feature types
        word_features = self.word_vectorizer.fit_transform(self.training_texts)
        char_features = self.char_vectorizer.fit_transform(self.training_texts)
        
        # POS tag features
        pos_texts = self.create_pos_tags(self.training_texts)
        pos_features = self.pos_vectorizer.fit_transform(pos_texts)
        
        # Custom features
        custom_features = self.extract_custom_features(self.training_texts)
        
        # Combine all features
        all_features = hstack([
            word_features,
            char_features * 0.5,  # Weight character features less
            pos_features * 0.3,   # Weight POS features even less
            custom_features * 2   # Weight custom features more
        ])
        
        # Scale features
        all_features = self.scaler.fit_transform(all_features)
        
        # Train model
        console.print("[yellow]Training enhanced model...[/yellow]")
        self.model.fit(all_features, self.training_labels)
        
        # Calculate training accuracy
        train_pred = self.model.predict(all_features)
        train_acc = np.mean(train_pred == self.training_labels)
        console.print(f"[green]Training accuracy: {train_acc:.2%}[/green]")
    
    def predict(self, query: str) -> PredictionResult:
        """Predict category and topic for a query"""
        # Extract all feature types
        word_features = self.word_vectorizer.transform([query])
        char_features = self.char_vectorizer.transform([query])
        
        pos_texts = self.create_pos_tags([query])
        pos_features = self.pos_vectorizer.transform(pos_texts)
        
        custom_features = self.extract_custom_features([query])
        
        # Combine features
        all_features = hstack([
            word_features,
            char_features * 0.5,
            pos_features * 0.3,
            custom_features * 2
        ])
        
        all_features = self.scaler.transform(all_features)
        
        # Predict
        prediction = self.model.predict(all_features)[0]
        probabilities = self.model.predict_proba(all_features)[0]
        
        # Get label and confidence
        label = self.idx_to_label[prediction]
        confidence = probabilities[prediction]
        
        category, topic = label.split(':')
        
        return PredictionResult(
            category=category,
            topic=topic,
            confidence=float(confidence),
            features_used="word_ngrams + char_ngrams + pos_patterns + custom_features"
        )


def display_welcome():
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🤖 ", style="bold cyan")
    welcome_text.append("QUERY MATCHER v1.A", style="bold yellow")
    welcome_text.append(" - Enhanced ML", style="bold cyan")
    
    panel = Panel(
        "[cyan]Enhanced Feature Engineering for ML Classification[/cyan]\n"
        "[dim]Combines multiple feature types for better accuracy[/dim]",
        title=welcome_text,
        border_style="bright_blue",
        padding=(1, 2),
        box=box.DOUBLE
    )
    console.print(panel)
    console.print()


def display_menu():
    """Display main menu"""
    table = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
    table.add_column("Option", style="bold yellow", width=3)
    table.add_column("Action", style="white")
    
    table.add_row("1", "🚀 Quick Match (Test a query)")
    table.add_row("2", "📊 Show Training Categories")
    table.add_row("3", "🧪 Run Quick Test (5 queries)")
    table.add_row("4", "🔬 Run Sample Test (100 queries)")
    table.add_row("5", "⚡ Run Full Test Suite (1000+ queries)")
    table.add_row("6", "📈 Show Test Statistics")
    table.add_row("7", "ℹ️  About Enhanced Features")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def quick_match(classifier):
    """Quick match a single query"""
    console.print("[bold cyan]Quick Match Mode - Enhanced ML[/bold cyan]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status("[cyan]Analyzing with enhanced features...[/cyan]"):
            try:
                result = classifier.predict(query)
                
                # Create result panel
                result_table = Table(show_header=False, box=box.SIMPLE)
                result_table.add_column("Field", style="cyan")
                result_table.add_column("Value", style="green")
                
                result_table.add_row("Category", f"[bold]{result.category}[/bold]")
                result_table.add_row("Topic", f"[bold]{result.topic}[/bold]")
                result_table.add_row("Confidence", f"{result.confidence:.2%}")
                result_table.add_row("Features", result.features_used)
                
                console.print(Panel(result_table, title="[green]✓ Match Found[/green]", 
                                  border_style="green"))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def run_quick_test(classifier):
    """Run the original 5-query test"""
    console.print("[bold cyan]Running Quick Test Suite - Enhanced ML[/bold cyan]\n")
    
    test_queries = [
        ("I can't login", "technical_support", "password_reset"),
        ("my payment didn't work", "billing", "payment_failed"),
        ("where's my stuff", "shipping", "track_order"),
        ("I forgot my password", "technical_support", "password_reset"),
        ("card declined", "billing", "payment_failed"),
    ]
    
    results_table = Table(title="Quick Test Results", box=box.ROUNDED)
    results_table.add_column("Query", style="white")
    results_table.add_column("Expected", style="cyan")
    results_table.add_column("Predicted", style="magenta")
    results_table.add_column("Confidence", style="yellow")
    results_table.add_column("✓/✗", style="green")
    
    correct = 0
    for query, expected_cat, expected_topic in test_queries:
        result = classifier.predict(query)
        expected = f"{expected_cat}:{expected_topic}"
        predicted = f"{result.category}:{result.topic}"
        is_correct = expected == predicted
        if is_correct:
            correct += 1
        
        results_table.add_row(
            query,
            expected,
            predicted,
            f"{result.confidence:.1%}",
            "✓" if is_correct else "✗"
        )
    
    console.print(results_table)
    console.print(f"\n[bold]Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.1f}%)[/bold]")
    console.print("[dim]Using enhanced feature engineering[/dim]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries"""
    console.print("[bold cyan]Running Sample Test Suite (100 queries) - Enhanced ML[/bold cyan]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    correct = 0
    category_accuracy = {}
    confidence_sum = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Testing with enhanced features...", total=len(test_cases))
        
        for query, expected_cat, expected_topic in test_cases:
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                
                confidence_sum += result.confidence
                
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
    avg_confidence = confidence_sum / len(test_cases)
    
    console.print(f"\n[bold green]Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)[/bold green]")
    console.print(f"[bold]Average Confidence: {avg_confidence:.2%}[/bold]\n")
    
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


def show_about():
    """Display about information"""
    about_text = """
[bold cyan]Iteration 1.A: Enhanced ML Features[/bold cyan]

[yellow]Feature Types:[/yellow]

1. [cyan]Word N-grams (1-3)[/cyan]
   • Captures words and phrases
   • TF-IDF weighted with sublinear scaling

2. [cyan]Character N-grams (3-5)[/cyan]
   • Handles typos and word variations
   • Word boundary aware

3. [cyan]POS Tag Patterns[/cyan]
   • Simplified part-of-speech tagging
   • Captures grammatical patterns
   • Examples: PRON_VERB, NEG_VERB

4. [cyan]Custom Features[/cyan]
   • Text length and word count
   • Punctuation counts (?, !, ')
   • Domain-specific keywords
   • Negation detection
   • Urgency indicators

[yellow]Improvements:[/yellow]
• Better handling of "I can't login" via negation detection
• Keyword features for common support topics
• Balanced class weights for imbalanced data
• Feature weighting to emphasize important signals

[yellow]Expected Performance:[/yellow]
• Should achieve 80-85% accuracy
• Better confidence calibration
• More robust to variations
    """
    
    console.print(Panel(about_text, title="[green]About Enhanced ML Features[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_categories(classifier):
    """Display training categories and topics"""
    console.print("[bold cyan]Training Categories & Topics[/bold cyan]\n")
    
    for category_name, category_data in classifier.categories.items():
        console.print(f"[bold yellow]📁 {category_name.upper()}[/bold yellow]")
        
        topics = category_data.get('topics', {})
        for topic_name in topics:
            console.print(f"  [cyan]└─[/cyan] {topic_name}")
        
        console.print()
    
    Prompt.ask("[dim]Press Enter to continue[/dim]")


def show_test_stats():
    """Show statistics about the test dataset"""
    console.print("[bold cyan]Test Dataset Statistics[/bold cyan]\n")
    
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


def main():
    """Main application loop"""
    # Initialize classifier
    console.print("[cyan]Initializing Enhanced ML Classifier...[/cyan]")
    classifier = EnhancedMLClassifier()
    
    try:
        classifier.load_data("data/training_data.yaml")
        classifier.train()
        console.print("[green]✓ Enhanced model trained successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    # Main menu loop
    while True:
        display_welcome()
        display_menu()
        
        choice = Prompt.ask("[bold]Select option[/bold]", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
        
        console.clear()
        
        if choice == "0":
            console.print("[yellow]Thanks for using Query Matcher v1.A! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            quick_match(classifier)
        elif choice == "2":
            show_categories(classifier)
        elif choice == "3":
            run_quick_test(classifier)
        elif choice == "4":
            run_sample_test(classifier)
        elif choice == "5":
            console.print("[yellow]Full test would take time - use sample test[/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
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