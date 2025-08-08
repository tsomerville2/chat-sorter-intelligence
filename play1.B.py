#!/usr/bin/env python3
"""
Query Matcher - Iteration 1.B (Different Classifiers)
Experimental: Testing SVM, Random Forest, and Neural Networks
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import yaml
from dataclasses import dataclass
from typing import List, Dict, Tuple
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats
import warnings
warnings.filterwarnings('ignore')


console = Console()


@dataclass
class PredictionResult:
    category: str
    topic: str
    confidence: float
    classifier_name: str


class MultiClassifierSystem:
    def __init__(self, classifier_type='svm'):
        self.classifier_type = classifier_type
        
        # TF-IDF Vectorizer with good settings
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True
        )
        
        # Initialize classifier based on type
        if classifier_type == 'svm':
            self.model = SVC(
                kernel='rbf',  # RBF kernel for non-linear boundaries
                probability=True,  # Enable probability estimates
                C=1.0,
                gamma='scale',
                class_weight='balanced'
            )
            
        elif classifier_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
            
        elif classifier_type == 'neural_net':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),  # Two hidden layers
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            
        elif classifier_type == 'naive_bayes':
            self.model = MultinomialNB(
                alpha=0.1,  # Smoothing parameter
                fit_prior=True
            )
        
        self.scaler = StandardScaler(with_mean=False) if classifier_type == 'neural_net' else None
        self.categories = {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        
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
        """Train the selected classifier"""
        if not hasattr(self, 'training_texts'):
            raise ValueError("No training data loaded")
        
        console.print(f"[yellow]Training {self.classifier_type} classifier...[/yellow]")
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(self.training_texts)
        
        # Scale if using neural network
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X, self.training_labels)
        
        # Calculate training accuracy
        train_pred = self.model.predict(X)
        train_acc = np.mean(train_pred == self.training_labels)
        console.print(f"[green]Training accuracy: {train_acc:.2%}[/green]")
    
    def predict(self, query: str) -> PredictionResult:
        """Predict category and topic for a query"""
        # Vectorize query
        X = self.vectorizer.transform([query])
        
        # Scale if using neural network
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get probabilities based on classifier type
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            confidence = probabilities[prediction]
        else:
            # For classifiers without predict_proba
            confidence = 1.0
        
        # Get label
        label = self.idx_to_label[prediction]
        category, topic = label.split(':')
        
        return PredictionResult(
            category=category,
            topic=topic,
            confidence=float(confidence),
            classifier_name=self.classifier_type
        )


def display_welcome(classifier_type):
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🤖 ", style="bold cyan")
    welcome_text.append("QUERY MATCHER v1.B", style="bold yellow")
    welcome_text.append(f" - {classifier_type.upper()}", style="bold cyan")
    
    panel = Panel(
        f"[cyan]Testing Different ML Classifiers: {classifier_type}[/cyan]\n"
        "[dim]Comparing SVM, Random Forest, Neural Networks, and Naive Bayes[/dim]",
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
    table.add_row("2", "🔄 Switch Classifier")
    table.add_row("3", "🧪 Run Quick Test (5 queries)")
    table.add_row("4", "🔬 Run Sample Test (100 queries)")
    table.add_row("5", "⚖️ Compare All Classifiers")
    table.add_row("6", "📈 Show Test Statistics")
    table.add_row("7", "ℹ️  About Classifiers")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def quick_match(classifier):
    """Quick match a single query"""
    console.print(f"[bold cyan]Quick Match Mode - {classifier.classifier_type.upper()}[/bold cyan]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status(f"[cyan]Classifying with {classifier.classifier_type}...[/cyan]"):
            try:
                result = classifier.predict(query)
                
                # Create result panel
                result_table = Table(show_header=False, box=box.SIMPLE)
                result_table.add_column("Field", style="cyan")
                result_table.add_column("Value", style="green")
                
                result_table.add_row("Category", f"[bold]{result.category}[/bold]")
                result_table.add_row("Topic", f"[bold]{result.topic}[/bold]")
                result_table.add_row("Confidence", f"{result.confidence:.2%}")
                result_table.add_row("Classifier", result.classifier_name.upper())
                
                console.print(Panel(result_table, title="[green]✓ Match Found[/green]", 
                                  border_style="green"))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def switch_classifier(current_classifier):
    """Switch between different classifiers"""
    console.print("[bold cyan]Switch Classifier[/bold cyan]\n")
    console.print(f"Current classifier: [yellow]{current_classifier.classifier_type}[/yellow]\n")
    
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Option", style="yellow")
    table.add_column("Classifier", style="white")
    table.add_column("Description", style="dim")
    
    table.add_row("1", "svm", "Support Vector Machine with RBF kernel")
    table.add_row("2", "random_forest", "Ensemble of decision trees")
    table.add_row("3", "neural_net", "Multi-layer perceptron")
    table.add_row("4", "naive_bayes", "Probabilistic classifier")
    
    console.print(table)
    
    choice = Prompt.ask("\n[bold]Select classifier[/bold]", choices=["1", "2", "3", "4"])
    
    classifiers = ["svm", "random_forest", "neural_net", "naive_bayes"]
    new_type = classifiers[int(choice) - 1]
    
    console.print(f"\n[yellow]Switching to {new_type}... Retraining...[/yellow]")
    
    # Create new classifier
    new_classifier = MultiClassifierSystem(classifier_type=new_type)
    new_classifier.load_data("data/training_data.yaml")
    new_classifier.train()
    
    console.print(f"[green]✓ Switched to {new_type}![/green]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")
    
    return new_classifier


def run_quick_test(classifier):
    """Run the original 5-query test"""
    console.print(f"[bold cyan]Running Quick Test - {classifier.classifier_type.upper()}[/bold cyan]\n")
    
    test_queries = [
        ("I can't login", "technical_support", "password_reset"),
        ("my payment didn't work", "billing", "payment_failed"),
        ("where's my stuff", "shipping", "track_order"),
        ("I forgot my password", "technical_support", "password_reset"),
        ("card declined", "billing", "payment_failed"),
    ]
    
    results_table = Table(title=f"Test Results - {classifier.classifier_type}", box=box.ROUNDED)
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
    console.print(f"[bold]Classifier: {classifier.classifier_type}[/bold]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def compare_all_classifiers():
    """Compare all classifiers on the quick test set"""
    console.print("[bold cyan]Comparing All Classifiers[/bold cyan]\n")
    
    test_queries = [
        ("I can't login", "technical_support", "password_reset"),
        ("my payment didn't work", "billing", "payment_failed"),
        ("where's my stuff", "shipping", "track_order"),
        ("I forgot my password", "technical_support", "password_reset"),
        ("card declined", "billing", "payment_failed"),
    ]
    
    classifiers = ["svm", "random_forest", "neural_net", "naive_bayes"]
    results = {}
    
    for clf_type in classifiers:
        console.print(f"[yellow]Training {clf_type}...[/yellow]")
        clf = MultiClassifierSystem(classifier_type=clf_type)
        clf.load_data("data/training_data.yaml")
        clf.train()
        
        correct = 0
        avg_conf = 0
        
        for query, expected_cat, expected_topic in test_queries:
            result = clf.predict(query)
            if result.category == expected_cat and result.topic == expected_topic:
                correct += 1
            avg_conf += result.confidence
        
        results[clf_type] = {
            'correct': correct,
            'avg_conf': avg_conf / len(test_queries)
        }
    
    # Display comparison table
    comparison_table = Table(title="Classifier Comparison", box=box.ROUNDED)
    comparison_table.add_column("Classifier", style="cyan")
    comparison_table.add_column("Accuracy", style="green")
    comparison_table.add_column("Avg Confidence", style="yellow")
    
    for clf_type, res in results.items():
        comparison_table.add_row(
            clf_type.upper(),
            f"{res['correct']}/5 ({res['correct']*20}%)",
            f"{res['avg_conf']:.2%}"
        )
    
    console.print(comparison_table)
    
    # Test specific problem query
    console.print("\n[bold]Testing 'I can't login' across all classifiers:[/bold]")
    problem_table = Table(box=box.SIMPLE)
    problem_table.add_column("Classifier", style="cyan")
    problem_table.add_column("Prediction", style="magenta")
    problem_table.add_column("Confidence", style="yellow")
    
    for clf_type in classifiers:
        clf = MultiClassifierSystem(classifier_type=clf_type)
        clf.load_data("data/training_data.yaml")
        clf.train()
        result = clf.predict("I can't login")
        problem_table.add_row(
            clf_type.upper(),
            f"{result.category}:{result.topic}",
            f"{result.confidence:.2%}"
        )
    
    console.print(problem_table)
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries"""
    console.print(f"[bold cyan]Running Sample Test (100 queries) - {classifier.classifier_type.upper()}[/bold cyan]\n")
    
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
        task = progress.add_task(f"[cyan]Testing with {classifier.classifier_type}...", total=len(test_cases))
        
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
    console.print(f"[bold]Average Confidence: {avg_confidence:.2%}[/bold]")
    console.print(f"[bold]Classifier: {classifier.classifier_type}[/bold]\n")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_about():
    """Display about information"""
    about_text = """
[bold cyan]Iteration 1.B: Different Classifiers[/bold cyan]

[yellow]Classifiers Implemented:[/yellow]

[cyan]1. SVM (Support Vector Machine)[/cyan]
   • RBF kernel for non-linear boundaries
   • Good for high-dimensional data
   • Works well with clear margin of separation

[cyan]2. Random Forest[/cyan]
   • Ensemble of decision trees
   • Handles non-linear patterns well
   • Provides feature importance
   • Less prone to overfitting

[cyan]3. Neural Network (MLP)[/cyan]
   • Multi-layer perceptron with 2 hidden layers
   • Can learn complex patterns
   • Adaptive learning rate
   • Early stopping to prevent overfitting

[cyan]4. Naive Bayes[/cyan]
   • Probabilistic classifier
   • Fast and simple
   • Works well with text data
   • Assumes feature independence

[yellow]Expected Performance:[/yellow]
• SVM: 75-85% accuracy, good confidence
• Random Forest: 80-85% accuracy, robust
• Neural Net: 75-90% accuracy (with enough data)
• Naive Bayes: 70-80% accuracy, fast

[yellow]Trade-offs:[/yellow]
• Speed: Naive Bayes > Random Forest > SVM > Neural Net
• Accuracy: Neural Net ≈ Random Forest > SVM > Naive Bayes
• Interpretability: Naive Bayes > Random Forest > SVM > Neural Net
    """
    
    console.print(Panel(about_text, title="[green]About Different Classifiers[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_test_stats():
    """Show statistics about the test dataset"""
    console.print("[bold cyan]Test Dataset Statistics[/bold cyan]\n")
    
    stats = get_test_stats()
    
    console.print(f"[bold]Total Test Cases:[/bold] {stats['total_cases']}")
    console.print(f"[bold]Unique Categories:[/bold] {stats['unique_categories']}")
    console.print(f"[bold]Unique Topics:[/bold] {stats['unique_topics']}\n")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Initialize with SVM by default
    console.print("[cyan]Initializing Multi-Classifier System (SVM)...[/cyan]")
    classifier = MultiClassifierSystem(classifier_type="svm")
    
    try:
        classifier.load_data("data/training_data.yaml")
        classifier.train()
        console.print("[green]✓ SVM classifier trained successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    # Main menu loop
    while True:
        display_welcome(classifier.classifier_type)
        display_menu()
        
        choice = Prompt.ask("[bold]Select option[/bold]", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
        
        console.clear()
        
        if choice == "0":
            console.print("[yellow]Thanks for using Query Matcher v1.B! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            quick_match(classifier)
        elif choice == "2":
            classifier = switch_classifier(classifier)
        elif choice == "3":
            run_quick_test(classifier)
        elif choice == "4":
            run_sample_test(classifier)
        elif choice == "5":
            compare_all_classifiers()
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