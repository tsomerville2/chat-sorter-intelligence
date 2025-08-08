#!/usr/bin/env python3
"""
Query Matcher - Iteration 1 (ML Classifier) with Comprehensive Testing
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
sys.path.append('iterations/iteration1')
sys.path.append('shared')
from query_classifier import QueryClassifier
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats


console = Console()


def display_welcome():
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🤖 ", style="bold cyan")
    welcome_text.append("QUERY MATCHER v1", style="bold yellow")
    welcome_text.append(" - ML Classifier", style="bold cyan")
    
    panel = Panel(
        "[cyan]ML-Based Classification System (Logistic Regression)[/cyan]\n"
        "[dim]TF-IDF vectorization with scikit-learn[/dim]",
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
    table.add_row("7", "ℹ️  About")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def quick_match(classifier):
    """Quick match a single query"""
    console.print("[bold cyan]Quick Match Mode[/bold cyan]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status("[cyan]Analyzing query...[/cyan]"):
            try:
                result = classifier.predict(query)
                
                # Create result panel
                result_table = Table(show_header=False, box=box.SIMPLE)
                result_table.add_column("Field", style="cyan")
                result_table.add_column("Value", style="green")
                
                result_table.add_row("Category", f"[bold]{result.category}[/bold]")
                result_table.add_row("Topic", f"[bold]{result.topic}[/bold]")
                result_table.add_row("Confidence", f"{result.confidence:.2%}")
                
                console.print(Panel(result_table, title="[green]✓ Match Found[/green]", 
                                  border_style="green"))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def run_quick_test(classifier):
    """Run the original 5-query test"""
    console.print("[bold cyan]Running Quick Test Suite (5 queries)...[/bold cyan]\n")
    
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
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries"""
    console.print("[bold cyan]Running Sample Test Suite (100 queries)...[/bold cyan]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    correct = 0
    category_accuracy = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Testing queries...", total=len(test_cases))
        
        for query, expected_cat, expected_topic in test_cases:
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                
                # Track category accuracy
                if expected_cat not in category_accuracy:
                    category_accuracy[expected_cat] = {'correct': 0, 'total': 0}
                category_accuracy[expected_cat]['total'] += 1
                if is_correct:
                    category_accuracy[expected_cat]['correct'] += 1
                
            except Exception as e:
                pass  # Skip errors
            
            progress.update(task, advance=1)
    
    # Display results
    accuracy = (correct / len(test_cases)) * 100
    
    console.print(f"\n[bold green]Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)[/bold green]\n")
    
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
    console.print("[bold cyan]Running Full Test Suite (1000+ queries)...[/bold cyan]\n")
    console.print("[yellow]This may take a minute...[/yellow]\n")
    
    test_cases = load_comprehensive_test_data()
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    console.print(f"[dim]Loaded {len(test_cases)} test cases[/dim]\n")
    
    correct = 0
    category_accuracy = {}
    topic_accuracy = {}
    confidence_sum = 0
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
        task = progress.add_task("[cyan]Testing queries...", total=len(test_cases))
        
        for i, (query, expected_cat, expected_topic) in enumerate(test_cases):
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                else:
                    if len(errors) < 10:  # Keep first 10 errors
                        errors.append({
                            'query': query[:50],
                            'expected': f"{expected_cat}:{expected_topic}",
                            'predicted': f"{result.category}:{result.topic}",
                            'confidence': result.confidence
                        })
                
                confidence_sum += result.confidence
                
                # Track category accuracy
                if expected_cat not in category_accuracy:
                    category_accuracy[expected_cat] = {'correct': 0, 'total': 0}
                category_accuracy[expected_cat]['total'] += 1
                if result.category == expected_cat:
                    category_accuracy[expected_cat]['correct'] += 1
                
                # Track topic accuracy
                topic_key = f"{expected_cat}:{expected_topic}"
                if topic_key not in topic_accuracy:
                    topic_accuracy[topic_key] = {'correct': 0, 'total': 0}
                topic_accuracy[topic_key]['total'] += 1
                if is_correct:
                    topic_accuracy[topic_key]['correct'] += 1
                
            except Exception as e:
                pass  # Skip errors
            
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    # Display results
    accuracy = (correct / len(test_cases)) * 100
    avg_confidence = confidence_sum / len(test_cases)
    
    console.print(f"\n[bold green]═══ Final Results ═══[/bold green]")
    console.print(f"[bold]Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)[/bold]")
    console.print(f"[bold]Average Confidence: {avg_confidence:.2%}[/bold]")
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
        error_table.add_column("Conf", style="yellow")
        
        for err in errors[:10]:
            error_table.add_row(
                err['query'],
                err['expected'],
                err['predicted'],
                f"{err['confidence']:.1%}"
            )
        
        console.print(error_table)
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


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


def show_categories(classifier):
    """Display training categories and topics"""
    console.print("[bold cyan]Training Categories & Topics[/bold cyan]\n")
    
    categories = classifier.data_manager.get_categories()
    
    for category_name, category_data in categories.items():
        console.print(f"[bold yellow]📁 {category_name.upper()}[/bold yellow]")
        
        topics = category_data.get('topics', {})
        for topic_name in topics:
            console.print(f"  [cyan]└─[/cyan] {topic_name}")
        
        console.print()
    
    Prompt.ask("[dim]Press Enter to continue[/dim]")


def show_about():
    """Display about information"""
    about_text = """
[bold cyan]Iteration 1: ML-Based Classification[/bold cyan]

[yellow]Approach:[/yellow]
• TF-IDF vectorization for text features
• Logistic Regression classifier
• Trained on hierarchical category/topic structure

[yellow]Strengths:[/yellow]
• Fast inference (<1ms per query)
• No API costs
• Reasonable accuracy on training distribution
• Completely data-driven

[yellow]Weaknesses:[/yellow]
• Low confidence scores due to multi-class problem
• Struggles with variations not in training
• Can't understand context or intent

[yellow]Best for:[/yellow]
• High-volume applications
• Cost-sensitive deployments
• When training data covers most variations
    """
    
    console.print(Panel(about_text, title="[green]About ML Classifier[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Initialize classifier
    console.print("[cyan]Initializing ML Classifier...[/cyan]")
    classifier = QueryClassifier()
    
    try:
        classifier.load_data("data/training_data.yaml")
        classifier.train()
        console.print("[green]✓ Model trained successfully![/green]\n")
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
            console.print("[yellow]Thanks for using Query Matcher v1! Goodbye! 👋[/yellow]")
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