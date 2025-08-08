#!/usr/bin/env python3
"""
Query Matcher - Iteration 2 (Cosine Similarity)
Direct similarity matching against all training examples
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
sys.path.append('iterations/iteration2')
sys.path.append('shared')
from cosine_classifier import CosineClassifier
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats


console = Console()


def display_welcome():
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🎯 ", style="bold magenta")
    welcome_text.append("QUERY MATCHER v2", style="bold yellow")
    welcome_text.append(" - Cosine Similarity", style="bold cyan")
    
    panel = Panel(
        "[magenta]Direct Similarity Matching System[/magenta]\n"
        "[dim]Finds the most similar training example using cosine similarity[/dim]",
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
    table.add_row("3", "📊 Show Training Categories")
    table.add_row("4", "🧪 Run Quick Test (5 queries)")
    table.add_row("5", "🔬 Run Sample Test (100 queries)")
    table.add_row("6", "⚡ Run Full Test Suite (1000+ queries)")
    table.add_row("7", "📈 Show Test Statistics")
    table.add_row("8", "ℹ️  About This Approach")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def quick_match(classifier):
    """Quick match showing the most similar example"""
    console.print("[bold magenta]Quick Match Mode - Cosine Similarity[/bold magenta]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status("[magenta]Finding most similar example...[/magenta]"):
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
                result_table.add_row("Matched Example", f'"{result.matched_example}"')
                
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
        
        with console.status("[magenta]Finding top 3 similar examples...[/magenta]"):
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
    console.print("[bold magenta]Running Quick Test Suite (5 queries)...[/bold magenta]\n")
    
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
    results_table.add_column("Similarity", style="yellow")
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
            f"{result.similarity_score:.3f}",
            "✓" if is_correct else "✗"
        )
    
    console.print(results_table)
    console.print(f"\n[bold]Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.1f}%)[/bold]")
    
    # Check for zero similarity issues
    zero_sim_count = 0
    for query, expected_cat, expected_topic in test_queries:
        result = classifier.predict(query)
        if result.similarity_score == 0:
            zero_sim_count += 1
    
    if zero_sim_count > 0:
        console.print(f"[bold red]⚠️  {zero_sim_count} queries had ZERO similarity (defaulted to first example)[/bold red]")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries"""
    console.print("[bold magenta]Running Sample Test Suite (100 queries)...[/bold magenta]\n")
    
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
        task = progress.add_task("[magenta]Testing queries...", total=len(test_cases))
        
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
    console.print(f"[bold yellow]Zero Similarity Cases:[/bold yellow]")
    console.print(f"  • Wrong (defaulted incorrectly): {zero_sim_wrong}")
    console.print(f"  • Right (lucky default): {zero_sim_right}\n")
    
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
    console.print("[bold magenta]Running Full Test Suite (1000+ queries)...[/bold magenta]\n")
    console.print("[yellow]This may take a minute...[/yellow]\n")
    
    test_cases = load_comprehensive_test_data()
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    console.print(f"[dim]Loaded {len(test_cases)} test cases[/dim]\n")
    
    correct = 0
    category_accuracy = {}
    zero_sim_wrong = 0
    zero_sim_right = 0
    perfect_matches = 0
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
        task = progress.add_task("[magenta]Testing queries...", total=len(test_cases))
        
        for i, (query, expected_cat, expected_topic) in enumerate(test_cases):
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                    if result.similarity_score == 0:
                        zero_sim_right += 1
                    elif result.similarity_score == 1.0:
                        perfect_matches += 1
                else:
                    if result.similarity_score == 0:
                        zero_sim_wrong += 1
                    if len(errors) < 10:  # Keep first 10 errors
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
    console.print(f"[bold]Perfect Matches (1.0): {perfect_matches}[/bold]")
    console.print(f"[bold red]Zero Similarity (Wrong): {zero_sim_wrong}[/bold red]")
    console.print(f"[bold yellow]Zero Similarity (Right/Lucky): {zero_sim_right}[/bold yellow]")
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
    
    if zero_sim_wrong > 0:
        console.print(f"\n[bold red]⚠️  CRITICAL: {zero_sim_wrong} queries defaulted to wrong category due to zero similarity![/bold red]")
        console.print("[dim]This happens when no training example matches the query[/dim]")
    
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
    """Display information about this approach"""
    about_text = """
[bold magenta]Iteration 2: Cosine Similarity Approach[/bold magenta]

[yellow]How it works:[/yellow]
• Vectorizes all training examples using TF-IDF
• For each query, calculates cosine similarity with ALL examples
• Returns the most similar example's category and topic
• No training phase - just direct comparison

[yellow]Advantages:[/yellow]
• Transparent - you can see which example matched
• No model training needed
• Works well for finding near-exact matches
• Similarity score gives intuitive confidence

[yellow]Disadvantages:[/yellow]
• Can't generalize beyond training examples
• Sensitive to exact wording
• CRITICAL BUG: Zero similarity defaults to first example
• Performance scales linearly with number of examples

[yellow red]Known Issue:[/yellow red]
When a query has zero similarity to ALL training examples,
it defaults to the first example (billing:payment_failed).
This causes incorrect classifications for novel phrasings.

[yellow]Best for:[/yellow]
• Small datasets with distinct examples
• When transparency is important
• Quick prototyping without model training
    """
    
    console.print(Panel(about_text, title="[green]About Cosine Similarity Approach[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_categories(classifier):
    """Display training categories"""
    console.print("[bold magenta]Training Categories & Topics[/bold magenta]\n")
    
    for category_name, category_data in classifier.categories.items():
        console.print(f"[bold yellow]📁 {category_name.upper()}[/bold yellow]")
        
        topics = category_data.get('topics', {})
        for topic_name in topics:
            console.print(f"  [magenta]└─[/magenta] {topic_name}")
        
        console.print()
    
    Prompt.ask("[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Initialize classifier
    console.print("[magenta]Initializing Cosine Similarity Classifier...[/magenta]")
    classifier = CosineClassifier()
    
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
            console.print("[yellow]Thanks for using Query Matcher v2! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            quick_match(classifier)
        elif choice == "2":
            show_top_matches(classifier)
        elif choice == "3":
            show_categories(classifier)
        elif choice == "4":
            run_quick_test(classifier)
        elif choice == "5":
            run_sample_test(classifier)
        elif choice == "6":
            run_full_test(classifier)
        elif choice == "7":
            show_test_stats()
        elif choice == "8":
            show_about()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        sys.exit(0)