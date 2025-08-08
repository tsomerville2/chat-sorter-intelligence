#!/usr/bin/env python3
"""
Query Matcher - Iteration 3 (LLM-based)
Uses Groq API with OpenAI OSS models for intelligent classification
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
import sys
import os
import time
sys.path.append('iterations/iteration3')
sys.path.append('shared')
from llm_classifier import LLMClassifier
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats


console = Console()


def display_welcome():
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🤖 ", style="bold cyan")
    welcome_text.append("QUERY MATCHER v3", style="bold yellow")
    welcome_text.append(" - LLM Intelligence", style="bold green")
    
    panel = Panel(
        "[green]AI-Powered Classification System[/green]\n"
        "[dim]Uses OpenAI OSS models via Groq for intelligent understanding[/dim]",
        title=welcome_text,
        border_style="bright_green",
        padding=(1, 2),
        box=box.DOUBLE
    )
    console.print(panel)
    console.print()


def display_menu():
    """Display main menu"""
    table = Table(show_header=False, box=box.ROUNDED, border_style="green")
    table.add_column("Option", style="bold yellow", width=3)
    table.add_column("Action", style="white")
    
    table.add_row("1", "🤖 Quick Match (with reasoning)")
    table.add_row("2", "⚖️  Compare 20B vs 120B Models")
    table.add_row("3", "🧪 Run Quick Test (7 queries)")
    table.add_row("4", "🔬 Run Sample Test (100 queries)")
    table.add_row("5", "⚡ Run Full Test Suite (1000+ queries)")
    table.add_row("6", "📈 Show Test Statistics")
    table.add_row("7", "📊 Show Training Categories")
    table.add_row("8", "🔄 Switch Model")
    table.add_row("9", "ℹ️  About This Approach")
    table.add_row("0", "🚪 Exit")
    
    console.print(table)
    console.print()


def check_api_key():
    """Check if GROQ_API_KEY is set"""
    if not os.environ.get("GROQ_API_KEY"):
        console.print("[red]Error: GROQ_API_KEY environment variable not set![/red]")
        console.print("[yellow]Please set it with: export GROQ_API_KEY='your-key-here'[/yellow]")
        console.print("[dim]Get your API key from: https://console.groq.com/keys[/dim]")
        return False
    return True


def quick_match(classifier):
    """Quick match with LLM reasoning"""
    console.print(f"[bold green]Quick Match Mode - {classifier.model_name}[/bold green]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status(f"[green]Analyzing with {classifier.model_name}...[/green]"):
            try:
                result = classifier.predict(query)
                
                # Create detailed result panel
                result_table = Table(show_header=False, box=box.SIMPLE)
                result_table.add_column("Field", style="green")
                result_table.add_column("Value", style="white")
                
                result_table.add_row("Category", f"[bold cyan]{result.category}[/bold cyan]")
                result_table.add_row("Topic", f"[bold cyan]{result.topic}[/bold cyan]")
                result_table.add_row("Confidence", f"{result.confidence:.2%}")
                result_table.add_row("Reasoning", f'"{result.reasoning}"')
                
                if result.alternative:
                    result_table.add_row("Alternative", 
                                       f"{result.alternative[0]}:{result.alternative[1]}")
                
                result_table.add_row("Model", result.model_used)
                
                console.print(Panel(result_table, title="[green]✓ LLM Analysis Complete[/green]", 
                                  border_style="green"))
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def compare_models(classifier):
    """Compare 20B vs 120B model results"""
    console.print("[bold green]Model Comparison Mode[/bold green]")
    console.print("[dim]Type 'back' to return to menu[/dim]\n")
    
    while True:
        query = Prompt.ask("[yellow]Enter customer query[/yellow]")
        
        if query.lower() == 'back':
            break
        
        with console.status("[green]Comparing models...[/green]"):
            try:
                results = classifier.compare_models(query)
                
                # Create comparison table
                comparison_table = Table(title=f"Query: '{query}'", box=box.ROUNDED)
                comparison_table.add_column("Aspect", style="cyan")
                comparison_table.add_column("GPT-OSS-20B", style="yellow")
                comparison_table.add_column("GPT-OSS-120B", style="green")
                
                comparison_table.add_row(
                    "Category",
                    results['20b'].category,
                    results['120b'].category
                )
                comparison_table.add_row(
                    "Topic",
                    results['20b'].topic,
                    results['120b'].topic
                )
                comparison_table.add_row(
                    "Confidence",
                    f"{results['20b'].confidence:.2%}",
                    f"{results['120b'].confidence:.2%}"
                )
                comparison_table.add_row(
                    "Reasoning",
                    results['20b'].reasoning[:50] + "...",
                    results['120b'].reasoning[:50] + "..."
                )
                
                # Check if results match
                match = (results['20b'].category == results['120b'].category and 
                        results['20b'].topic == results['120b'].topic)
                
                comparison_table.add_row(
                    "Agreement",
                    "✓" if match else "✗",
                    "✓" if match else "✗"
                )
                
                console.print(comparison_table)
                
                if not match:
                    console.print("[yellow]⚠️  Models disagree on classification![/yellow]")
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print()


def run_quick_test(classifier):
    """Run the original 7-query test"""
    console.print(f"[bold green]Running Quick Test Suite (7 queries) with {classifier.model_name}...[/bold green]\n")
    
    test_queries = [
        ("I can't login", "technical_support", "password_reset"),
        ("my payment didn't work", "billing", "payment_failed"),
        ("where's my stuff", "shipping", "track_order"),
        ("I forgot my password", "technical_support", "password_reset"),
        ("card declined", "billing", "payment_failed"),
        ("package never arrived", "shipping", "delivery_problem"),
        ("want my money back", "billing", "refund_request"),
    ]
    
    results_table = Table(title="LLM Test Results", box=box.ROUNDED)
    results_table.add_column("Query", style="white")
    results_table.add_column("Expected", style="cyan")
    results_table.add_column("Predicted", style="green")
    results_table.add_column("Confidence", style="yellow")
    results_table.add_column("✓/✗", style="green")
    
    correct = 0
    for query, expected_cat, expected_topic in test_queries:
        with console.status(f"[dim]Testing: {query}[/dim]"):
            result = classifier.predict(query)
            expected = f"{expected_cat}:{expected_topic}"
            predicted = f"{result.category}:{result.topic}"
            is_correct = expected == predicted
            if is_correct:
                correct += 1
            
            results_table.add_row(
                query[:30],
                expected,
                predicted,
                f"{result.confidence:.2%}",
                "✓" if is_correct else "✗"
            )
    
    console.print(results_table)
    console.print(f"\n[bold]Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.1f}%)[/bold]")
    console.print(f"[dim]Model: {classifier.model_name}[/dim]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_sample_test(classifier):
    """Run test on 100 sample queries"""
    console.print(f"[bold green]Running Sample Test Suite (100 queries) with {classifier.model_name}...[/bold green]\n")
    console.print("[yellow]Note: LLM calls may take time and cost API credits[/yellow]\n")
    
    test_cases = load_test_data_sample(100)
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return
    
    correct = 0
    category_accuracy = {}
    total_tokens = 0
    errors = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[green]Testing queries...", total=len(test_cases))
        
        for query, expected_cat, expected_topic in test_cases:
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                else:
                    if len(errors) < 5:  # Keep first 5 errors for LLM (API cost)
                        errors.append({
                            'query': query[:50],
                            'expected': f"{expected_cat}:{expected_topic}",
                            'predicted': f"{result.category}:{result.topic}",
                            'reasoning': result.reasoning[:50] if result.reasoning else "N/A"
                        })
                
                # Track category accuracy
                if expected_cat not in category_accuracy:
                    category_accuracy[expected_cat] = {'correct': 0, 'total': 0}
                category_accuracy[expected_cat]['total'] += 1
                if is_correct:
                    category_accuracy[expected_cat]['correct'] += 1
                
            except Exception as e:
                pass  # Skip API errors
            
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
    
    # Show sample errors
    if errors:
        console.print("\n[bold red]Sample Errors (First 5):[/bold red]")
        error_table = Table(box=box.SIMPLE)
        error_table.add_column("Query", style="white", width=30)
        error_table.add_column("Expected", style="green")
        error_table.add_column("Got", style="red")
        error_table.add_column("Reasoning", style="dim")
        
        for err in errors:
            error_table.add_row(
                err['query'],
                err['expected'],
                err['predicted'],
                err['reasoning']
            )
        
        console.print(error_table)
    
    console.print(f"\n[dim]Model: {classifier.model_name}[/dim]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def run_full_test(classifier):
    """Run test on all 1000+ queries"""
    console.print(f"[bold green]Running Full Test Suite (1000+ queries) with {classifier.model_name}...[/bold green]\n")
    console.print("[bold yellow]⚠️  WARNING: This will make 1000+ API calls and may cost significant credits![/bold yellow]")
    
    if not Confirm.ask("[red]Are you sure you want to continue?[/red]"):
        console.print("[yellow]Test cancelled[/yellow]")
        return
    
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
    api_errors = 0
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console
    ) as progress:
        task = progress.add_task("[green]Testing queries...", total=len(test_cases))
        
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
                api_errors += 1
                if api_errors % 10 == 0:
                    console.print(f"[yellow]API errors: {api_errors}[/yellow]")
            
            progress.update(task, advance=1)
            
            # Add small delay to avoid rate limiting
            if i % 10 == 0:
                time.sleep(0.1)
    
    elapsed_time = time.time() - start_time
    
    # Display results
    successful_tests = len(test_cases) - api_errors
    if successful_tests > 0:
        accuracy = (correct / successful_tests) * 100
        avg_confidence = confidence_sum / successful_tests
    else:
        accuracy = 0
        avg_confidence = 0
    
    console.print(f"\n[bold green]═══ Final Results ═══[/bold green]")
    console.print(f"[bold]Overall Accuracy: {correct}/{successful_tests} ({accuracy:.1f}%)[/bold]")
    console.print(f"[bold]Average Confidence: {avg_confidence:.2%}[/bold]")
    console.print(f"[bold]API Errors: {api_errors}[/bold]")
    console.print(f"[bold]Time Taken: {elapsed_time:.2f} seconds[/bold]")
    console.print(f"[bold]Speed: {len(test_cases)/elapsed_time:.1f} queries/second[/bold]\n")
    
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
    
    console.print(f"\n[dim]Model: {classifier.model_name}[/dim]")
    console.print(f"[dim]Estimated API cost: ~${(len(test_cases) * 0.0001):.2f} (rough estimate)[/dim]")
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_test_stats():
    """Show statistics about the test dataset"""
    console.print("[bold green]Test Dataset Statistics[/bold green]\n")
    
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


def switch_model(classifier):
    """Switch between 20B and 120B models"""
    console.print("[bold green]Switch Model[/bold green]\n")
    console.print(f"Current model: [cyan]{classifier.model_name}[/cyan]\n")
    
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Option", style="yellow")
    table.add_column("Model", style="white")
    table.add_column("Description", style="dim")
    
    table.add_row("1", "openai/gpt-oss-20b", "Faster, lower cost, good accuracy")
    table.add_row("2", "openai/gpt-oss-120b", "Slower, higher cost, best accuracy")
    
    console.print(table)
    
    choice = Prompt.ask("\n[bold]Select model[/bold]", choices=["1", "2"])
    
    if choice == "1":
        classifier.model_name = "openai/gpt-oss-20b"
        console.print("[green]✓ Switched to GPT-OSS-20B[/green]")
    else:
        classifier.model_name = "openai/gpt-oss-120b"
        console.print("[green]✓ Switched to GPT-OSS-120B[/green]")
    
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_about():
    """Display information about LLM approach"""
    about_text = """
[bold green]Iteration 3: LLM-based Classification[/bold green]

[yellow]How it works:[/yellow]
• Uses OpenAI OSS models via Groq API
• Provides categories/topics to LLM as context
• LLM intelligently matches queries using reasoning
• Returns structured JSON with confidence scores

[yellow]Models Available:[/yellow]
• GPT-OSS-20B: 1000+ tokens/sec, $0.10/M input
• GPT-OSS-120B: 500+ tokens/sec, $0.15/M input

[yellow]Advantages:[/yellow]
• Understands context and intent
• Handles typos and variations naturally
• Provides reasoning for decisions
• Can identify ambiguous queries

[yellow]Disadvantages:[/yellow]
• Requires API key and internet connection
• Costs money per query
• Slower than local models
• May hallucinate if not constrained properly

[yellow]Best for:[/yellow]
• Complex queries with nuanced intent
• When explanation/reasoning is needed
• Handling ambiguous or multi-intent queries
• Production systems with budget for API costs
    """
    
    console.print(Panel(about_text, title="[green]About LLM Classification[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def show_categories(classifier):
    """Display training categories"""
    console.print("[bold green]Available Categories & Topics[/bold green]\n")
    
    for category_name, category_data in classifier.categories.items():
        console.print(f"[bold yellow]📁 {category_name.upper()}[/bold yellow]")
        
        topics = category_data.get('topics', {})
        for topic_name in topics:
            console.print(f"  [green]└─[/green] {topic_name}")
        
        console.print()
    
    Prompt.ask("[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Check for API key
    if not check_api_key():
        sys.exit(1)
    
    # Initialize classifier
    console.print("[green]Initializing LLM Classifier...[/green]")
    
    try:
        classifier = LLMClassifier(model_name="openai/gpt-oss-20b")
        classifier.load_data("data/training_data.yaml")
        console.print(f"[green]✓ LLM Classifier ready with {classifier.model_name}![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        sys.exit(1)
    
    # Main menu loop
    while True:
        display_welcome()
        display_menu()
        
        choice = Prompt.ask("[bold]Select option[/bold]", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        
        console.clear()
        
        if choice == "0":
            console.print("[yellow]Thanks for using Query Matcher v3! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            quick_match(classifier)
        elif choice == "2":
            compare_models(classifier)
        elif choice == "3":
            run_quick_test(classifier)
        elif choice == "4":
            run_sample_test(classifier)
        elif choice == "5":
            run_full_test(classifier)
        elif choice == "6":
            show_test_stats()
        elif choice == "7":
            show_categories(classifier)
        elif choice == "8":
            switch_model(classifier)
        elif choice == "9":
            show_about()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        sys.exit(0)