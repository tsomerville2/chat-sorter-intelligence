#!/usr/bin/env python3
"""
Query Matcher - Beautiful CLI Menu
Match customer queries to categories and topics using ML
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich import box
from query_classifier import QueryClassifier
import sys
import os


console = Console()


def display_welcome():
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🤖 ", style="bold cyan")
    welcome_text.append("QUERY MATCHER", style="bold yellow")
    welcome_text.append(" 🤖", style="bold cyan")
    
    panel = Panel(
        "[cyan]ML-Based Customer Query Classification System[/cyan]\n"
        "[dim]Match customer queries to categories and topics[/dim]",
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
    table.add_row("3", "🔄 Retrain Model")
    table.add_row("4", "📝 Batch Process Queries")
    table.add_row("5", "ℹ️  About")
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


def show_categories(classifier):
    """Display training categories and topics"""
    console.print("[bold cyan]Training Categories & Topics[/bold cyan]\n")
    
    categories = classifier.data_manager.get_categories()
    
    for category_name, category_data in categories.items():
        # Category header
        console.print(f"[bold yellow]📁 {category_name.upper()}[/bold yellow]")
        
        topics = category_data.get('topics', {})
        for topic_name in topics:
            console.print(f"  [cyan]└─[/cyan] {topic_name}")
        
        console.print()
    
    Prompt.ask("[dim]Press Enter to continue[/dim]")


def batch_process(classifier):
    """Process multiple queries"""
    console.print("[bold cyan]Batch Process Mode[/bold cyan]")
    console.print("[dim]Enter queries one per line. Type 'done' when finished.[/dim]\n")
    
    queries = []
    while True:
        query = Prompt.ask(f"[yellow]Query {len(queries) + 1}[/yellow]")
        if query.lower() == 'done':
            break
        queries.append(query)
    
    if queries:
        console.print(f"\n[cyan]Processing {len(queries)} queries...[/cyan]\n")
        
        results_table = Table(title="Batch Results", box=box.ROUNDED)
        results_table.add_column("Query", style="white", width=40)
        results_table.add_column("Category", style="cyan")
        results_table.add_column("Topic", style="green")
        results_table.add_column("Confidence", style="yellow")
        
        for query in queries:
            try:
                result = classifier.predict(query)
                results_table.add_row(
                    query[:40] + "..." if len(query) > 40 else query,
                    result.category,
                    result.topic,
                    f"{result.confidence:.2%}"
                )
            except Exception as e:
                results_table.add_row(query[:40], "[red]Error[/red]", str(e), "-")
        
        console.print(results_table)
        console.print()
        Prompt.ask("[dim]Press Enter to continue[/dim]")


def show_about():
    """Display about information"""
    about_text = """
[bold cyan]Query Matcher v1.0[/bold cyan]

[yellow]Features:[/yellow]
• ML-based text classification using TF-IDF and Logistic Regression
• Trained on hierarchical category/topic structure
• No hardcoded logic - completely data-driven
• Generalizable to any domain by changing training data

[yellow]How it works:[/yellow]
1. Loads training data from YAML file
2. Trains classifier on example queries
3. Matches new queries to most likely category+topic
4. Returns confidence score for the match

[yellow]Tech Stack:[/yellow]
• Python with scikit-learn
• TF-IDF vectorization
• Logistic Regression classifier
• BDD with Behave framework
    """
    
    console.print(Panel(about_text, title="[green]About Query Matcher[/green]", 
                       border_style="green"))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


def main():
    """Main application loop"""
    # Initialize classifier
    console.print("[cyan]Initializing Query Matcher...[/cyan]")
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
        
        choice = Prompt.ask("[bold]Select option[/bold]", choices=["0", "1", "2", "3", "4", "5"])
        
        console.clear()
        
        if choice == "0":
            console.print("[yellow]Thanks for using Query Matcher! Goodbye! 👋[/yellow]")
            break
        elif choice == "1":
            quick_match(classifier)
        elif choice == "2":
            show_categories(classifier)
        elif choice == "3":
            console.print("[cyan]Retraining model...[/cyan]")
            classifier.train()
            console.print("[green]✓ Model retrained![/green]\n")
            Prompt.ask("[dim]Press Enter to continue[/dim]")
        elif choice == "4":
            batch_process(classifier)
        elif choice == "5":
            show_about()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        sys.exit(0)