#!/usr/bin/env python3
"""
Query Matcher - Beautiful CLI Menu
Main entry point with Rich UI for customer query classification
"""

import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich import box
from rich.columns import Columns
import time
import json
from main_controller import MainController

console = Console()


class QueryMatcherMenu:
    """Beautiful CLI menu for Query Matcher"""
    
    def __init__(self):
        self.controller = MainController()
        self.is_initialized = False
    
    def show_welcome_screen(self):
        """Show beautiful welcome screen"""
        console.clear()
        
        # Create ASCII art title
        title = Text("┌────────────────────────────────────────────────┐\n", style="cyan bold")
        title.append("│  🤖 QUERY MATCHER - AI Customer Service Router 🎯  │\n", style="cyan bold")
        title.append("│        Match customer queries to categories!        │\n", style="cyan")
        title.append("└────────────────────────────────────────────────┘", style="cyan bold")
        
        console.print(Align.center(title))
        
        # Mission statement
        mission_panel = Panel(
            Text(
                "Build a simple, generalizable ML-based system that matches customer queries\n"
                "to predefined categories and topics without using LLMs or complex generative models.\n\n"
                "🎥 Success Criteria: 80%+ accuracy matching test queries to correct category+topic pairs", 
                style="white",
                justify="center"
            ),
            title="🎯 Mission",
            border_style="green",
            box=box.ROUNDED
        )
        
        console.print(mission_panel)
        console.print()
    
    def show_main_menu(self):
        """Show main menu with options"""
        
        # Status indicator
        status_text = "🔴 Not Initialized" if not self.is_initialized else "🔵 Ready for Queries"
        status_color = "red" if not self.is_initialized else "green"
        
        status_panel = Panel(
            Text(status_text, style=status_color, justify="center"),
            title="System Status",
            border_style=status_color,
            box=box.HEAVY
        )
        console.print(status_panel)
        console.print()
        
        # Main menu options
        menu_table = Table(show_header=False, box=box.ROUNDED, border_style="blue")
        menu_table.add_column("Option", style="cyan bold", width=4)
        menu_table.add_column("Action", style="white", width=35)
        menu_table.add_column("Description", style="dim white")
        
        menu_table.add_row("1.", "🚀 Quick Start", "Initialize system & start classifying queries")
        menu_table.add_row("2.", "📂 Classify Query", "Classify a single customer query")
        menu_table.add_row("3.", "📃 Batch Classify", "Process multiple queries from file")
        menu_table.add_row("4.", "📊 System Status", "View system status and algorithm info")
        menu_table.add_row("5.", "🗋 Categories Info", "View available categories and topics")
        menu_table.add_row("6.", "⚙️  Advanced Options", "Advanced configuration and testing")
        menu_table.add_row("7.", "🎆 Run BDD Tests", "Execute behavior-driven development tests")
        menu_table.add_row("8.", "❓ Help", "Show help and documentation")
        menu_table.add_row("9.", "🚪 Exit", "Exit the application")
        
        menu_panel = Panel(
            menu_table,
            title="📜 Main Menu",
            border_style="blue",
            box=box.DOUBLE
        )
        
        console.print(menu_panel)
        console.print()
    
    def initialize_system(self):
        """Initialize the system with progress display"""
        if self.is_initialized:
            console.print("ℹ️  System already initialized!", style="yellow")
            return True
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Initializing Query Matcher...", total=None)
            
            result = self.controller.initialize()
            
            progress.stop()
            
            if result["success"]:
                self.is_initialized = True
                
                success_panel = Panel(
                    Text(
                        f"✅ System initialized successfully!\n\n"
                        f"🕰️ Initialization time: {result['initialization_time_seconds']}s\n"
                        f"🤖 Algorithm: {result['algorithm_info']['algorithm']}\n"
                        f"📁 Data source: {result['data_path']}",
                        style="green"
                    ),
                    title="✅ Initialization Success",
                    border_style="green",
                    box=box.ROUNDED
                )
                console.print(success_panel)
                return True
            else:
                error_panel = Panel(
                    Text(
                        f"❌ Initialization failed!\n\n"
                        f"Error: {result['error']}\n"
                        f"Type: {result.get('error_type', 'unknown')}",
                        style="red"
                    ),
                    title="❌ Initialization Error",
                    border_style="red",
                    box=box.ROUNDED
                )
                console.print(error_panel)
                return False
    
    def classify_single_query(self):
        """Classify a single query with beautiful output"""
        if not self.is_initialized:
            console.print("⚠️  Please initialize the system first (option 1)!", style="yellow")
            return
            
        console.print("\n💬 [bold cyan]Enter your customer query:[/bold cyan]")
        query = Prompt.ask("> ", default="")
        
        if not query.strip():
            console.print("⚠️  Empty query provided.", style="yellow")
            return
            
        # Process query with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing query...", total=None)
            result = self.controller.process_query(query)
            progress.stop()
            
        if result["success"]:
            r = result["result"]
            
            # Create result display
            result_table = Table(show_header=False, box=box.ROUNDED, border_style="green")
            result_table.add_column("Field", style="cyan bold")
            result_table.add_column("Value", style="white")
            
            result_table.add_row("📂 Category", f"[bold green]{r['category']}[/bold green]")
            result_table.add_row("🏷️ Topic", f"[bold blue]{r['topic']}[/bold blue]")
            result_table.add_row("🎯 Confidence", f"[bold yellow]{r['confidence']:.1%}[/bold yellow]")
            result_table.add_row("⏱️ Processing Time", f"{result['processing_time_seconds']}s")
            
            if r['matched_example']:
                result_table.add_row("📝 Matched Example", f"[italic]{r['matched_example']}[/italic]")
            
            if r['similarity_score']:
                result_table.add_row("🔍 Similarity Score", f"{r['similarity_score']:.1%}")
            
            result_panel = Panel(
                result_table,
                title=f"🎯 Classification Result: {r['category']}:{r['topic']}",
                border_style="green",
                box=box.DOUBLE
            )
            
            console.print(result_panel)
            
            # Confidence indicator
            if r['confidence'] >= 0.8:
                console.print("✅ [bold green]High confidence result![/bold green]")
            elif r['confidence'] >= 0.5:
                console.print("⚠️ [yellow]Medium confidence - might need review.[/yellow]")
            else:
                console.print("❌ [red]Low confidence - manual review recommended.[/red]")
        else:
            error_panel = Panel(
                Text(f"❌ Error: {result['error']}", style="red"),
                title="❌ Classification Error",
                border_style="red"
            )
            console.print(error_panel)
    
    def show_system_status(self):
        """Show detailed system status"""
        status = self.controller.get_status()
        
        status_table = Table(show_header=False, box=box.ROUNDED)
        status_table.add_column("Property", style="cyan bold")
        status_table.add_column("Value", style="white")
        
        status_table.add_row("Initialized", "🔵 Yes" if status['initialized'] else "🔴 No")
        status_table.add_row("Data Path", status['data_path'] or "Not set")
        status_table.add_row("Init Time", f"{status['initialization_time_seconds']}s" if status['initialization_time_seconds'] else "N/A")
        
        if status['algorithm_info']:
            for key, value in status['algorithm_info'].items():
                status_table.add_row(f"Algorithm: {key}", str(value))
        
        console.print(Panel(
            status_table,
            title="📊 System Status",
            border_style="blue"
        ))
    
    def show_categories_info(self):
        """Show categories and topics information"""
        if not self.is_initialized:
            console.print("⚠️  Please initialize the system first!", style="yellow")
            return
            
        categories_result = self.controller.get_categories_info()
        
        if not categories_result["success"]:
            console.print(f"❌ Error: {categories_result['error']}", style="red")
            return
            
        # Summary info
        summary_table = Table(show_header=False, box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan bold")
        summary_table.add_column("Count", style="white")
        
        summary_table.add_row("Total Categories", str(categories_result['total_categories']))
        summary_table.add_row("Total Training Examples", str(categories_result['total_examples']))
        
        console.print(Panel(
            summary_table,
            title="🗋 Training Data Summary",
            border_style="green"
        ))
        
        # Categories breakdown
        for cat_name, cat_info in categories_result['categories'].items():
            cat_table = Table(show_header=True, box=box.SIMPLE)
            cat_table.add_column("Topic", style="blue bold")
            cat_table.add_column("Examples", style="white")
            cat_table.add_column("Sample", style="dim white")
            
            for topic_name, topic_info in cat_info['topics'].items():
                sample = topic_info['sample_examples'][0] if topic_info['sample_examples'] else "No examples"
                cat_table.add_row(
                    topic_name,
                    str(topic_info['example_count']),
                    sample[:50] + "..." if len(sample) > 50 else sample
                )
            
            console.print(Panel(
                cat_table,
                title=f"🗋 {cat_name.title()} Category ({cat_info['topic_count']} topics)",
                border_style="blue"
            ))
    
    def run_bdd_tests(self):
        """Run BDD tests"""
        console.print("🧪 Running BDD Tests...", style="cyan bold")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running behave tests...", total=None)
            
            # Run behave tests
            import subprocess
            try:
                result = subprocess.run(
                    ["behave", "--format", "progress3", "--no-capture"],
                    capture_output=True,
                    text=True
                )
                
                progress.stop()
                
                if result.returncode == 0:
                    console.print("✅ [bold green]All BDD tests passed![/bold green]")
                    console.print(f"Output:\n{result.stdout}")
                else:
                    console.print("❌ [bold red]Some BDD tests failed![/bold red]")
                    console.print(f"Errors:\n{result.stderr}")
                    console.print(f"Output:\n{result.stdout}")
                    
            except Exception as e:
                progress.stop()
                console.print(f"❌ Error running tests: {e}", style="red")
    
    def show_help(self):
        """Show help information"""
        help_text = """
🖍️ [bold cyan]Query Matcher Help[/bold cyan]

This application uses machine learning to classify customer service queries into categories and topics.

🎯 [bold]How it works:[/bold]
1. Load training data from YAML files
2. Train ML model (TF-IDF + Naive Bayes or Sentence Transformers)
3. Classify new queries based on learned patterns

📁 [bold]Data Format:[/bold]
Training data is in YAML format with categories, topics, and example queries.

🔍 [bold]Accuracy:[/bold]
The system typically achieves 75-85% accuracy on well-trained data.

📞 [bold]Support:[/bold]
For issues or questions, check the project documentation or run BDD tests.
        """
        
        console.print(Panel(
            Text(help_text, style="white"),
            title="❓ Help & Documentation",
            border_style="yellow",
            box=box.DOUBLE
        ))
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                self.show_welcome_screen()
                self.show_main_menu()
                
                choice = IntPrompt.ask(
                    "[bold cyan]Select an option[/bold cyan]",
                    choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    default=1
                )
                
                console.print()  # Add spacing
                
                if choice == 1:
                    self.initialize_system()
                    if self.is_initialized:
                        # After successful init, go straight to query classification
                        console.print("\n🚀 [bold green]Ready! Let's classify some queries![/bold green]")
                        while True:
                            self.classify_single_query()
                            
                            continue_prompt = Prompt.ask(
                                "\n🔄 Classify another query?",
                                choices=["y", "n", "m"],
                                default="y"
                            )
                            
                            if continue_prompt == "n":
                                console.print("👋 Thanks for using Query Matcher!", style="green")
                                return
                            elif continue_prompt == "m":
                                break  # Go back to main menu
                
                elif choice == 2:
                    self.classify_single_query()
                    
                elif choice == 3:
                    console.print("🔧 [yellow]Batch processing coming soon![/yellow]")
                    
                elif choice == 4:
                    self.show_system_status()
                    
                elif choice == 5:
                    self.show_categories_info()
                    
                elif choice == 6:
                    console.print("🔧 [yellow]Advanced options coming soon![/yellow]")
                    
                elif choice == 7:
                    self.run_bdd_tests()
                    
                elif choice == 8:
                    self.show_help()
                    
                elif choice == 9:
                    console.print("👋 [bold green]Thanks for using Query Matcher! Goodbye![/bold green]")
                    break
                
                # Wait for user before continuing
                if choice != 1:  # Skip for quick start which has its own flow
                    Prompt.ask("\n⏎️ Press Enter to continue", default="")
                    
        except KeyboardInterrupt:
            console.print("\n\n👋 [bold yellow]Interrupted by user. Goodbye![/bold yellow]")
        except Exception as e:
            console.print(f"\n\n❌ [bold red]Unexpected error: {e}[/bold red]")


def main():
    """Application entry point"""
    app = QueryMatcherMenu()
    app.run()


if __name__ == '__main__':
    main()
