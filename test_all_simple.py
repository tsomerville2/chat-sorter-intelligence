#!/usr/bin/env python3
"""
Simple test runner that tests each approach independently
"""

import subprocess
import time
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def run_test_script(script_name, options):
    """Run a test script with given options and parse output"""
    try:
        # Run the script with the options
        result = subprocess.run(
            f'echo "{options}" | python {script_name}',
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout
        
        # Parse accuracy from output
        if "Overall Accuracy:" in output:
            for line in output.split('\n'):
                if "Overall Accuracy:" in line:
                    # Extract percentage
                    parts = line.split('(')
                    if len(parts) > 1:
                        pct = parts[1].split('%')[0]
                        return float(pct)
        
        # For quick test results
        if "Accuracy:" in output and "/5" in output:
            for line in output.split('\n'):
                if "Accuracy:" in line and "/5" in line:
                    # Extract percentage
                    parts = line.split('(')
                    if len(parts) > 1:
                        pct = parts[1].split('%')[0]
                        return float(pct)
        
        return None
    except Exception as e:
        console.print(f"[red]Error testing {script_name}: {e}[/red]")
        return None


def main():
    console.print("[bold magenta]Testing All Query Matcher Variants[/bold magenta]")
    console.print("[yellow]Running 100 sample tests where available, quick tests otherwise[/yellow]\n")
    
    tests = [
        # play1 variants
        ("play1.py - Original ML", "play1.py", "4\n0"),  # Sample test (100)
        ("play1.py - Quick Test", "play1.py", "3\n0"),   # Quick test (5)
        
        # play2 variants
        ("play2.py - Original Cosine", "play2.py", "5\n0"),  # Sample test (100)
        ("play2.py - Quick Test", "play2.py", "4\n0"),       # Quick test (5)
        ("play2.A.py - No Stop Words (100)", "play2.A.py", "5\n0"),  # Sample test
        ("play2.A.py - No Stop Words (5)", "play2.A.py", "4\n0"),    # Quick test
        
        # play2.B variants - need to switch metric first
        ("play2.B - Jaccard (100)", "play2.B.py", "5\n0"),           # Jaccard (default)
        ("play2.B - Jaccard (5)", "play2.B.py", "4\n0"),             # Quick test
        ("play2.B - Char N-grams (5)", "play2.B.py", "3\n3\n4\n0"),  # Switch to char, then test
        
        # play1.A
        ("play1.A - Enhanced ML (100)", "play1.A.py", "4\n0"),  # Sample test
        ("play1.A - Enhanced ML (5)", "play1.A.py", "3\n0"),    # Quick test
        
        # play1.B variants
        ("play1.B - SVM (100)", "play1.B.py", "4\n0"),          # Sample test (default SVM)
        ("play1.B - Compare All (5)", "play1.B.py", "5\n0"),    # Compare all classifiers
        
        # play2.C
        ("play2.C - Embeddings (100)", "play2.C.py", "4\n0"),   # Sample test
        ("play2.C - Embeddings (5)", "play2.C.py", "3\n0"),     # Quick test
    ]
    
    results = []
    
    for test_name, script, options in tests:
        console.print(f"[cyan]Testing {test_name}...[/cyan]")
        accuracy = run_test_script(script, options)
        
        if accuracy is not None:
            results.append((test_name, accuracy))
            console.print(f"  → {accuracy:.1f}% accuracy")
        else:
            console.print(f"  → [red]Failed to get results[/red]")
        
        time.sleep(0.5)  # Small delay between tests
    
    # Display results table
    console.print("\n[bold green]═══ TEST RESULTS ═══[/bold green]\n")
    
    table = Table(title="All Test Results", box=box.ROUNDED)
    table.add_column("Test", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Dataset Size", style="yellow")
    
    # Sort by accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    
    for test_name, accuracy in results:
        if "(100)" in test_name or "Sample" in test_name:
            dataset = "100 samples"
        elif "(5)" in test_name or "Quick" in test_name:
            dataset = "5 queries"
        else:
            dataset = "Unknown"
        
        table.add_row(
            test_name,
            f"{accuracy:.1f}%",
            dataset
        )
    
    console.print(table)
    
    # Find best performers
    best_100 = [r for r in results if "(100)" in r[0] or "Sample" in r[0]]
    best_5 = [r for r in results if "(5)" in r[0] or "Quick" in r[0]]
    
    if best_100:
        console.print(f"\n[bold green]🏆 Best on 100 samples: {best_100[0][0]} - {best_100[0][1]:.1f}%[/bold green]")
    
    if best_5:
        console.print(f"[bold yellow]🥇 Best on 5 queries: {best_5[0][0]} - {best_5[0][1]:.1f}%[/bold yellow]")


if __name__ == "__main__":
    main()