#!/usr/bin/env python3
"""
Test edge cases and non-training queries on Iteration 2
to find correct matches with 0 similarity
"""

import sys
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.append('iterations/iteration2')
from cosine_classifier import CosineClassifier

console = Console()


def test_edge_cases():
    """Test queries that aren't in training set"""
    
    # Load classifier
    classifier = CosineClassifier()
    classifier.load_data("data/training_data.yaml")
    
    console.print("[bold magenta]Testing Iteration 2 with Edge Cases & Variations[/bold magenta]\n")
    
    # Test queries with expected results
    test_cases = [
        # Variations of training examples
        ("credit card declined", "billing", "payment_failed"),
        ("my card was rejected", "billing", "payment_failed"),
        ("payment failed", "billing", "payment_failed"),
        ("transaction declined", "billing", "payment_failed"),
        
        # Common variations that should work
        ("forgot password", "technical_support", "password_reset"),
        ("lost my password", "technical_support", "password_reset"),
        ("reset password", "technical_support", "password_reset"),
        ("can't login", "technical_support", "password_reset"),
        ("cannot log in", "technical_support", "password_reset"),
        ("unable to login", "technical_support", "password_reset"),
        
        # Shipping variations
        ("track package", "shipping", "track_order"),
        ("package status", "shipping", "track_order"),
        ("where's my order", "shipping", "track_order"),
        ("delivery status", "shipping", "track_order"),
        
        # Refund variations
        ("refund please", "billing", "refund_request"),
        ("want refund", "billing", "refund_request"),
        ("money back please", "billing", "refund_request"),
        
        # Very short queries
        ("password", "technical_support", "password_reset"),
        ("refund", "billing", "refund_request"),
        ("package", "shipping", "track_order"),
        ("payment", "billing", "payment_failed"),
        
        # Typos and misspellings
        ("pasword reset", "technical_support", "password_reset"),
        ("refnd request", "billing", "refund_request"),
        ("were is my pakage", "shipping", "track_order"),
    ]
    
    results = []
    zero_similarity_results = []
    
    for query, expected_cat, expected_topic in test_cases:
        result = classifier.predict(query)
        
        correct = (result.category == expected_cat and result.topic == expected_topic)
        
        results.append({
            'query': query,
            'expected': f"{expected_cat}:{expected_topic}",
            'predicted': f"{result.category}:{result.topic}",
            'score': result.similarity_score,
            'correct': correct,
            'matched_example': result.matched_example
        })
        
        # Track zero similarity cases
        if result.similarity_score == 0.0:
            zero_similarity_results.append({
                'query': query,
                'expected': f"{expected_cat}:{expected_topic}",
                'predicted': f"{result.category}:{result.topic}",
                'correct': correct,
                'matched_example': result.matched_example
            })
    
    # Display all results
    table = Table(title="Edge Cases & Variations Test", box=box.ROUNDED)
    table.add_column("Query", style="white", width=25)
    table.add_column("Expected", style="cyan", width=25)
    table.add_column("Got", style="magenta", width=25)
    table.add_column("Score", style="yellow")
    table.add_column("✓/✗", style="green")
    
    for r in results:
        style = "green" if r['correct'] else "red"
        table.add_row(
            r['query'],
            r['expected'],
            r['predicted'],
            f"{r['score']:.3f}",
            "✓" if r['correct'] else "✗",
            style=style if not r['correct'] else None
        )
    
    console.print(table)
    console.print()
    
    # Highlight zero similarity cases
    if zero_similarity_results:
        console.print(f"[bold yellow]Found {len(zero_similarity_results)} queries with ZERO similarity:[/bold yellow]\n")
        
        correct_zeros = [r for r in zero_similarity_results if r['correct']]
        incorrect_zeros = [r for r in zero_similarity_results if not r['correct']]
        
        if correct_zeros:
            console.print(f"[green]✓ {len(correct_zeros)} were classified CORRECTLY despite zero similarity:[/green]")
            for r in correct_zeros:
                console.print(f"  • \"{r['query']}\" → {r['predicted']} (matched: \"{r['matched_example'][:30]}...\")")
        
        if incorrect_zeros:
            console.print(f"\n[red]✗ {len(incorrect_zeros)} were classified INCORRECTLY with zero similarity:[/red]")
            for r in incorrect_zeros:
                console.print(f"  • \"{r['query']}\" → Expected: {r['expected']}, Got: {r['predicted']}")
    
    # Calculate statistics
    correct_count = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct_count / total) * 100
    
    avg_score_correct = sum(r['score'] for r in results if r['correct']) / max(1, correct_count)
    avg_score_incorrect = sum(r['score'] for r in results if not r['correct']) / max(1, total - correct_count)
    
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"• Accuracy: {correct_count}/{total} ({accuracy:.1f}%)")
    console.print(f"• Average similarity for correct: {avg_score_correct:.3f}")
    console.print(f"• Average similarity for incorrect: {avg_score_incorrect:.3f}")
    console.print(f"• Queries with zero similarity: {len(zero_similarity_results)}")
    
    # Key insights
    console.print("\n[bold cyan]Key Insights:[/bold cyan]")
    console.print("• Short queries like 'password' or 'refund' often get zero similarity")
    console.print("• Common variations like 'can't login' fail completely (zero similarity)")
    console.print("• Typos usually result in zero similarity matches")
    console.print("• The system defaults to the first training example when similarity is zero")


if __name__ == "__main__":
    test_edge_cases()