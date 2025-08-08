#!/usr/bin/env python3
"""
Test Iteration 2 (Cosine Similarity) on the training set itself
to find correct classifications with 0 similarity score
"""

import sys
import yaml
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.append('iterations/iteration2')
from cosine_classifier import CosineClassifier

console = Console()


def test_training_set():
    """Test cosine similarity on the training examples"""
    
    # Load classifier
    classifier = CosineClassifier()
    classifier.load_data("data/training_data.yaml")
    
    # Load training data to get all examples
    with open("data/training_data.yaml", 'r') as f:
        data = yaml.safe_load(f)
    
    console.print("[bold magenta]Testing Iteration 2 on Training Set[/bold magenta]\n")
    
    # Track results
    perfect_matches = []
    zero_score_correct = []
    zero_score_incorrect = []
    partial_matches = []
    
    # Test each training example
    for category_name, category_data in data['categories'].items():
        for topic_name, topic_data in category_data['topics'].items():
            for example in topic_data['examples']:
                # Predict using cosine similarity
                result = classifier.predict(example)
                
                # Check if prediction is correct
                correct_category = result.category == category_name
                correct_topic = result.topic == topic_name
                fully_correct = correct_category and correct_topic
                
                # Categorize results
                if result.similarity_score == 1.0 and fully_correct:
                    perfect_matches.append({
                        'query': example,
                        'category': category_name,
                        'topic': topic_name,
                        'score': result.similarity_score
                    })
                elif result.similarity_score == 0.0 and fully_correct:
                    zero_score_correct.append({
                        'query': example,
                        'expected': f"{category_name}:{topic_name}",
                        'predicted': f"{result.category}:{result.topic}",
                        'matched_example': result.matched_example
                    })
                elif result.similarity_score == 0.0 and not fully_correct:
                    zero_score_incorrect.append({
                        'query': example,
                        'expected': f"{category_name}:{topic_name}",
                        'predicted': f"{result.category}:{result.topic}",
                        'matched_example': result.matched_example
                    })
                elif 0 < result.similarity_score < 1.0:
                    partial_matches.append({
                        'query': example,
                        'expected': f"{category_name}:{topic_name}",
                        'predicted': f"{result.category}:{result.topic}",
                        'score': result.similarity_score,
                        'correct': fully_correct,
                        'matched_example': result.matched_example
                    })
    
    # Print summary statistics
    console.print(f"[green]Perfect matches (score=1.0, correct):[/green] {len(perfect_matches)}")
    console.print(f"[yellow]Zero score but correct:[/yellow] {len(zero_score_correct)}")
    console.print(f"[red]Zero score and incorrect:[/red] {len(zero_score_incorrect)}")
    console.print(f"[cyan]Partial matches (0 < score < 1):[/cyan] {len(partial_matches)}\n")
    
    # Show zero score correct matches (these are the interesting ones)
    if zero_score_correct:
        console.print("[bold yellow]⚠️  Correct Classifications with Zero Similarity Score:[/bold yellow]")
        console.print("[dim]These were classified correctly despite having no similarity![/dim]\n")
        
        table = Table(box=box.ROUNDED)
        table.add_column("Query", style="white", width=40)
        table.add_column("Expected", style="green")
        table.add_column("Got", style="yellow")
        table.add_column("Matched Example", style="dim")
        
        for item in zero_score_correct:
            table.add_row(
                item['query'][:40],
                item['expected'],
                item['predicted'],
                item['matched_example'][:30] + "..."
            )
        
        console.print(table)
        console.print()
    
    # Show zero score incorrect matches
    if zero_score_incorrect:
        console.print("[bold red]❌ Incorrect Classifications with Zero Similarity Score:[/bold red]\n")
        
        table = Table(box=box.ROUNDED)
        table.add_column("Query", style="white", width=40)
        table.add_column("Expected", style="green")
        table.add_column("Got", style="red")
        table.add_column("Matched Example", style="dim")
        
        for item in zero_score_incorrect[:10]:  # Show first 10
            table.add_row(
                item['query'][:40],
                item['expected'],
                item['predicted'],
                item['matched_example'][:30] + "..."
            )
        
        console.print(table)
        console.print()
    
    # Show partial matches
    if partial_matches:
        console.print("[bold cyan]Partial Matches (0 < score < 1):[/bold cyan]\n")
        
        # Sort by score
        partial_matches.sort(key=lambda x: x['score'], reverse=True)
        
        table = Table(box=box.ROUNDED)
        table.add_column("Query", style="white", width=35)
        table.add_column("Expected", style="cyan")
        table.add_column("Got", style="magenta")
        table.add_column("Score", style="yellow")
        table.add_column("✓/✗", style="green")
        
        for item in partial_matches[:15]:  # Show top 15
            table.add_row(
                item['query'][:35],
                item['expected'],
                item['predicted'],
                f"{item['score']:.3f}",
                "✓" if item['correct'] else "✗"
            )
        
        console.print(table)
        console.print()
    
    # Final analysis
    console.print("[bold]Analysis:[/bold]")
    console.print(f"• Training set has {len(perfect_matches) + len(zero_score_correct) + len(zero_score_incorrect) + len(partial_matches)} examples")
    console.print(f"• {len(perfect_matches)} examples match themselves perfectly (expected)")
    console.print(f"• {len(partial_matches)} examples have partial similarity to other examples")
    console.print(f"• {len(zero_score_incorrect)} examples have NO similarity to any example (including themselves!)")
    
    if zero_score_incorrect:
        console.print("\n[yellow]⚠️  WARNING: Some training examples have zero similarity even to themselves![/yellow]")
        console.print("[dim]This suggests the vectorizer might be filtering them out (stopwords, etc.)[/dim]")


if __name__ == "__main__":
    test_training_set()