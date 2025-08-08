#!/usr/bin/env python3
"""
Query Matcher - Iteration 8 (Improved play2.C3)
Takes the successful play2.C3 approach (82.5%) and enhances it with:
- Better base model (MPNet instead of MiniLM)
- Fine-tuning on training data
- Adaptive confidence thresholds per category
- Error-based active learning
Target: EXCEED 82.5% and achieve 85%+ accuracy
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
from typing import List, Tuple, Dict, Optional, Union
import yaml
from dataclasses import dataclass
from load_test_data import load_comprehensive_test_data, load_test_data_sample
from pydantic import BaseModel

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

console = Console()


class MatchDetails(BaseModel):
    """Details of a single match"""
    category: str
    topic: str
    confidence: float
    matched_example: str
    similarity_score: float


class SingleAnswer(BaseModel):
    """Response when confidence is high"""
    answer_type: str = "single"
    match: MatchDetails
    model_used: str


class MultipleAnswers(BaseModel):
    """Response when query is ambiguous"""
    answer_type: str = "multiple"
    matches: List[MatchDetails]
    model_used: str
    reason: str


ClassificationResult = Union[SingleAnswer, MultipleAnswers]


class ImprovedProductionClassifier:
    """
    Enhanced version of play2.C3's ProductionEmbeddingClassifier
    Uses better model, fine-tuning, and adaptive thresholds
    """
    
    def __init__(self,
                 model_name: str = 'all-mpnet-base-v2',  # Better than MiniLM
                 confidence_threshold: float = 0.55,      # Slightly lower for MPNet
                 alternatives_count: int = 3):
        """
        Initialize classifier
        
        Args:
            model_name: Sentence transformer model to use (MPNet is better)
            confidence_threshold: Below this, return multiple alternatives
            alternatives_count: Number of alternatives to return when uncertain
        """
        self.use_embeddings = SENTENCE_TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.alternatives_count = alternatives_count
        self.category_thresholds = {}  # Adaptive thresholds per category
        
        # Data storage
        self.training_examples = []
        self.training_labels = []
        self.training_embeddings = None
        self.categories = {}
        
        # Models
        self.model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Error tracking for active learning
        self.error_patterns = defaultdict(list)
        
    def load_data(self, filepath: str) -> bool:
        """Load training data from YAML"""
        try:
            with open(filepath, 'r') as file:
                yaml_data = yaml.safe_load(file)
                self.categories = yaml_data.get('categories', {})
                
                # Prepare training data
                for category_name, category_data in self.categories.items():
                    topics = category_data.get('topics', {})
                    
                    for topic_name, topic_data in topics.items():
                        examples = topic_data.get('examples', [])
                        
                        for example in examples:
                            self.training_examples.append(example)
                            self.training_labels.append(f"{category_name}:{topic_name}")
                
                console.print(f"[green]✓ Loaded {len(self.training_examples)} training examples[/green]")
                console.print(f"[green]✓ {len(set(self.training_labels))} unique labels[/green]")
                
                return True
                
        except Exception as e:
            console.print(f"[red]Error loading data: {e}[/red]")
            return False
    
    def train(self):
        """Train the classifier with fine-tuning"""
        console.print("[magenta]Training Enhanced Production Classifier...[/magenta]")
        
        if self.use_embeddings:
            console.print(f"[yellow]Loading {self.model_name}...[/yellow]")
            try:
                self.model = SentenceTransformer(f'sentence-transformers/{self.model_name}')
                
                # Fine-tune the model with contrastive learning
                if self._fine_tune_model():
                    console.print("[green]✓ Model fine-tuned successfully[/green]")
                
                # Create embeddings
                console.print("[yellow]Creating embeddings...[/yellow]")
                self.training_embeddings = self.model.encode(
                    self.training_examples,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                console.print(f"[green]✓ Created {len(self.training_embeddings)} embeddings[/green]")
                
                # Calculate adaptive thresholds per category
                self._calculate_adaptive_thresholds()
                
            except Exception as e:
                console.print(f"[red]Embedding model failed: {e}[/red]")
                console.print("[yellow]Falling back to TF-IDF[/yellow]")
                self.use_embeddings = False
        
        if not self.use_embeddings:
            # Fallback to TF-IDF with character n-grams
            console.print("[yellow]Training TF-IDF with character n-grams...[/yellow]")
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 5),
                max_features=5000
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.training_examples)
            console.print("[green]✓ TF-IDF model trained[/green]")
        
        return True
    
    def _fine_tune_model(self) -> bool:
        """Fine-tune the model with contrastive learning"""
        try:
            console.print("[dim]Creating contrastive pairs...[/dim]")
            
            # Create positive pairs from same label
            train_examples = []
            label_to_examples = defaultdict(list)
            
            for example, label in zip(self.training_examples, self.training_labels):
                label_to_examples[label].append(example)
            
            # Create positive pairs (same label)
            for label, examples in label_to_examples.items():
                if len(examples) >= 2:
                    for i in range(len(examples)):
                        for j in range(i + 1, min(i + 3, len(examples))):
                            train_examples.append(InputExample(
                                texts=[examples[i], examples[j]],
                                label=1.0
                            ))
            
            # Create negative pairs (different labels)
            all_labels = list(label_to_examples.keys())
            for i, label1 in enumerate(all_labels):
                for label2 in all_labels[i+1:i+3]:  # Limit negative pairs
                    if label1 != label2:
                        ex1 = label_to_examples[label1][0]
                        ex2 = label_to_examples[label2][0]
                        train_examples.append(InputExample(
                            texts=[ex1, ex2],
                            label=0.0
                        ))
            
            if len(train_examples) > 0:
                console.print(f"[dim]Fine-tuning with {len(train_examples)} pairs...[/dim]")
                
                # Create DataLoader
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
                train_loss = losses.CosineSimilarityLoss(self.model)
                
                # Fine-tune (1 epoch for speed)
                self.model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=1,
                    warmup_steps=10,
                    show_progress_bar=False
                )
                
                return True
            
        except Exception as e:
            console.print(f"[yellow]Fine-tuning skipped: {e}[/yellow]")
        
        return False
    
    def _calculate_adaptive_thresholds(self):
        """Calculate optimal confidence thresholds per category"""
        if not self.use_embeddings:
            return
        
        console.print("[dim]Calculating adaptive thresholds...[/dim]")
        
        # Group by category
        category_similarities = defaultdict(list)
        
        for i, label in enumerate(self.training_labels):
            category = label.split(':')[0]
            
            # Get similarities to other examples in same category
            for j, other_label in enumerate(self.training_labels):
                if i != j and other_label.startswith(category):
                    sim = cosine_similarity(
                        [self.training_embeddings[i]],
                        [self.training_embeddings[j]]
                    )[0][0]
                    category_similarities[category].append(sim)
        
        # Set thresholds based on category statistics
        for category, sims in category_similarities.items():
            if sims:
                # Use mean - 1 std as threshold
                mean_sim = np.mean(sims)
                std_sim = np.std(sims)
                self.category_thresholds[category] = max(0.4, mean_sim - std_sim)
            else:
                self.category_thresholds[category] = self.confidence_threshold
        
        console.print(f"[dim]Adaptive thresholds: {self.category_thresholds}[/dim]")
    
    def get_top_k_matches(self, query: str, k: int = 3) -> List[MatchDetails]:
        """Get top k matches for a query"""
        if self.use_embeddings and self.training_embeddings is not None:
            # Use embeddings
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            similarities = cosine_similarity(query_embedding, self.training_embeddings)[0]
        else:
            # Use TF-IDF
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        matches = []
        for idx in top_indices:
            label_parts = self.training_labels[idx].split(':')
            category = label_parts[0]
            topic = label_parts[1] if len(label_parts) > 1 else "unknown"
            
            # Use adaptive threshold for this category
            threshold = self.category_thresholds.get(category, self.confidence_threshold)
            
            # Adjust confidence based on threshold
            raw_confidence = float(similarities[idx])
            adjusted_confidence = raw_confidence if raw_confidence >= threshold else raw_confidence * 0.9
            
            matches.append(MatchDetails(
                category=category,
                topic=topic,
                confidence=adjusted_confidence,
                matched_example=self.training_examples[idx],
                similarity_score=float(similarities[idx])
            ))
        
        return matches
    
    def classify(self, query: str) -> ClassificationResult:
        """Classify a query with confidence-based single/multiple answers"""
        if self.training_embeddings is None and self.tfidf_matrix is None:
            raise ValueError("Classifier not trained")
        
        # Get top matches
        top_matches = self.get_top_k_matches(query, self.alternatives_count)
        
        if not top_matches:
            return MultipleAnswers(
                matches=[],
                model_used=self.model_name if self.use_embeddings else "tfidf_char_ngrams",
                reason="No matches found"
            )
        
        best_match = top_matches[0]
        
        # Use adaptive threshold for the predicted category
        category_threshold = self.category_thresholds.get(
            best_match.category, 
            self.confidence_threshold
        )
        
        # Determine if we're confident enough for a single answer
        if best_match.confidence >= category_threshold:
            return SingleAnswer(
                match=best_match,
                model_used=self.model_name if self.use_embeddings else "tfidf_char_ngrams"
            )
        else:
            # Low confidence - return multiple options
            if len(top_matches) > 1:
                confidence_gap = best_match.confidence - top_matches[1].confidence
                if confidence_gap < 0.1:
                    reason = f"Ambiguous query - top {len(top_matches)} matches have similar confidence"
                else:
                    reason = f"Low confidence ({best_match.confidence:.2f}) - presenting alternatives"
            else:
                reason = "Low confidence - only one match available"
            
            return MultipleAnswers(
                matches=top_matches,
                model_used=self.model_name if self.use_embeddings else "tfidf_char_ngrams",
                reason=reason
            )
    
    def learn_from_error(self, query: str, predicted: str, correct: str):
        """Track errors for active learning"""
        self.error_patterns[predicted].append({
            'query': query,
            'correct': correct
        })


def run_full_test(classifier):
    """Run test on all 777 queries"""
    console.print("[bold magenta]Running Full Test Suite (777 queries)[/bold magenta]\n")
    
    test_cases = load_comprehensive_test_data()
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return 0
    
    console.print(f"[dim]Loaded {len(test_cases)} test cases[/dim]\n")
    
    correct = 0
    correct_with_alternatives = 0
    category_accuracy = {}
    answer_types = {'single': 0, 'multiple': 0}
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console
    ) as progress:
        task = progress.add_task("[magenta]Testing enhanced classifier...", total=len(test_cases))
        
        for i, (query, expected_cat, expected_topic) in enumerate(test_cases):
            try:
                result = classifier.classify(query)
                expected_label = f"{expected_cat}:{expected_topic}"
                
                if result.answer_type == "single":
                    answer_types['single'] += 1
                    predicted_label = f"{result.match.category}:{result.match.topic}"
                    is_correct = predicted_label == expected_label
                    
                    if is_correct:
                        correct += 1
                        correct_with_alternatives += 1
                    else:
                        # Learn from error
                        classifier.learn_from_error(query, predicted_label, expected_label)
                else:
                    answer_types['multiple'] += 1
                    # Check if correct answer is in alternatives
                    found_in_alternatives = False
                    for match in result.matches:
                        predicted_label = f"{match.category}:{match.topic}"
                        if predicted_label == expected_label:
                            correct_with_alternatives += 1
                            found_in_alternatives = True
                            break
                    
                    if not found_in_alternatives and result.matches:
                        # Learn from error
                        predicted_label = f"{result.matches[0].category}:{result.matches[0].topic}"
                        classifier.learn_from_error(query, predicted_label, expected_label)
                
                # Track category accuracy
                if expected_cat not in category_accuracy:
                    category_accuracy[expected_cat] = {'correct': 0, 'total': 0}
                category_accuracy[expected_cat]['total'] += 1
                
                # For category accuracy, use best match
                if result.answer_type == "single":
                    if result.match.category == expected_cat and result.match.topic == expected_topic:
                        category_accuracy[expected_cat]['correct'] += 1
                elif result.matches:
                    if result.matches[0].category == expected_cat and result.matches[0].topic == expected_topic:
                        category_accuracy[expected_cat]['correct'] += 1
                
            except Exception as e:
                pass
            
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    # Display results
    single_answer_accuracy = (correct / len(test_cases)) * 100
    with_alternatives_accuracy = (correct_with_alternatives / len(test_cases)) * 100
    
    console.print(f"\n[bold green]═══ Final Results ═══[/bold green]")
    console.print(f"[bold]Single Answer Accuracy: {correct}/{len(test_cases)} ({single_answer_accuracy:.1f}%)[/bold]")
    console.print(f"[bold]With Alternatives: {correct_with_alternatives}/{len(test_cases)} ({with_alternatives_accuracy:.1f}%)[/bold]")
    console.print(f"[bold]Time Taken: {elapsed_time:.2f} seconds[/bold]")
    console.print(f"[bold]Speed: {len(test_cases)/elapsed_time:.0f} queries/second[/bold]\n")
    
    # Answer type breakdown
    console.print("[bold]Answer Types:[/bold]")
    console.print(f"  Single (confident): {answer_types['single']} ({100*answer_types['single']/len(test_cases):.1f}%)")
    console.print(f"  Multiple (ambiguous): {answer_types['multiple']} ({100*answer_types['multiple']/len(test_cases):.1f}%)")
    
    # Comparison with previous iterations
    console.print("\n[bold]Comparison with Previous Iterations:[/bold]")
    if single_answer_accuracy > 82.5:
        console.print(f"[bold green]✓ Enhanced (v8): {single_answer_accuracy:.1f}% - NEW BEST![/bold green]")
        console.print(f"[dim]  play2.C3: 82.5% (previous best)[/dim]")
        console.print(f"[bold green]🎉 BREAKTHROUGH ACHIEVED: +{single_answer_accuracy - 82.5:.1f}% improvement![/bold green]")
    else:
        console.print(f"[yellow]Enhanced (v8): {single_answer_accuracy:.1f}% (single answer)[/yellow]")
        console.print(f"[yellow]Enhanced (v8): {with_alternatives_accuracy:.1f}% (with alternatives)[/yellow]")
        console.print(f"[green]play2.C3: 82.5% (still best for single answer)[/green]")
    
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
    
    # Show error patterns if any
    if classifier.error_patterns:
        console.print("\n[bold]Common Error Patterns:[/bold]")
        for predicted, errors in list(classifier.error_patterns.items())[:5]:
            console.print(f"  {predicted}: {len(errors)} errors")
    
    return single_answer_accuracy


def main():
    """Main application"""
    console.print("[magenta]Initializing Enhanced Production Classifier...[/magenta]")
    console.print("[dim]Improving upon play2.C3 (82.5%) with better model and fine-tuning[/dim]\n")
    
    classifier = ImprovedProductionClassifier()
    
    if not classifier.load_data("data/training_data.yaml"):
        console.print("[red]Failed to load training data![/red]")
        sys.exit(1)
    
    if not classifier.train():
        console.print("[red]Failed to train classifier![/red]")
        sys.exit(1)
    
    console.print("\n[green]✓ Enhanced classifier ready![/green]\n")
    
    # Run full test
    accuracy = run_full_test(classifier)
    
    return accuracy


if __name__ == "__main__":
    try:
        accuracy = main()
        sys.exit(0 if accuracy > 82.5 else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)