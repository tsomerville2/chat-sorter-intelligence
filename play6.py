#!/usr/bin/env python3
"""
Query Matcher - Iteration 6 (Contrastive Learning with Augmentation)
Based on 2024-2025 research: SimCSE, HGCLADA, and prompt-based augmentation
Target: Exceed 82.5% accuracy through contrastive learning and data augmentation
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
from typing import List, Tuple, Dict, Optional, Set
import yaml
from dataclasses import dataclass
from load_test_data import load_comprehensive_test_data, load_test_data_sample, get_test_stats
import random
import re

try:
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from sentence_transformers import evaluation
    from torch.utils.data import DataLoader
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: Install sentence-transformers and torch for full functionality")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

console = Console()


@dataclass 
class ContrastiveResult:
    category: str
    topic: str
    confidence: float
    augmentation_used: bool
    contrastive_score: float


class DataAugmenter:
    """Advanced data augmentation techniques from 2024-2025 research"""
    
    def __init__(self):
        self.synonym_map = {
            # Common synonyms in customer service
            "can't": ["cannot", "unable to", "can not", "couldn't"],
            "login": ["log in", "sign in", "access", "enter"],
            "password": ["passcode", "passphrase", "credentials", "pin"],
            "payment": ["transaction", "charge", "billing"],
            "card": ["credit card", "debit card", "payment method"],
            "declined": ["rejected", "failed", "denied", "not accepted"],
            "order": ["purchase", "transaction", "shipment"],
            "package": ["parcel", "delivery", "shipment", "item"],
            "refund": ["reimbursement", "money back", "return"],
            "account": ["profile", "user account", "membership"],
            "issue": ["problem", "error", "trouble", "difficulty"],
            "help": ["assistance", "support", "aid"],
            "wrong": ["incorrect", "mistaken", "error", "bad"],
        }
        
    def augment_with_synonyms(self, text: str) -> List[str]:
        """Generate variations using synonym replacement"""
        augmented = [text]  # Include original
        words = text.lower().split()
        
        for word in words:
            if word in self.synonym_map:
                for synonym in self.synonym_map[word][:2]:  # Limit to 2 synonyms
                    new_text = text.lower().replace(word, synonym)
                    augmented.append(new_text)
        
        return list(set(augmented))[:3]  # Return up to 3 variations
    
    def augment_with_paraphrase(self, text: str) -> List[str]:
        """Generate paraphrases using simple patterns"""
        augmented = [text]
        
        # Pattern-based paraphrasing
        patterns = [
            (r"I can't (.*)", r"I am unable to \1"),
            (r"my (.*) doesn't work", r"my \1 is not working"),
            (r"where is my (.*)", r"where's my \1"),
            (r"I need (.*)", r"I require \1"),
            (r"please (.*)", r"could you \1"),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                augmented.append(new_text)
        
        return list(set(augmented))[:3]
    
    def augment_with_dropout(self, text: str, dropout_rate: float = 0.1) -> str:
        """Randomly drop words (SimCSE-style augmentation)"""
        words = text.split()
        if len(words) <= 2:
            return text
        
        kept_words = []
        for word in words:
            if random.random() > dropout_rate:
                kept_words.append(word)
        
        return ' '.join(kept_words) if kept_words else text


class ContrastiveLearningClassifier:
    """
    Implements contrastive learning with data augmentation
    Based on SimCSE, HGCLADA, and 2024-2025 research
    """
    
    def __init__(self):
        self.model = None
        self.augmenter = DataAugmenter()
        self.training_data = []
        self.training_labels = []
        self.categories = {}
        self.label_to_category = {}
        self.classifier_head = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
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
                        label = f"{category_name}:{topic_name}"
                        
                        for example in examples:
                            self.training_data.append(example)
                            self.training_labels.append(label)
                            self.label_to_category[label] = (category_name, topic_name)
                
                console.print(f"[green]✓ Loaded {len(self.training_data)} training examples[/green]")
                console.print(f"[green]✓ {len(set(self.training_labels))} unique labels[/green]")
                
                return True
                
        except Exception as e:
            console.print(f"[red]Error loading data: {e}[/red]")
            return False
    
    def create_contrastive_pairs(self) -> List[InputExample]:
        """Create positive and negative pairs for contrastive learning"""
        pairs = []
        
        # Group examples by label
        label_examples = {}
        for text, label in zip(self.training_data, self.training_labels):
            if label not in label_examples:
                label_examples[label] = []
            label_examples[label].append(text)
        
        # Create positive pairs (same label) and negative pairs (different labels)
        for label, examples in label_examples.items():
            # Positive pairs within same label
            for i, example in enumerate(examples):
                # Augment to create positive pairs
                augmented_versions = self.augmenter.augment_with_synonyms(example)
                augmented_versions.extend(self.augmenter.augment_with_paraphrase(example))
                
                for aug_text in augmented_versions:
                    if aug_text != example:
                        # Positive pair
                        pairs.append(InputExample(texts=[example, aug_text], label=1.0))
                
                # Create negative pairs with other labels
                other_labels = [l for l in label_examples.keys() if l != label]
                if other_labels:
                    neg_label = random.choice(other_labels)
                    neg_example = random.choice(label_examples[neg_label])
                    pairs.append(InputExample(texts=[example, neg_example], label=0.0))
        
        console.print(f"[yellow]✓ Created {len(pairs)} contrastive pairs[/yellow]")
        return pairs
    
    def train(self):
        """Train model with contrastive learning"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            console.print("[red]Sentence transformers not available![/red]")
            return False
        
        console.print("[magenta]Training with contrastive learning...[/magenta]")
        
        try:
            # Initialize base model
            console.print("[yellow]Loading base model: all-mpnet-base-v2[/yellow]")
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            # Create contrastive pairs
            contrastive_pairs = self.create_contrastive_pairs()
            
            if len(contrastive_pairs) > 0:
                # Fine-tune with contrastive learning
                train_dataloader = DataLoader(contrastive_pairs, shuffle=True, batch_size=16)
                train_loss = losses.CosineSimilarityLoss(self.model)
                
                # Quick training (1 epoch for speed)
                console.print("[yellow]Fine-tuning with contrastive loss...[/yellow]")
                self.model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=1,
                    warmup_steps=100,
                    show_progress_bar=False
                )
                console.print("[green]✓ Contrastive fine-tuning complete[/green]")
            
            # Create augmented training data
            augmented_data = []
            augmented_labels = []
            
            for text, label in zip(self.training_data, self.training_labels):
                # Original
                augmented_data.append(text)
                augmented_labels.append(label)
                
                # Add augmented versions
                for aug_text in self.augmenter.augment_with_synonyms(text):
                    if aug_text != text:
                        augmented_data.append(aug_text)
                        augmented_labels.append(label)
            
            console.print(f"[yellow]✓ Augmented to {len(augmented_data)} training examples[/yellow]")
            
            # Encode all training data
            console.print("[yellow]Encoding training data...[/yellow]")
            embeddings = self.model.encode(augmented_data)
            
            # Normalize and scale embeddings
            embeddings = self.scaler.fit_transform(embeddings)
            
            # Train classification head
            console.print("[yellow]Training classification head...[/yellow]")
            self.classifier_head = LogisticRegression(max_iter=1000, C=1.0)
            self.classifier_head.fit(embeddings, augmented_labels)
            
            self.is_trained = True
            console.print("[green]✓ Contrastive learning model ready![/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Training failed: {e}[/red]")
            return False
    
    def predict(self, query: str) -> ContrastiveResult:
        """Predict with contrastive model"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        try:
            # Generate augmented versions of query
            augmented_queries = [query]
            augmented_queries.extend(self.augmenter.augment_with_synonyms(query))
            
            # Encode all versions
            embeddings = self.model.encode(augmented_queries)
            embeddings = self.scaler.transform(embeddings)
            
            # Get predictions for all versions
            predictions = self.classifier_head.predict(embeddings)
            probs = self.classifier_head.predict_proba(embeddings)
            
            # Ensemble voting on augmented predictions
            label_votes = {}
            label_confidences = {}
            
            for pred, prob in zip(predictions, probs):
                if pred not in label_votes:
                    label_votes[pred] = 0
                    label_confidences[pred] = []
                label_votes[pred] += 1
                label_confidences[pred].append(np.max(prob))
            
            # Get best prediction
            best_label = max(label_votes.items(), key=lambda x: x[1])[0]
            avg_confidence = np.mean(label_confidences[best_label])
            
            # Calculate contrastive score (how different from other classes)
            all_probs = probs[0]  # Use original query probabilities
            sorted_probs = sorted(all_probs, reverse=True)
            contrastive_score = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            
            # Parse label
            if best_label in self.label_to_category:
                category, topic = self.label_to_category[best_label]
            else:
                if ':' in best_label:
                    category, topic = best_label.split(':', 1)
                else:
                    category = best_label
                    topic = "unknown"
            
            return ContrastiveResult(
                category=category,
                topic=topic,
                confidence=avg_confidence,
                augmentation_used=len(augmented_queries) > 1,
                contrastive_score=contrastive_score
            )
            
        except Exception as e:
            console.print(f"[red]Prediction error: {e}[/red]")
            return ContrastiveResult(
                category="error",
                topic="error",
                confidence=0.0,
                augmentation_used=False,
                contrastive_score=0.0
            )


def display_welcome():
    """Display welcome screen"""
    console.clear()
    welcome_text = Text()
    welcome_text.append("🚀 ", style="bold magenta")
    welcome_text.append("QUERY MATCHER v6", style="bold yellow")
    welcome_text.append(" - Contrastive Learning", style="bold cyan")
    
    panel = Panel(
        "[magenta]Advanced Contrastive Learning with Data Augmentation[/magenta]\n"
        "[dim]SimCSE + Synonym/Paraphrase Augmentation + Ensemble Voting[/dim]",
        title=welcome_text,
        border_style="bright_magenta",
        padding=(1, 2),
        box=box.DOUBLE
    )
    console.print(panel)
    console.print()


def run_full_test(classifier):
    """Run test on all 777 queries"""
    console.print("[bold magenta]Running Full Test Suite (777 queries)[/bold magenta]\n")
    
    test_cases = load_comprehensive_test_data()
    
    if not test_cases:
        console.print("[red]Failed to load test data![/red]")
        return 0
    
    console.print(f"[dim]Loaded {len(test_cases)} test cases[/dim]\n")
    
    correct = 0
    category_accuracy = {}
    total_confidence = 0
    
    start_time = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console
    ) as progress:
        task = progress.add_task("[magenta]Testing with contrastive model...", total=len(test_cases))
        
        for i, (query, expected_cat, expected_topic) in enumerate(test_cases):
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                
                total_confidence += result.confidence
                
                # Track category accuracy
                if expected_cat not in category_accuracy:
                    category_accuracy[expected_cat] = {'correct': 0, 'total': 0}
                category_accuracy[expected_cat]['total'] += 1
                if result.category == expected_cat and result.topic == expected_topic:
                    category_accuracy[expected_cat]['correct'] += 1
                
            except Exception as e:
                pass
            
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    # Display results
    accuracy = (correct / len(test_cases)) * 100
    avg_confidence = total_confidence / len(test_cases)
    
    console.print(f"\n[bold green]═══ Final Results ═══[/bold green]")
    console.print(f"[bold]Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)[/bold]")
    console.print(f"[bold]Average Confidence: {avg_confidence:.1%}[/bold]")
    console.print(f"[bold]Time Taken: {elapsed_time:.2f} seconds[/bold]")
    console.print(f"[bold]Speed: {len(test_cases)/elapsed_time:.0f} queries/second[/bold]\n")
    
    # Comparison with previous iterations
    console.print("[bold]Comparison with Previous Iterations:[/bold]")
    if accuracy > 82.5:
        console.print(f"[green]✓ Contrastive (v6): {accuracy:.1f}% - NEW BEST![/green]")
        console.print(f"[dim]  play2.C3: 82.5% (previous best)[/dim]")
        console.print(f"[bold green]🎉 BREAKTHROUGH: +{accuracy - 82.5:.1f}% improvement![/bold green]")
    else:
        console.print(f"[yellow]Contrastive (v6): {accuracy:.1f}%[/yellow]")
        console.print(f"[green]play2.C3: 82.5% (still best)[/green]")
        console.print(f"[dim]Need {82.5 - accuracy:.1f}% more to beat current best[/dim]")
    
    return accuracy


def main():
    """Main application"""
    console.print("[magenta]Initializing Contrastive Learning Classifier...[/magenta]")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        console.print("[red]Required libraries not installed![/red]")
        console.print("[yellow]Please install: pip install sentence-transformers torch[/yellow]")
        sys.exit(1)
    
    classifier = ContrastiveLearningClassifier()
    
    if not classifier.load_data("data/training_data.yaml"):
        console.print("[red]Failed to load training data![/red]")
        sys.exit(1)
    
    console.print("\n[magenta]Training with contrastive learning...[/magenta]")
    if not classifier.train():
        console.print("[red]Failed to train model![/red]")
        sys.exit(1)
    
    console.print("[green]✓ Contrastive model ready![/green]\n")
    
    # Run full test
    accuracy = run_full_test(classifier)
    
    return accuracy


if __name__ == "__main__":
    try:
        accuracy = main()
        # Return accuracy for tracking
        sys.exit(0 if accuracy > 82.5 else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)