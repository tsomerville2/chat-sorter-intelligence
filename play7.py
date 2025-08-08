#!/usr/bin/env python3
"""
Query Matcher - Iteration 7 (Advanced Weighted Ensemble)
Based on 2024-2025 research: 98.76% accuracy achieved with weighted voting
Combines best models with cross-efficiency weighted voting
Target: EXCEED 82.5% and achieve breakthrough performance
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
from typing import List, Tuple, Dict, Optional
import yaml
from dataclasses import dataclass
from load_test_data import load_comprehensive_test_data, load_test_data_sample
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

console = Console()


@dataclass
class EnsembleResult:
    category: str
    topic: str
    confidence: float
    voting_details: Dict[str, str]
    weighted_scores: Dict[str, float]
    method: str


class WeightedEnsembleClassifier:
    """
    Advanced weighted ensemble classifier
    Combines multiple models with optimized weights based on cross-efficiency
    Research shows 98.76% accuracy possible with proper weighting
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.training_data = []
        self.training_labels = []
        self.categories = {}
        self.label_to_category = {}
        self.scalers = {}
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
    
    def train(self):
        """Train ensemble of models with cross-efficiency weighting"""
        console.print("[magenta]Training Advanced Weighted Ensemble...[/magenta]")
        
        # 1. TF-IDF with Logistic Regression (98.76% in research)
        console.print("[yellow]Training TF-IDF + Logistic Regression...[/yellow]")
        try:
            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
            X_tfidf = tfidf.fit_transform(self.training_data)
            
            lr_model = LogisticRegression(max_iter=1000, C=1.0)
            lr_model.fit(X_tfidf, self.training_labels)
            
            self.models['tfidf_lr'] = (tfidf, lr_model)
            
            # Cross-validation score for weighting
            cv_score = cross_val_score(lr_model, X_tfidf, self.training_labels, cv=3).mean()
            self.weights['tfidf_lr'] = cv_score
            console.print(f"[green]✓ TF-IDF LR trained (CV: {cv_score:.3f})[/green]")
        except Exception as e:
            console.print(f"[red]TF-IDF LR failed: {e}[/red]")
        
        # 2. TF-IDF with Naive Bayes
        console.print("[yellow]Training TF-IDF + Naive Bayes...[/yellow]")
        try:
            tfidf2 = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
            X_tfidf2 = tfidf2.fit_transform(self.training_data)
            
            nb_model = MultinomialNB(alpha=0.1)
            nb_model.fit(X_tfidf2, self.training_labels)
            
            self.models['tfidf_nb'] = (tfidf2, nb_model)
            
            cv_score = cross_val_score(nb_model, X_tfidf2, self.training_labels, cv=3).mean()
            self.weights['tfidf_nb'] = cv_score
            console.print(f"[green]✓ TF-IDF NB trained (CV: {cv_score:.3f})[/green]")
        except Exception as e:
            console.print(f"[red]TF-IDF NB failed: {e}[/red]")
        
        # 3. Sentence Transformer with Logistic Regression
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            console.print("[yellow]Training Sentence Transformer + LR...[/yellow]")
            try:
                transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                X_emb = transformer.encode(self.training_data)
                
                # Scale embeddings
                scaler = StandardScaler()
                X_emb_scaled = scaler.fit_transform(X_emb)
                self.scalers['transformer'] = scaler
                
                lr_emb = LogisticRegression(max_iter=1000)
                lr_emb.fit(X_emb_scaled, self.training_labels)
                
                self.models['transformer_lr'] = (transformer, lr_emb)
                
                cv_score = cross_val_score(lr_emb, X_emb_scaled, self.training_labels, cv=3).mean()
                self.weights['transformer_lr'] = cv_score
                console.print(f"[green]✓ Transformer LR trained (CV: {cv_score:.3f})[/green]")
            except Exception as e:
                console.print(f"[red]Transformer LR failed: {e}[/red]")
        
        # 4. TF-IDF with Random Forest
        console.print("[yellow]Training TF-IDF + Random Forest...[/yellow]")
        try:
            tfidf3 = TfidfVectorizer(max_features=2000)
            X_tfidf3 = tfidf3.fit_transform(self.training_data)
            
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf_model.fit(X_tfidf3, self.training_labels)
            
            self.models['tfidf_rf'] = (tfidf3, rf_model)
            
            cv_score = cross_val_score(rf_model, X_tfidf3, self.training_labels, cv=3).mean()
            self.weights['tfidf_rf'] = cv_score
            console.print(f"[green]✓ TF-IDF RF trained (CV: {cv_score:.3f})[/green]")
        except Exception as e:
            console.print(f"[red]TF-IDF RF failed: {e}[/red]")
        
        # 5. Advanced Sentence Transformer (MPNet - best performer)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            console.print("[yellow]Training MPNet Transformer...[/yellow]")
            try:
                mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                X_mpnet = mpnet.encode(self.training_data)
                
                # Scale embeddings
                scaler_mpnet = StandardScaler()
                X_mpnet_scaled = scaler_mpnet.fit_transform(X_mpnet)
                self.scalers['mpnet'] = scaler_mpnet
                
                lr_mpnet = LogisticRegression(max_iter=1000, C=2.0)
                lr_mpnet.fit(X_mpnet_scaled, self.training_labels)
                
                self.models['mpnet_lr'] = (mpnet, lr_mpnet)
                
                cv_score = cross_val_score(lr_mpnet, X_mpnet_scaled, self.training_labels, cv=3).mean()
                self.weights['mpnet_lr'] = cv_score * 1.2  # Boost weight for best model
                console.print(f"[green]✓ MPNet LR trained (CV: {cv_score:.3f}, boosted)[/green]")
            except Exception as e:
                console.print(f"[red]MPNet LR failed: {e}[/red]")
        
        # Normalize weights to sum to 1
        if self.weights:
            total_weight = sum(self.weights.values())
            for model_name in self.weights:
                self.weights[model_name] /= total_weight
        
        console.print(f"\n[green]✓ Ensemble trained with {len(self.models)} models[/green]")
        console.print("[yellow]Model weights:[/yellow]")
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {name}: {weight:.3f}")
        
        self.is_trained = True
        return True
    
    def predict(self, query: str) -> EnsembleResult:
        """Predict using weighted ensemble voting"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for model_name, (vectorizer_or_encoder, classifier) in self.models.items():
            try:
                if 'tfidf' in model_name:
                    # TF-IDF based models
                    X = vectorizer_or_encoder.transform([query])
                    pred = classifier.predict(X)[0]
                    prob = classifier.predict_proba(X)[0]
                elif 'transformer' in model_name or 'mpnet' in model_name:
                    # Transformer based models
                    X = vectorizer_or_encoder.encode([query])
                    
                    # Scale if needed
                    if model_name == 'transformer_lr' and 'transformer' in self.scalers:
                        X = self.scalers['transformer'].transform(X)
                    elif model_name == 'mpnet_lr' and 'mpnet' in self.scalers:
                        X = self.scalers['mpnet'].transform(X)
                    
                    pred = classifier.predict(X)[0]
                    prob = classifier.predict_proba(X)[0]
                else:
                    continue
                
                predictions[model_name] = pred
                probabilities[model_name] = prob
                
            except Exception as e:
                console.print(f"[dim]Model {model_name} failed: {e}[/dim]")
        
        if not predictions:
            return EnsembleResult(
                category="error",
                topic="error",
                confidence=0.0,
                voting_details={},
                weighted_scores={},
                method="none"
            )
        
        # Weighted voting
        label_scores = {}
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1.0 / len(predictions))
            
            if pred not in label_scores:
                label_scores[pred] = 0
            
            # Add weighted score
            if model_name in probabilities:
                # Use probability of predicted class
                prob_idx = list(classifier.classes_).index(pred) if hasattr(classifier, 'classes_') else 0
                confidence = probabilities[model_name][prob_idx] if prob_idx < len(probabilities[model_name]) else 0.5
            else:
                confidence = 1.0
            
            label_scores[pred] += weight * confidence
        
        # Get best prediction
        best_label = max(label_scores.items(), key=lambda x: x[1])[0]
        best_score = label_scores[best_label]
        
        # Determine if it's unanimous or weighted
        unique_predictions = set(predictions.values())
        if len(unique_predictions) == 1:
            method = "unanimous"
            best_score = min(1.0, best_score * 1.1)  # Boost unanimous predictions
        else:
            method = "weighted_ensemble"
        
        # Parse label
        if best_label in self.label_to_category:
            category, topic = self.label_to_category[best_label]
        else:
            if ':' in best_label:
                category, topic = best_label.split(':', 1)
            else:
                category = best_label
                topic = "unknown"
        
        return EnsembleResult(
            category=category,
            topic=topic,
            confidence=min(1.0, best_score),
            voting_details=predictions,
            weighted_scores=label_scores,
            method=method
        )


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
    method_counts = {}
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
        task = progress.add_task("[magenta]Testing weighted ensemble...", total=len(test_cases))
        
        for i, (query, expected_cat, expected_topic) in enumerate(test_cases):
            try:
                result = classifier.predict(query)
                is_correct = (result.category == expected_cat and result.topic == expected_topic)
                
                if is_correct:
                    correct += 1
                elif len(errors) < 10:
                    errors.append({
                        'query': query[:50],
                        'expected': f"{expected_cat}:{expected_topic}",
                        'predicted': f"{result.category}:{result.topic}",
                        'confidence': result.confidence,
                        'votes': len(result.voting_details)
                    })
                
                # Track method usage
                method_counts[result.method] = method_counts.get(result.method, 0) + 1
                
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
    
    console.print(f"\n[bold green]═══ Final Results ═══[/bold green]")
    console.print(f"[bold]Overall Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)[/bold]")
    console.print(f"[bold]Time Taken: {elapsed_time:.2f} seconds[/bold]")
    console.print(f"[bold]Speed: {len(test_cases)/elapsed_time:.0f} queries/second[/bold]\n")
    
    # Comparison with previous iterations
    console.print("[bold]Comparison with Previous Iterations:[/bold]")
    if accuracy > 82.5:
        console.print(f"[bold green]✓ Weighted Ensemble (v7): {accuracy:.1f}% - NEW BEST![/bold green]")
        console.print(f"[dim]  play2.C3: 82.5% (previous best)[/dim]")
        console.print(f"[bold green]🎉 BREAKTHROUGH ACHIEVED: +{accuracy - 82.5:.1f}% improvement![/bold green]")
        console.print(f"[bold green]🚀 Target of 80%+ EXCEEDED![/bold green]")
    else:
        console.print(f"[yellow]Weighted Ensemble (v7): {accuracy:.1f}%[/yellow]")
        console.print(f"[green]play2.C3: 82.5% (still best)[/green]")
        console.print(f"[dim]Need {82.5 - accuracy:.1f}% more to beat current best[/dim]")
    
    # Method usage
    if method_counts:
        console.print("\n[bold]Voting Methods:[/bold]")
        for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {method}: {count} ({100*count/len(test_cases):.1f}%)")
    
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
    if errors and accuracy < 90:
        console.print("\n[bold red]Sample Errors:[/bold red]")
        error_table = Table(box=box.SIMPLE)
        error_table.add_column("Query", style="white", width=30)
        error_table.add_column("Expected", style="green")
        error_table.add_column("Got", style="red")
        error_table.add_column("Conf", style="yellow")
        
        for err in errors[:5]:
            error_table.add_row(
                err['query'],
                err['expected'],
                err['predicted'],
                f"{err['confidence']:.1%}"
            )
        
        console.print(error_table)
    
    return accuracy


def main():
    """Main application"""
    console.print("[magenta]Initializing Advanced Weighted Ensemble Classifier...[/magenta]")
    console.print("[dim]Based on 2024-2025 research: 98.76% accuracy achievable[/dim]\n")
    
    classifier = WeightedEnsembleClassifier()
    
    if not classifier.load_data("data/training_data.yaml"):
        console.print("[red]Failed to load training data![/red]")
        sys.exit(1)
    
    console.print("\n[magenta]Training ensemble models...[/magenta]")
    if not classifier.train():
        console.print("[red]Failed to train ensemble![/red]")
        sys.exit(1)
    
    console.print("\n[green]✓ Weighted ensemble ready![/green]\n")
    
    # Run full test
    accuracy = run_full_test(classifier)
    
    return accuracy


if __name__ == "__main__":
    try:
        accuracy = main()
        # Exit with success if we beat the target
        sys.exit(0 if accuracy > 82.5 else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)