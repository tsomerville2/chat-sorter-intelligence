#!/usr/bin/env python3
"""
Comprehensive test runner for all query matcher variants
Tests each approach on 100 sample queries
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.append('shared')
sys.path.append('iterations/iteration1')
sys.path.append('iterations/iteration2')

from load_test_data import load_test_data_sample
import numpy as np
import yaml
from dataclasses import dataclass
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

@dataclass
class TestResult:
    name: str
    accuracy: float
    avg_confidence: float
    zero_sim_count: int
    time_taken: float
    notes: str = ""


def test_play1_original():
    """Test original ML classifier"""
    try:
        from query_classifier import QueryClassifier
        
        classifier = QueryClassifier()
        classifier.load_data("data/training_data.yaml")
        classifier.train()
        
        test_cases = load_test_data_sample(100)
        correct = 0
        conf_sum = 0
        
        start = time.time()
        for query, exp_cat, exp_topic in test_cases:
            result = classifier.predict(query)
            if result.category == exp_cat and result.topic == exp_topic:
                correct += 1
            conf_sum += result.confidence
        elapsed = time.time() - start
        
        return TestResult(
            name="play1.py (Original ML)",
            accuracy=correct/100,
            avg_confidence=conf_sum/100,
            zero_sim_count=0,
            time_taken=elapsed
        )
    except Exception as e:
        return TestResult(name="play1.py", accuracy=0, avg_confidence=0, zero_sim_count=0, time_taken=0, notes=f"Error: {e}")


def test_play1_A():
    """Test enhanced ML with custom features"""
    try:
        # Import inline to avoid conflicts
        exec(open('play1.A.py').read(), {'__name__': '__test__'})
        
        classifier = EnhancedMLClassifier()
        classifier.load_data("data/training_data.yaml")
        classifier.train()
        
        test_cases = load_test_data_sample(100)
        correct = 0
        conf_sum = 0
        
        start = time.time()
        for query, exp_cat, exp_topic in test_cases:
            result = classifier.predict(query)
            if result.category == exp_cat and result.topic == exp_topic:
                correct += 1
            conf_sum += result.confidence
        elapsed = time.time() - start
        
        return TestResult(
            name="play1.A.py (Enhanced ML)",
            accuracy=correct/100,
            avg_confidence=conf_sum/100,
            zero_sim_count=0,
            time_taken=elapsed
        )
    except Exception as e:
        return TestResult(name="play1.A.py", accuracy=0, avg_confidence=0, zero_sim_count=0, time_taken=0, notes=f"Error: {e}")


def test_play1_B_classifiers():
    """Test different classifiers from play1.B"""
    results = []
    
    try:
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Load training data
        with open('data/training_data.yaml', 'r') as f:
            data = yaml.safe_load(f)
        
        texts = []
        labels = []
        for cat_name, cat_data in data['categories'].items():
            for topic_name, topic_data in cat_data['topics'].items():
                for example in topic_data.get('examples', []):
                    texts.append(example)
                    labels.append(f'{cat_name}:{topic_name}')
        
        # Create label mappings
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        y = [label_to_idx[label] for label in labels]
        
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=1, max_df=0.95, sublinear_tf=True)
        X = vectorizer.fit_transform(texts)
        
        test_cases = load_test_data_sample(100)
        
        classifiers = [
            ("SVM", SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("Neural Net", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)),
            ("Naive Bayes", MultinomialNB(alpha=0.1))
        ]
        
        for clf_name, clf in classifiers:
            clf.fit(X, y)
            
            correct = 0
            conf_sum = 0
            
            start = time.time()
            for query, exp_cat, exp_topic in test_cases:
                X_test = vectorizer.transform([query])
                pred = clf.predict(X_test)[0]
                if hasattr(clf, 'predict_proba'):
                    prob = clf.predict_proba(X_test)[0]
                    conf = prob[pred]
                else:
                    conf = 1.0
                
                pred_label = idx_to_label[pred]
                expected = f'{exp_cat}:{exp_topic}'
                if pred_label == expected:
                    correct += 1
                conf_sum += conf
            elapsed = time.time() - start
            
            results.append(TestResult(
                name=f"play1.B ({clf_name})",
                accuracy=correct/100,
                avg_confidence=conf_sum/100,
                zero_sim_count=0,
                time_taken=elapsed
            ))
    except Exception as e:
        results.append(TestResult(name="play1.B", accuracy=0, avg_confidence=0, zero_sim_count=0, time_taken=0, notes=f"Error: {e}"))
    
    return results


def test_play2_original():
    """Test original cosine similarity"""
    try:
        from cosine_classifier import CosineClassifier
        
        classifier = CosineClassifier()
        classifier.load_data("data/training_data.yaml")
        
        test_cases = load_test_data_sample(100)
        correct = 0
        conf_sum = 0
        zero_sim = 0
        
        start = time.time()
        for query, exp_cat, exp_topic in test_cases:
            result = classifier.predict(query)
            if result.category == exp_cat and result.topic == exp_topic:
                correct += 1
            conf_sum += result.confidence
            if result.similarity_score == 0:
                zero_sim += 1
        elapsed = time.time() - start
        
        return TestResult(
            name="play2.py (Original Cosine)",
            accuracy=correct/100,
            avg_confidence=conf_sum/100,
            zero_sim_count=zero_sim,
            time_taken=elapsed
        )
    except Exception as e:
        return TestResult(name="play2.py", accuracy=0, avg_confidence=0, zero_sim_count=0, time_taken=0, notes=f"Error: {e}")


def test_play2_A():
    """Test cosine without stop words"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Load training data
        with open('data/training_data.yaml', 'r') as f:
            data = yaml.safe_load(f)
        
        texts = []
        labels = []
        for cat_name, cat_data in data['categories'].items():
            for topic_name, topic_data in cat_data['topics'].items():
                for example in topic_data.get('examples', []):
                    texts.append(example)
                    labels.append(f'{cat_name}:{topic_name}')
        
        # No stop words
        vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        train_vecs = vectorizer.fit_transform(texts)
        
        test_cases = load_test_data_sample(100)
        correct = 0
        conf_sum = 0
        zero_sim = 0
        
        start = time.time()
        for query, exp_cat, exp_topic in test_cases:
            query_vec = vectorizer.transform([query])
            similarities = (train_vecs * query_vec.T).toarray().flatten()
            best_idx = np.argmax(similarities)
            pred = labels[best_idx]
            expected = f'{exp_cat}:{exp_topic}'
            
            if pred == expected:
                correct += 1
            conf_sum += similarities[best_idx]
            if similarities[best_idx] == 0:
                zero_sim += 1
        elapsed = time.time() - start
        
        return TestResult(
            name="play2.A (No Stop Words)",
            accuracy=correct/100,
            avg_confidence=conf_sum/100,
            zero_sim_count=zero_sim,
            time_taken=elapsed
        )
    except Exception as e:
        return TestResult(name="play2.A", accuracy=0, avg_confidence=0, zero_sim_count=0, time_taken=0, notes=f"Error: {e}")


def test_play2_B_metrics():
    """Test different similarity metrics"""
    results = []
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        
        # Load training data
        with open('data/training_data.yaml', 'r') as f:
            data = yaml.safe_load(f)
        
        texts = []
        labels = []
        for cat_name, cat_data in data['categories'].items():
            for topic_name, topic_data in cat_data['topics'].items():
                for example in topic_data.get('examples', []):
                    texts.append(example)
                    labels.append(f'{cat_name}:{topic_name}')
        
        test_cases = load_test_data_sample(100)
        
        # Test different metrics
        metrics = [
            ("Jaccard", CountVectorizer(lowercase=True, binary=True, ngram_range=(1, 2))),
            ("Char N-grams", TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(2, 4))),
            ("Weighted Cosine", TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=1, sublinear_tf=True))
        ]
        
        for metric_name, vectorizer in metrics:
            train_vecs = vectorizer.fit_transform(texts)
            
            correct = 0
            conf_sum = 0
            zero_sim = 0
            
            start = time.time()
            for query, exp_cat, exp_topic in test_cases:
                query_vec = vectorizer.transform([query])
                
                if metric_name == "Jaccard":
                    # Jaccard similarity
                    similarities = []
                    for i in range(train_vecs.shape[0]):
                        intersection = np.minimum(train_vecs[i].toarray(), query_vec.toarray()).sum()
                        union = np.maximum(train_vecs[i].toarray(), query_vec.toarray()).sum()
                        sim = intersection / union if union > 0 else 0
                        similarities.append(sim)
                    similarities = np.array(similarities)
                else:
                    # Cosine similarity
                    similarities = (train_vecs * query_vec.T).toarray().flatten()
                
                best_idx = np.argmax(similarities)
                pred = labels[best_idx]
                expected = f'{exp_cat}:{exp_topic}'
                
                if pred == expected:
                    correct += 1
                conf_sum += similarities[best_idx]
                if similarities[best_idx] == 0:
                    zero_sim += 1
            elapsed = time.time() - start
            
            results.append(TestResult(
                name=f"play2.B ({metric_name})",
                accuracy=correct/100,
                avg_confidence=conf_sum/100,
                zero_sim_count=zero_sim,
                time_taken=elapsed
            ))
    except Exception as e:
        results.append(TestResult(name="play2.B", accuracy=0, avg_confidence=0, zero_sim_count=0, time_taken=0, notes=f"Error: {e}"))
    
    return results


def test_play2_C():
    """Test embeddings (simple version)"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Load training data
        with open('data/training_data.yaml', 'r') as f:
            data = yaml.safe_load(f)
        
        texts = []
        labels = []
        for cat_name, cat_data in data['categories'].items():
            for topic_name, topic_data in cat_data['topics'].items():
                for example in topic_data.get('examples', []):
                    texts.append(example)
                    labels.append(f'{cat_name}:{topic_name}')
        
        # Combined word and char features (simple embeddings)
        word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=1, sublinear_tf=True)
        char_vec = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        
        word_vecs = word_vec.fit_transform(texts)
        char_vecs = char_vec.fit_transform(texts)
        
        # Combine features
        train_embeddings = np.hstack([word_vecs.toarray(), char_vecs.toarray() * 0.5])
        
        test_cases = load_test_data_sample(100)
        correct = 0
        conf_sum = 0
        
        start = time.time()
        for query, exp_cat, exp_topic in test_cases:
            word_q = word_vec.transform([query]).toarray()
            char_q = char_vec.transform([query]).toarray()
            query_emb = np.hstack([word_q, char_q * 0.5])[0]
            
            # Cosine similarity
            similarities = []
            for train_emb in train_embeddings:
                dot = np.dot(query_emb, train_emb)
                norm1 = np.linalg.norm(query_emb)
                norm2 = np.linalg.norm(train_emb)
                sim = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
                similarities.append(sim)
            
            best_idx = np.argmax(similarities)
            pred = labels[best_idx]
            expected = f'{exp_cat}:{exp_topic}'
            
            if pred == expected:
                correct += 1
            conf_sum += similarities[best_idx]
        elapsed = time.time() - start
        
        return TestResult(
            name="play2.C (Simple Embeddings)",
            accuracy=correct/100,
            avg_confidence=conf_sum/100,
            zero_sim_count=0,
            time_taken=elapsed
        )
    except Exception as e:
        return TestResult(name="play2.C", accuracy=0, avg_confidence=0, zero_sim_count=0, time_taken=0, notes=f"Error: {e}")


def main():
    """Run all tests and display results"""
    console.print("[bold magenta]Running Comprehensive Tests on All Query Matcher Variants[/bold magenta]")
    console.print("[yellow]Testing on 100 sample queries...[/yellow]\n")
    
    all_results = []
    
    # Test play1 variants
    console.print("[cyan]Testing play1.py (Original ML)...[/cyan]")
    all_results.append(test_play1_original())
    
    console.print("[cyan]Testing play1.A.py (Enhanced ML)...[/cyan]")
    all_results.append(test_play1_A())
    
    console.print("[cyan]Testing play1.B.py (Different Classifiers)...[/cyan]")
    all_results.extend(test_play1_B_classifiers())
    
    # Test play2 variants
    console.print("[cyan]Testing play2.py (Original Cosine)...[/cyan]")
    all_results.append(test_play2_original())
    
    console.print("[cyan]Testing play2.A.py (No Stop Words)...[/cyan]")
    all_results.append(test_play2_A())
    
    console.print("[cyan]Testing play2.B.py (Different Metrics)...[/cyan]")
    all_results.extend(test_play2_B_metrics())
    
    console.print("[cyan]Testing play2.C.py (Simple Embeddings)...[/cyan]")
    all_results.append(test_play2_C())
    
    # Display results table
    console.print("\n[bold green]═══ FINAL RESULTS ═══[/bold green]\n")
    
    from rich import box
    table = Table(title="All Approaches Tested on 100 Samples", box=box.ROUNDED)
    table.add_column("Approach", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Avg Confidence", style="yellow")
    table.add_column("Zero Sim", style="red")
    table.add_column("Time (s)", style="magenta")
    table.add_column("Notes", style="dim")
    
    # Sort by accuracy
    all_results = [r for r in all_results if r is not None]
    all_results.sort(key=lambda x: x.accuracy, reverse=True)
    
    for result in all_results:
        table.add_row(
            result.name,
            f"{result.accuracy:.1%}",
            f"{result.avg_confidence:.3f}",
            str(result.zero_sim_count) if result.zero_sim_count > 0 else "-",
            f"{result.time_taken:.2f}",
            result.notes[:30] if result.notes else ""
        )
    
    console.print(table)
    
    # Find best performer
    if all_results:
        best = all_results[0]
        console.print(f"\n[bold green]🏆 Best Performer: {best.name}[/bold green]")
        console.print(f"   Accuracy: {best.accuracy:.1%}")
        console.print(f"   Zero Sim Issues: {best.zero_sim_count}")
    
    # Save results to file
    with open('test_results_100.txt', 'w') as f:
        f.write("Test Results on 100 Samples\n")
        f.write("=" * 50 + "\n\n")
        for result in all_results:
            f.write(f"{result.name}: {result.accuracy:.1%} accuracy\n")
    
    console.print("\n[dim]Results saved to test_results_100.txt[/dim]")


if __name__ == "__main__":
    main()