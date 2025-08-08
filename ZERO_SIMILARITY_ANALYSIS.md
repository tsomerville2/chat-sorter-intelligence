# Zero Similarity Analysis - Iteration 2 (Cosine Similarity)

## Critical Finding

**When cosine similarity is 0 for ALL training examples, the system defaults to the FIRST training example in the dataset**, which happens to be:
- Category: `billing`
- Topic: `payment_failed`
- Example: "My credit card was declined"

## Queries That Get Zero Similarity (But Still "Correct")

After testing the training set and edge cases, we found **3 critical failures** where queries had ZERO similarity to any training example:

### Failed Queries (Zero Similarity → Wrong Classification)

1. **"can't login"**
   - Expected: `technical_support:password_reset`
   - Got: `billing:payment_failed` (defaulted to first example)
   - Similarity: 0.000
   - Why: "can't" and "login" are not in any training example verbatim

2. **"unable to login"**
   - Expected: `technical_support:password_reset`
   - Got: `billing:payment_failed` (defaulted to first example)
   - Similarity: 0.000
   - Why: "unable" and "login" don't match training vocabulary

3. **"were is my pakage"** (with typos)
   - Expected: `shipping:track_order`
   - Got: `billing:payment_failed` (defaulted to first example)
   - Similarity: 0.000
   - Why: Typos prevent any word matching

## Why This Happens

### TF-IDF Vectorizer Settings
```python
self.vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',  # Filters out common words
    ngram_range=(1, 2)     # Uses unigrams and bigrams
)
```

### The Problem
1. **Stop words removal**: Words like "can't", "is", "my" are filtered out
2. **Vocabulary mismatch**: "login" vs "log into", "unable" not in training
3. **np.argmax() behavior**: When all similarities are 0, returns index 0

### Code Issue
```python
# In cosine_classifier.py, line 73
best_idx = np.argmax(similarities)  # Returns 0 when all values are 0
```

## Interesting Observations

### Queries That Work Despite Being Different

These queries had HIGH similarity despite not being exact matches:

| Query | Matched Example | Similarity | Result |
|-------|----------------|------------|---------|
| "forgot password" | "I forgot my password" | 0.894 | ✓ Correct |
| "reset password" | "Need to reset my password" | 0.756 | ✓ Correct |
| "track package" | "Track my shipment" | 0.596 | ✓ Correct |
| "payment failed" | "Payment didn't go through" | 0.569 | ✓ Correct |

### Single Word Queries

Surprisingly, single-word queries often work:

| Query | Similarity | Result |
|-------|------------|---------|
| "password" | 0.549 | ✓ Correct (technical_support:password_reset) |
| "refund" | 0.580 | ✓ Correct (billing:refund_request) |
| "package" | 0.447 | ✓ Correct (shipping:track_order) |

## The Default Fallback Problem

**Critical Issue**: When similarity is 0, the system doesn't indicate uncertainty—it confidently returns the first training example. This creates silent failures.

### Current Behavior
```
Query: "can't login"
→ All similarities: [0, 0, 0, ..., 0]
→ argmax([0, 0, 0, ...]) = 0
→ Returns: billing:payment_failed (confidence: 0%)
```

### Better Approach Would Be
```python
if best_similarity == 0.0:
    # Return a special "no match" result
    # Or return with extremely low confidence
    # Or raise an exception
```

## Statistics Summary

From testing 24 edge cases:
- **20/24 correct** (83.3% accuracy)
- **3 queries with zero similarity** (all incorrect)
- **Average similarity for correct**: 0.678
- **Average similarity for incorrect**: 0.088

## Key Takeaways

1. **Zero similarity = Wrong answer**: Every query with 0 similarity was misclassified
2. **Default to first**: The system always defaults to `billing:payment_failed`
3. **"login" variations fail**: Common queries like "can't login" completely fail
4. **Typos are fatal**: Any typo results in zero similarity
5. **Confidence is misleading**: 0% confidence still returns a "match"

## Recommendation

Iteration 2 should handle zero similarity as a special case:
- Return "unknown" category
- Raise an exception
- Fall back to another method (like LLM)
- At minimum, warn that the match is completely arbitrary

This analysis proves that **cosine similarity works well for close matches but fails catastrophically for novel phrasings**, even common ones like "can't login".