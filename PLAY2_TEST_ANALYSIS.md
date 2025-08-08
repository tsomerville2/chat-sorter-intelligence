# Play2.py Test Suite Analysis - Zero Similarity Failures

## Test Results from play2.py Built-in Test Suite

| Query | Expected | Got | Similarity | Result |
|-------|----------|-----|------------|--------|
| **I can't login** | technical_support:password_reset | billing:payment_failed | **0.000** | ❌ |
| my payment didn't work | billing:payment_failed | billing:payment_failed | 1.000 | ✅ |
| **where's my stuff** | shipping:track_order | billing:payment_failed | **0.000** | ❌ |
| I forgot my password | technical_support:password_reset | technical_support:password_reset | 1.000 | ✅ |
| card declined | billing:payment_failed | billing:payment_failed | 0.766 | ✅ |

**Accuracy: 3/5 (60%)**

## Critical Finding: Zero Similarity Pattern

### Both Zero-Similarity Queries Default to `billing:payment_failed`

When cosine similarity returns 0.000 for ALL training examples, the system defaults to the **first training example** which is:
- "My credit card was declined" → `billing:payment_failed`

### Why These Queries Get Zero Similarity

#### 1. "I can't login"
- **Problem**: No training example uses exactly "can't login"
- **Training has**: "Can't log into my account", "Locked out of my account"
- **TF-IDF issue**: 
  - "can't" is filtered as a stop word
  - "login" ≠ "log" (different tokens after stemming)
  - Result: NO words match any training example

#### 2. "where's my stuff"
- **Problem**: Informal language not in training
- **Training has**: "Where is my package", "Track my shipment"
- **TF-IDF issue**:
  - "where's" contraction not handled well
  - "stuff" doesn't match "package", "shipment", or "order"
  - Result: NO vocabulary overlap

## The Key Insight

**There are NO correct matches with 0 similarity!** 

Every query that achieves 0.000 similarity is **guaranteed to be wrong** because it defaults to `billing:payment_failed` regardless of the actual intent.

### Successful Matches Show Various Similarity Scores
- **1.000**: Exact match found in training ("I forgot my password")
- **1.000**: Very close match ("my payment didn't work" matches "Payment didn't go through")
- **0.766**: Partial match ("card declined" matches parts of "My credit card was declined")

### Failed Matches Always Show 0.000
- Never partially correct
- Always default to first training example
- Confidence score of 0% but still returns an answer

## Verification: What's Actually in Training Data?

Looking at the training data for password reset:
```yaml
technical_support:
  password_reset:
    examples:
      - "I forgot my password"  # Exact match for test ✅
      - "Can't log into my account"  # Should match "I can't login" but doesn't
      - "Need to reset my password"
      - "Locked out of my account"
      - "Password isn't working"
```

The phrase "Can't log into my account" should theoretically match "I can't login" but fails because:
1. Stop word removal filters "can't"
2. "login" is one word, "log into" is two words
3. TF-IDF sees no overlap

## Conclusion

**Your observation is correct**: There are **NO cases where a query gets 0 similarity but is still classified correctly**. 

The two failures ("I can't login" and "where's my stuff") both:
1. Have 0.000 similarity to ALL training examples
2. Default to the first training example (billing:payment_failed)
3. Are completely wrong

This proves that Iteration 2 (cosine similarity) has a **critical flaw**: it can't handle even simple variations in phrasing and defaults to an arbitrary wrong answer rather than admitting uncertainty.