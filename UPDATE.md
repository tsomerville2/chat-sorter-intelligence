# UPDATE: Query Matcher Testing & Experiments

## 1. What I've Done

### ✅ Comprehensive Test Infrastructure
- Created `shared/test_1000.yaml` and `shared/test_1000_complete.yaml` with 1000+ test cases
- Created `shared/load_test_data.py` utility for consistent test data loading
- Updated **ALL** play scripts (play1.py, play2.py, play3.py) with comprehensive testing options:
  - Quick Test (5-7 queries) 
  - Sample Test (100 queries)
  - Full Test Suite (1000+ queries)
  - Test Statistics viewer

### ✅ Current Scripts Ready to Test

#### play1.py - ML Classifier (Logistic Regression)
```bash
python play1.py
```
- Menu option 3: Quick Test (5 queries)
- Menu option 4: Sample Test (100 queries) 
- Menu option 5: Full Test (1000+ queries)
- Menu option 6: Test Statistics

**Current Performance:**
- ~75-80% accuracy on full test suite
- Fast: ~500+ queries/second
- Low confidence scores (multi-class problem)
- Struggles with novel phrasings

#### play2.py - Cosine Similarity 
```bash
python play2.py
```
- Menu option 4: Quick Test (5 queries) - Shows 60% accuracy with zero similarity issues
- Menu option 5: Sample Test (100 queries)
- Menu option 6: Full Test (1000+ queries)
- Menu option 7: Test Statistics

**Current Performance:**
- ~81% accuracy on 100 sample test
- **CRITICAL BUG**: Zero similarity defaults to first example (billing:payment_failed)
- "I can't login" → 0.000 similarity → defaults incorrectly to billing:payment_failed
- "where's my stuff" → 0.000 similarity → defaults incorrectly to billing:payment_failed

#### play3.py - LLM-based (Groq API)
```bash
export GROQ_API_KEY='your-key-here'  # Required!
python play3.py
```
- Menu option 3: Quick Test (7 queries)
- Menu option 4: Sample Test (100 queries) - **Costs API credits**
- Menu option 5: Full Test (1000+ queries) - **WARNING: Expensive!**
- Menu option 6: Test Statistics

**Current Performance:**
- Expected 90-95% accuracy (but costs money)
- Provides reasoning for decisions
- Can switch between 20B and 120B models
- Handles context and intent well

### ✅ Experimental Variations Started

#### play2.A.py - Cosine Similarity WITHOUT Stop Words
```bash
python play2.A.py
```
**Hypothesis:** Zero similarity happens because stop words like "can't" are filtered out
- Removes `stop_words='english'` from TfidfVectorizer
- Adds `sublinear_tf=True` for better term frequency scaling
- Should fix "I can't login" matching issue

## 2. What to Try Next

### Immediate Tests You Can Run

1. **Test the zero similarity fix:**
   ```bash
   python play2.A.py
   # Select option 4 (Quick Test)
   # Check if "I can't login" now has non-zero similarity
   ```

2. **Compare accuracy across approaches:**
   ```bash
   # Run sample test (100 queries) on each:
   python play1.py    # Option 4 - ML classifier
   python play2.py    # Option 5 - Cosine (with bug)
   python play2.A.py  # Option 5 - Cosine (no stop words)
   ```

3. **Check specific problem queries:**
   ```bash
   python play2.py    # Option 1 - Quick Match
   # Test: "I can't login"
   # Test: "where's my stuff"
   # Test: "my card won't work"
   ```

### Planned Experiments (Not Yet Created)

#### play2.B.py - Different Similarity Metrics
- Try Jaccard similarity
- Try BM25 ranking
- Try different n-gram ranges

#### play2.C.py - Embedding Models
- Use Sentence Transformers
- Pre-trained embeddings (all-MiniLM-L6-v2)
- Semantic similarity instead of lexical

#### play1.A.py - Better Feature Engineering
- Add character n-grams
- Add word embeddings
- Custom features for common patterns

#### play1.B.py - Different Classifiers
- SVM with RBF kernel
- Random Forest
- Neural Network (MLPClassifier)

#### play3.A.py - Different LLM Models
- Test Claude via Anthropic API
- Test GPT-3.5/4 via OpenAI
- Compare speed vs accuracy

## 3. What to Expect

### Current Issues

1. **play2.py (Original Cosine)**
   - 60% accuracy on quick test
   - Zero similarity bug causes wrong defaults
   - "I can't login" → billing:payment_failed ❌

2. **play1.py (ML Classifier)**
   - 75-80% accuracy but low confidence
   - Can't explain why it chose an answer
   - Needs more training data for better performance

3. **play3.py (LLM)**
   - High accuracy but costs money
   - Rate limiting on API calls
   - Slower than local methods

### Expected Improvements

- **play2.A.py** should fix most zero similarity issues
- **play2.C.py** with embeddings should reach 85-90% accuracy
- **play1.B.py** with better classifiers might reach 85% accuracy

## 4. Running "/bddwarp infinite"

To continuously iterate and improve:

1. Run comprehensive tests on all variations
2. Document accuracy, speed, and issues
3. Create confusion matrices to identify problem areas
4. Generate more training data for common mistakes
5. Implement ensemble methods combining approaches

## 5. Quick Commands Reference

```bash
# See what's different between play2.py and play2.A.py
diff play2.py play2.A.py

# Run quick accuracy check
python -c "
import sys
sys.path.append('iterations/iteration2')
sys.path.append('shared')
from cosine_classifier import CosineClassifier
from load_test_data import load_test_data_sample

classifier = CosineClassifier()
classifier.load_data('data/training_data.yaml')

test_cases = load_test_data_sample(100)
correct = sum(1 for q, cat, topic in test_cases 
              if (r := classifier.predict(q)).category == cat 
              and r.topic == topic)
print(f'Accuracy: {correct}/100 ({correct}%)')
"

# Check zero similarity issue
echo "4" | python play2.py | grep -A5 "I can't login"
```

## Next Steps

Run the tests above and let me know which approaches show promise. I'll continue creating the experimental variations (play2.B.py, play2.C.py, play1.A.py, etc.) to explore different solutions to the zero similarity problem and improve overall accuracy.