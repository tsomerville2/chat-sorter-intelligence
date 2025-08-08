# Query Matcher ML Experiments - Results Summary

## Mission
Build a simple, generalizable ML-based system that matches customer queries to predefined categories and topics without using LLMs or complex generative models.

**Target: 80%+ accuracy on 777 test queries**

## Final Results

### 🏆 Best Performer: play2.C3.py - 82.5% accuracy ✅

Despite implementing numerous advanced techniques from 2024-2025 research, the relatively simple play2.C3 implementation remains the best performer.

## All Implementations Tested

| Implementation | Accuracy | Approach | Key Features |
|---------------|----------|----------|--------------|
| **play2.C3.py** | **82.5%** ✅ | Sentence embeddings + confidence | MiniLM-L6-v2, ambiguity detection, Pydantic models |
| play9.py | ~81%* | MPNet embeddings | Better base model, same approach as play2.C3 |
| play6.py | 79.4% | Contrastive learning | Data augmentation, ensemble voting |
| play5.py | 78.5% | SetFit few-shot | 8 samples per class, contrastive pairs |
| play8.py | 77.9% | Enhanced production | Fine-tuning, adaptive thresholds |
| play7.py | 76.2% | Weighted ensemble | Multiple models, cross-efficiency weighting |
| play4.py | 75.8% | State-of-art hybrid | SetFit + USE + cross-encoders |
| query_classifier.py | 74.1% | BDD-driven | Clean architecture, sentence transformers |
| play2.py | ~50% | TF-IDF cosine | Basic cosine similarity |
| play1.py | ~40% | Traditional ML | Naive Bayes, Random Forest |

*Note: play9 shows 81% on 100-sample test but not fully tested on 777 queries

## Techniques Attempted

### ✅ Successful Approaches
1. **Sentence Transformers** - Biggest improvement (40% → 82.5%)
2. **Confidence Thresholds** - Ambiguity detection helps accuracy
3. **Character N-grams** - Good fallback when embeddings fail
4. **MPNet Model** - Shows 3% improvement over MiniLM in testing

### ❌ Unsuccessful "Advanced" Techniques
1. **SetFit Few-Shot Learning** - Only 78.5% despite research showing 92.7%
2. **Contrastive Learning** - 79.4%, didn't improve as expected
3. **Weighted Ensemble Voting** - 76.2%, worse than single model
4. **Cross-Encoder Reranking** - Actually hurt performance
5. **Fine-Tuning** - Minimal or negative impact

## Key Insights

### Why play2.C3 Wins
1. **Simplicity** - Direct cosine similarity without overengineering
2. **Good Base Model** - MiniLM-L6-v2 is well-suited for this task
3. **Confidence Thresholds** - Returns alternatives when uncertain
4. **No Overfitting** - Doesn't try to fine-tune on limited data

### Lessons Learned
1. **More complex ≠ better** - Advanced techniques often underperformed
2. **Pre-trained models are powerful** - Fine-tuning often made things worse
3. **Data quality matters** - Some training examples are ambiguous
4. **Ensemble paradox** - Multiple weak models don't make a strong one
5. **Research vs Reality** - Published accuracies don't always transfer

## Breakthrough Attempts That Failed

Despite extensive research and implementation:
- **SetFit** promised 92.7% → achieved 78.5%
- **Weighted ensemble** promised 98.76% → achieved 76.2%
- **Contrastive learning** promised significant gains → achieved 79.4%

## Recommendations

### For Production Use
- **Use play2.C3.py** - Proven 82.5% accuracy, clean Pydantic interface
- **Consider MPNet** - Test play9.py fully for potential 3% improvement
- **Monitor ambiguous queries** - Use MultipleAnswers response type

### For Further Improvement
1. **Data Quality** - Review and clean ambiguous training examples
2. **Active Learning** - Collect real user feedback to improve
3. **Category-Specific Models** - Train specialized models per category
4. **Hybrid Approach** - Use play2.C3 with LLM fallback for low confidence

## Conclusion

After extensive experimentation with state-of-the-art techniques from 2024-2025 research, the relatively simple sentence transformer approach (play2.C3) remains the best performer at 82.5% accuracy, exceeding our 80% target.

**The key lesson: Start simple, measure everything, and don't assume complexity equals performance.**

---

*Generated through infinite BDD warp iterations with real testing on 777 queries*