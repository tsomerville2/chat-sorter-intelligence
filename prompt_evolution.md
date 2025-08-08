# Prompt Evolution - Iteration 1

## Improvements for Next Time
- Instead of: "Expecting 0.5 confidence threshold for multi-class problems"
- Try: "Calculate realistic threshold as 3x random chance (3/num_classes)"

## Effective Patterns
- "class_weight='balanced'" works well for small training datasets
- TF-IDF with bigrams captures query patterns effectively
- Simple models (LogisticRegression) often outperform complex ones (SVM)

## Self-Notes
- Remember to test the user interface early in the process
- Consider data augmentation techniques for small datasets
- Always check if predictions are correct before worrying about confidence scores
- Rich library makes CLI apps look professional with minimal code

## Lessons Learned
- Multi-class classification naturally has lower confidence scores
- The goal is correct classification, not high confidence
- Simple ML techniques (TF-IDF + LogisticRegression) are powerful for text classification
- BDD helps catch unrealistic expectations early

## Architecture Decisions That Worked
- Separating DataManager, MLClassifier, and QueryClassifier
- Using dataclasses for MatchResult
- Making the system completely data-driven (no hardcoded categories)