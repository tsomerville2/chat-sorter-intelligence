# BDD Project Initialized - Query Matcher

## Generated Structure
- ✅ BDD framework configured (behave)
- ✅ Domain model defined (docs/ddd.md)
- ✅ State flow mapped (docs/state-diagram.md)
- ✅ Mission clarified (docs/mission.md)
- ✅ Features created (features/query_matching.feature)
- ✅ Architecture planned (pseudocode/*.pseudo)
- ✅ Training data synthesized (data/training_data.yaml)

## Quick Start
1. Review the generated documents in docs/
2. Examine the features/ directory for BDD scenarios
3. Check pseudocode/ for the planned architecture
4. View training data in data/training_data.yaml

## Training Data Structure
The YAML file contains:
- 4 main categories (billing, technical_support, shipping, account_management)
- 4 topics per category (16 total topics)
- 5 example queries per topic (80 total training examples)

## Next Steps
Run the bddloop command to:
- Generate step definitions from features
- Implement the pseudocode as Python code
- Create the ML classifier using scikit-learn
- Make all BDD tests pass

## Configuration
- Tech Stack: Python with scikit-learn
- BDD Framework: behave
- ML Approach: TF-IDF vectorization + Naive Bayes classifier
- App Goal: "Simple ML-based query matching to categories and topics"

## Key Design Decisions
- **No hardcoded logic**: The classifier learns entirely from YAML data
- **Simple ML pipeline**: TF-IDF for text vectorization, Naive Bayes for classification
- **Hierarchical structure**: Category -> Topic -> Examples
- **Generalizable**: Will work with any YAML file following the same structure