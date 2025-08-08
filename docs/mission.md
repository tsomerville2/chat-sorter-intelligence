# Mission - Query Matcher

## Vision
Build a simple, generalizable ML-based system that matches customer queries to predefined categories and topics without using LLMs or complex generative models. The system must learn from example data to recognize patterns in how customers express their needs, then accurately route their queries to the appropriate category and specific topic within that category.

The solution uses straightforward machine learning techniques (TF-IDF vectorization with a simple classifier) to prove that pattern matching can work across different domains. The system is designed to be domain-agnostic - it learns entirely from the structure and examples in the training data without any hardcoded logic about specific categories or topics.

## Success Criteria
1. Successfully train a model from YAML training data containing categories, topics, and examples
2. Achieve 80%+ accuracy matching test queries to correct category+topic pairs
3. Work with completely different category/topic sets without code changes

## In Scope
- Load hierarchical training data from YAML format
- Train simple ML classifier (scikit-learn based)
- Match new queries to category+topic pairs
- Return confidence scores with matches

## Out of Scope
- LLM or generative AI models
- Complex deep learning architectures
- Real-time retraining
- Multi-language support
- Handling queries outside training distribution

## App Name Rationale
**Chosen Name**: Query Matcher
**Reasoning**: Direct, descriptive name that clearly indicates the system's purpose - matching customer queries to predefined categories. Simple and professional for a call center context.