# Domain Model - Query Matcher

## Bounded Context
Customer Query Classification System

## Aggregates

### Query Classifier
- **Root Entity**: Classifier
- **Value Objects**: CustomerQuery, MatchResult
- **Business Rules**: 
  - Every query must map to exactly one category+topic pair
  - Confidence threshold must be met for valid match
  - Training data must contain at least 3 examples per topic

### Training Data
- **Root Entity**: TrainingSet
- **Value Objects**: Category, Topic, Example
- **Business Rules**:
  - Categories must have unique names
  - Topics must be unique within their category
  - Examples must be non-empty strings

## Domain Events
1. **QueryReceived** - Customer submits a query
2. **ClassifierTrained** - ML model trained on data
3. **MatchFound** - Query matched to category+topic
4. **MatchFailed** - No confident match found

## Ubiquitous Language
- **Category**: High-level classification grouping (e.g., "Billing", "Technical Support")
- **Topic**: Specific subject within a category (e.g., "Payment Failed", "Router Issues")
- **Query**: Customer's natural language input
- **Match**: Successful mapping of query to category+topic
- **Confidence**: Probability score of match accuracy
- **Training Set**: Collection of example queries with known category+topic labels