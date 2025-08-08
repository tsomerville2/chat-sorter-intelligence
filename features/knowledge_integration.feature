Feature: Knowledge-Augmented Query Classification
  As a customer service system
  I want to use comprehensive knowledge bases and external context
  So that I can achieve near-human accuracy in query understanding

  Background:
    Given a knowledge graph with domain relationships
    And a context accumulator tracking conversation history
    And external knowledge sources are available
    And the system has been initialized with training data

  @knowledge_graph
  Scenario: Classify using knowledge graph relationships
    Given the query "I can't access my account after changing my email"
    When I analyze the query with knowledge graph
    Then I should identify entities: ["account", "email", "access"]
    And I should find relationships: ["email" -> "account", "access" -> "account"]
    And I should infer this is about "account_management:update_information"
    And the confidence should be at least 0.85

  @context_aware
  Scenario: Use conversation history for disambiguation
    Given the conversation history:
      | user_query                    | system_response                |
      | "my card was declined"        | "billing:payment_failed"       |
      | "I tried three times"         | "billing:payment_failed"       |
    When the user says "what's wrong with it"
    Then I should understand "it" refers to the payment card
    And I should classify as "billing:payment_failed" with context
    And the confidence should be higher than without context

  @comprehend_it
  Scenario: Deep semantic understanding with ComprehendIt
    Given a query with multiple interpretations: "charge"
    And ComprehendIt semantic analyzer is enabled
    When I analyze semantic frames:
      | Frame          | Confidence | Context Clues           |
      | billing        | 0.7        | financial transaction   |
      | technical      | 0.2        | battery/device charge   |
      | shipping       | 0.1        | rush charge/fee         |
    Then I should select the highest confidence frame
    And provide reasoning: "Financial context dominates based on training data"

  @knowledge_fusion
  Scenario: Combine multiple knowledge sources
    Given the query "can't login after payment"
    When I consult knowledge sources:
      | Source            | Classification              | Confidence |
      | Embeddings        | technical_support:password  | 0.6        |
      | Knowledge Graph   | billing:payment_failed      | 0.7        |
      | Context History   | technical_support:password  | 0.5        |
      | ComprehendIt      | billing:subscription_cancel | 0.4        |
    Then I should use weighted fusion with source reliability
    And the final classification should be "billing:payment_failed"
    And I should explain: "Payment context overrides login issue"

  @entity_recognition
  Scenario: Named entity recognition improves classification
    Given the query "PayPal isn't working"
    When I recognize entities:
      | Entity  | Type            | Category Hint |
      | PayPal  | payment_service | billing       |
    Then I should boost billing-related classifications
    And classify as "billing:payment_failed"
    And Not "technical_support:app_crash"

  @causal_reasoning
  Scenario: Use causal reasoning for complex queries
    Given the query "I was charged but didn't get my items"
    When I apply causal reasoning:
      """
      Charged -> Payment Successful
      No Items -> Delivery Failed
      Payment + No Delivery -> Shipping Issue (primary) or Refund Request (secondary)
      """
    Then I should identify the causal chain
    And classify as "shipping:delivery_problem" 
    With secondary option "billing:refund_request"

  @multi_hop_reasoning
  Scenario: Multi-hop reasoning for indirect queries
    Given the query "the thing I asked about yesterday still isn't fixed"
    And yesterday's query was "password reset not working"
    When I perform multi-hop reasoning:
      | Hop | Reasoning                        | Result                    |
      | 1   | "yesterday" -> retrieve context | "password reset"          |
      | 2   | "still not fixed" -> ongoing    | technical issue persists  |
      | 3   | password + ongoing -> classify  | technical_support:password|
    Then I should classify as "technical_support:password_reset"
    With reasoning trace available

  @knowledge_distillation
  Scenario: Learn from high-confidence predictions
    Given 100 queries classified with high confidence (>0.9)
    When I distill knowledge patterns:
      | Pattern                    | Category                  | Frequency |
      | "can't" + "login"          | technical_support:password| 95%       |
      | "charged" + "wrong"        | billing:invoice_question  | 88%       |
      | "where" + "package"        | shipping:track_order      | 92%       |
    Then I should create fast-path rules for these patterns
    And apply them before expensive model inference
    And maintain 90%+ accuracy on these patterns

  @adversarial_robustness
  Scenario: Handle adversarial or unusual queries
    Given an unusual query "my whatchamacallit is broken"
    When I detect low confidence across all models (<0.3)
    Then I should:
      | Action                | Purpose                           |
      | Request clarification | Identify the actual object/service|
      | Show top 3 categories | Let user choose                   |
      | Learn from feedback   | Improve on similar future queries |
    And mark this as ambiguous for human review

  @knowledge_graph_update
  Scenario: Dynamically update knowledge graph
    Given a new pattern emerges: "crypto payment failed"
    And this appears 10+ times in recent queries
    When I update the knowledge graph:
      """
      Add: crypto -> payment_method
      Link: crypto -> billing
      Weight: 0.8 (high correlation)
      """
    Then future "crypto" queries should favor billing classification
    And the system should adapt without retraining

  @semantic_similarity_cache
  Scenario: Cache and reuse semantic computations
    Given a query "I cannot log in to my account"
    And I've seen similar query "can't login to account"
    When I check the semantic similarity cache:
      | Cached Query        | Similarity | Classification           |
      | can't login account | 0.95       | technical_support:password|
    Then I should reuse the cached classification
    And skip expensive embedding computation
    And respond 10x faster

  @knowledge_aggregation
  Scenario Outline: Aggregate knowledge for category patterns
    Given queries in category "<category>"
    When I aggregate linguistic patterns:
      | Pattern Type | Examples                    |
      | Negations    | can't, unable, won't        |
      | Actions      | <actions>                   |
      | Objects      | <objects>                   |
    Then I should build category-specific language models
    And improve classification for that category by 5%

    Examples:
      | category           | actions            | objects                |
      | billing            | charge, pay, refund| card, invoice, payment |
      | shipping           | track, deliver, send| package, order, item  |
      | technical_support  | login, reset, fix  | password, account, app |