Feature: Query Matching
  As a call center system
  I want to match customer queries to categories and topics
  So that queries are routed to the right department

  Background:
    Given training data is loaded from "data/training_data.yaml"
    And the classifier is trained

  Scenario: Match billing query
    When I submit the query "my card was declined yesterday"
    Then the category should be "billing"
    And the topic should be "payment_failed"
    And confidence should be above 0.15

  Scenario: Match technical support query  
    When I submit the query "I can't remember my password"
    Then the category should be "technical_support"
    And the topic should be "password_reset"
    And confidence should be above 0.15

  Scenario: Match shipping query
    When I submit the query "I need to know where my package is"
    Then the category should be "shipping"
    And the topic should be "track_order"
    And confidence should be above 0.15