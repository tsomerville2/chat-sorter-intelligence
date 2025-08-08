from behave import given, when, then
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from query_classifier import QueryClassifier


@given('training data is loaded from "{filepath}"')
def step_load_training_data(context, filepath):
    context.query_classifier = QueryClassifier()
    context.query_classifier.load_data(filepath)


@given('the classifier is trained')
def step_train_classifier(context):
    context.query_classifier.train()


@when('I submit the query "{query}"')
def step_submit_query(context, query):
    context.result = context.query_classifier.predict(query)


@then('the category should be "{expected_category}"')
def step_verify_category(context, expected_category):
    assert context.result.category == expected_category, \
        f"Expected category '{expected_category}' but got '{context.result.category}'"


@then('the topic should be "{expected_topic}"')
def step_verify_topic(context, expected_topic):
    assert context.result.topic == expected_topic, \
        f"Expected topic '{expected_topic}' but got '{context.result.topic}'"


@then('confidence should be above {threshold:f}')
def step_verify_confidence(context, threshold):
    assert context.result.confidence > threshold, \
        f"Confidence {context.result.confidence} is not above {threshold}"