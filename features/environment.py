"""
BDD Environment Configuration for Query Matcher
"""

def before_all(context):
    """Setup before all tests"""
    context.config.setup_logging()
    
def before_scenario(context, scenario):
    """Setup before each scenario"""
    context.query_matcher = None
    context.results = None
    
def after_scenario(context, scenario):
    """Cleanup after each scenario"""
    pass