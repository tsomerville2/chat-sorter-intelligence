"""
Iteration 3: LLM-based Classification
Uses Groq API with OpenAI OSS models for intelligent classification
"""

import os
import json
import yaml
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from groq import Groq
from pydantic import BaseModel, Field


# Pydantic models for structured output
class ClassificationResult(BaseModel):
    category: str = Field(description="The matched category from the available options")
    topic: str = Field(description="The specific topic within the category")
    confidence: float = Field(description="Confidence score from 0 to 1")
    reasoning: str = Field(description="Brief explanation of why this match was chosen")
    alternative_category: Optional[str] = Field(None, description="Second best category if applicable")
    alternative_topic: Optional[str] = Field(None, description="Second best topic if applicable")


@dataclass
class LLMResult:
    category: str
    topic: str
    confidence: float
    reasoning: str
    alternative: Optional[Tuple[str, str]]
    model_used: str


class LLMClassifier:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        """
        Initialize LLM classifier
        Args:
            model_name: Either "openai/gpt-oss-20b" or "openai/gpt-oss-120b"
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.categories = {}
        self.categories_json = ""
        
    def load_data(self, filepath: str) -> bool:
        """Load categories and topics from YAML"""
        try:
            with open(filepath, 'r') as file:
                yaml_data = yaml.safe_load(file)
                self.categories = yaml_data.get('categories', {})
                
                # Create a simplified structure for the LLM
                simplified = {}
                for cat_name, cat_data in self.categories.items():
                    topics = list(cat_data.get('topics', {}).keys())
                    simplified[cat_name] = topics
                
                self.categories_json = json.dumps(simplified, indent=2)
                return True
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def predict(self, query: str) -> LLMResult:
        """Classify query using LLM"""
        if not self.categories:
            raise ValueError("No categories loaded")
        
        # Create the prompt
        system_prompt = """You are a customer query classification system. Given a customer query and a list of available categories and topics, you must classify the query into the most appropriate category and topic.

Available categories and topics:
{categories}

Respond with a JSON object containing:
- category: The matched category (must be from the available list)
- topic: The specific topic within that category (must be from the available list)
- confidence: A confidence score from 0 to 1
- reasoning: Brief explanation (max 50 words)
- alternative_category: Second best category if there's ambiguity (optional)
- alternative_topic: Second best topic if there's ambiguity (optional)

Be strict about only using categories and topics from the provided list.""".format(
            categories=self.categories_json
        )
        
        user_prompt = f"Classify this customer query: \"{query}\""
        
        try:
            # Make the API call with structured output
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_object"
                },
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500
            )
            
            # Parse the response
            response_text = completion.choices[0].message.content
            result_data = json.loads(response_text)
            
            # Extract alternative if provided
            alternative = None
            if result_data.get('alternative_category') and result_data.get('alternative_topic'):
                alternative = (result_data['alternative_category'], result_data['alternative_topic'])
            
            return LLMResult(
                category=result_data.get('category', ''),
                topic=result_data.get('topic', ''),
                confidence=float(result_data.get('confidence', 0.0)),
                reasoning=result_data.get('reasoning', ''),
                alternative=alternative,
                model_used=self.model_name
            )
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Return a fallback result
            return LLMResult(
                category="error",
                topic="error",
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                alternative=None,
                model_used=self.model_name
            )
    
    def compare_models(self, query: str) -> Dict:
        """Compare results from both 20B and 120B models"""
        results = {}
        
        # Test with 20B model
        self.model_name = "openai/gpt-oss-20b"
        results['20b'] = self.predict(query)
        
        # Test with 120B model
        self.model_name = "openai/gpt-oss-120b"
        results['120b'] = self.predict(query)
        
        return results