"""
Conversation Generator Module

Pure function for generating conversations using LLM API.
"""

import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI


class GenerationError(Exception):
    """Raised when conversation generation fails."""
    pass


class ConversationGenerator:
    """Generates conversations using Nebius Token Factory API."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    ):
        """
        Initialize the conversation generator.
        
        Args:
            api_key: Nebius API key
            model_name: Name of the model to use
            base_url: API base URL
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate a single conversation from a prompt.
        
        This is a pure function that takes a prompt and returns raw JSON.
        Raises GenerationError on failure after all retries.
        
        Args:
            prompt: The conversation generation prompt
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        
        Returns:
            Raw JSON dictionary from the model
        
        Raises:
            GenerationError: If generation fails after all retries
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Call API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You generate Arabic conversations in Saudi Najdi Dialect. Output ONLY JSON, no markdown."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=2000,
                    top_p=0.9
                )
                
                # Extract response text
                response_text = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith('```'):
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    else:
                        response_text = response_text.split('```')[1].split('```')[0].strip()
                
                # Parse JSON
                raw_json = json.loads(response_text)
                
                # Success - return raw JSON
                return raw_json
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing failed: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue
                
            except Exception as e:
                last_error = f"API error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                continue
        
        # All retries failed
        raise GenerationError(
            f"Failed to generate conversation after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
