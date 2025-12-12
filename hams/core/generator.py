"""
Conversation Generator Module

Generates conversations using LLM API and parses CSV output.
"""

import csv
import io
import logging
import time
from typing import List, Dict, Any
from openai import OpenAI
from .prompt_builder import EOUAwarePromptBuilder


class GenerationError(Exception):
    """Raised when conversation generation fails."""
    pass


class ConversationGenerator:
    """Generates conversations using LLM API with CSV output."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        base_url: str = "https://api.tokenfactory.nebius.com/v1/",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the conversation generator.
        
        Args:
            api_key: API key for the LLM service
            model_name: Name of the model to use
            base_url: Base URL for the API
            temperature: Sampling temperature
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prompt_builder = EOUAwarePromptBuilder()
        self.system_message = self.prompt_builder.get_system_message()
        
        logging.info(f"Initialized model: {model_name}")
    
    def generate_csv_samples(
        self,
        prompt: str,
        target_samples: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate CSV samples using the LLM.
        
        Args:
            prompt: The prompt to use for generation
            target_samples: Target number of samples to generate
        
        Returns:
            List of dictionaries with keys: utterance, style, label
        
        Raises:
            GenerationError: If generation fails after all retries
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # Call API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_message
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=4096
                )
                
                # Extract response text
                response_text = response.choices[0].message.content.strip()
                
                # Parse CSV
                samples = self._parse_csv_response(response_text)
                
                if len(samples) == 0:
                    raise GenerationError("No samples generated")
                
                logging.info(f"Generated {len(samples)} samples")
                return samples
                
            except Exception as e:
                logging.warning(f"Attempt {attempt}/{self.max_retries}: Error - {str(e)}")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                else:
                    raise GenerationError(f"Failed after {self.max_retries} attempts: {str(e)}")
        
        raise GenerationError("Generation failed")
    
    def _parse_csv_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse CSV response from LLM.
        
        Args:
            response_text: Raw response text from LLM
        
        Returns:
            List of dictionaries with keys: utterance, style, label
        """
        # Remove markdown code blocks if present
        response_text = response_text.replace("```csv", "").replace("```", "").strip()
        
        # Parse CSV
        samples = []
        csv_reader = csv.reader(io.StringIO(response_text))
        
        for row in csv_reader:
            if len(row) != 3:
                logging.warning(f"Skipping invalid row: {row}")
                continue
            
            utterance, style, label = row
            
            # Validate style
            if style not in ["informal", "formal", "asr_like"]:
                logging.warning(f"Invalid style '{style}', defaulting to 'informal'")
                style = "informal"
            
            # Validate label
            try:
                label_int = int(label)
                if label_int not in [0, 1]:
                    raise ValueError
            except ValueError:
                logging.warning(f"Invalid label '{label}', skipping row")
                continue
            
            samples.append({
                "utterance": utterance.strip(),
                "style": style.strip(),
                "label": label_int
            })
        
        return samples
    
    def generate_batch(
        self,
        num_samples: int,
        samples_per_call: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of samples across all domains.
        
        Args:
            num_samples: Total number of samples to generate
            samples_per_call: Number of samples to request per API call
        
        Returns:
            List of sample dictionaries
        """
        all_samples = []
        prompts = self.prompt_builder.get_all_csv_prompts(
            target_samples_per_domain=samples_per_call
        )
        
        num_calls = (num_samples + len(prompts) * samples_per_call - 1) // (len(prompts) * samples_per_call)
        
        for call_idx in range(num_calls):
            for prompt_idx, prompt in enumerate(prompts):
                if len(all_samples) >= num_samples:
                    break
                
                logging.info(f"Generating batch {call_idx+1}/{num_calls}, prompt {prompt_idx+1}/{len(prompts)}")
                
                try:
                    samples = self.generate_csv_samples(prompt, samples_per_call)
                    all_samples.extend(samples)
                    logging.info(f"Total samples so far: {len(all_samples)}")
                except GenerationError as e:
                    logging.error(f"Failed to generate samples: {e}")
                    continue
            
            if len(all_samples) >= num_samples:
                break
        
        return all_samples[:num_samples]
