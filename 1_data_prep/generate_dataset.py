"""
Arabic EOU Dataset Generator

Generates Arabic conversational dataset with End-of-Utterance labels.
Simplified version combining all HAMS functionality into one script.

Usage:
    python generate_dataset.py --num-samples 10000 --output dataset.csv
"""

import csv
import io
import logging
import argparse
import os
import yaml
from typing import List, Dict, Any
from openai import OpenAI
import time


# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================================
# Prompt Builder
# ============================================================================

class PromptBuilder:
    """Builds prompts for conversation generation."""
    
    def __init__(self, prompts_file="prompts.yaml"):
        """Load prompts from YAML file."""
        with open(prompts_file, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
    
    def get_system_message(self) -> str:
        """Get system message for LLM."""
        return self.prompts.get('system_message', '')
    
    def get_all_csv_prompts(self, target_samples_per_domain: int = 50) -> List[str]:
        """Get all domain-specific prompts."""
        prompts = []
        domains = self.prompts.get('domains', {})
        
        for domain_name, domain_config in domains.items():
            prompt = f"""Generate {target_samples_per_domain} Arabic utterances for domain: {domain_name}

Description: {domain_config.get('description', '')}

Examples:
{self._format_examples(domain_config.get('examples', []))}

Requirements:
- Generate EXACTLY {target_samples_per_domain} samples
- Output as CSV: utterance,style,label
- Style: informal/formal/asr_like
- Label: 0 (incomplete) or 1 (complete)
- Mix of complete and incomplete utterances
- Saudi dialect emphasis
- Natural conversational Arabic

Output CSV only, no headers, no explanations."""
            
            prompts.append(prompt)
        
        return prompts
    
    def _format_examples(self, examples: List[Dict]) -> str:
        """Format examples for prompt."""
        formatted = []
        for ex in examples:
            formatted.append(f"- {ex.get('utterance', '')} (Label: {ex.get('label', 0)})")
        return '\n'.join(formatted)


# ============================================================================
# Dataset Generator
# ============================================================================

class DatasetGenerator:
    """Generates Arabic EOU dataset using LLM."""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        """
        Initialize generator.
        
        Args:
            api_key: OpenAI API key (or from env OPENAI_API_KEY)
            model_name: Model to use
            temperature: Sampling temperature
            max_retries: Max retries on failure
        """
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key parameter.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_builder = PromptBuilder()
        
        logging.info(f"Initialized with model: {model_name}")
    
    def generate_samples(self, prompt: str, target_samples: int = 50) -> List[Dict[str, Any]]:
        """
        Generate samples using LLM.
        
        Args:
            prompt: Generation prompt
            target_samples: Target number of samples
        
        Returns:
            List of dicts with keys: utterance, style, label
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # Call LLM
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.prompt_builder.get_system_message()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=4096
                )
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                samples = self._parse_csv(response_text)
                
                if len(samples) == 0:
                    raise ValueError("No samples generated")
                
                logging.info(f"Generated {len(samples)} samples")
                return samples
                
            except Exception as e:
                logging.warning(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(attempt)  # Exponential backoff
                else:
                    logging.error(f"Failed after {self.max_retries} attempts")
                    return []
        
        return []
    
    def _parse_csv(self, text: str) -> List[Dict[str, Any]]:
        """Parse CSV response from LLM."""
        # Remove markdown code blocks
        text = text.replace("```csv", "").replace("```", "").strip()
        
        samples = []
        reader = csv.reader(io.StringIO(text))
        
        for row in reader:
            if len(row) != 3:
                continue
            
            utterance, style, label = row
            
            # Validate
            if style not in ["informal", "formal", "asr_like"]:
                style = "informal"
            
            try:
                label = int(label)
                if label not in [0, 1]:
                    continue
            except ValueError:
                continue
            
            samples.append({
                "utterance": utterance.strip(),
                "style": style.strip(),
                "label": label
            })
        
        return samples
    
    def generate_dataset(
        self,
        num_samples: int,
        samples_per_call: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Generate complete dataset.
        
        Args:
            num_samples: Total samples to generate
            samples_per_call: Samples per API call
        
        Returns:
            List of sample dicts
        """
        all_samples = []
        prompts = self.prompt_builder.get_all_csv_prompts(samples_per_call)
        
        # Calculate number of rounds
        num_rounds = (num_samples + len(prompts) * samples_per_call - 1) // (len(prompts) * samples_per_call)
        
        logging.info(f"Generating {num_samples} samples across {num_rounds} rounds...")
        
        for round_idx in range(num_rounds):
            for prompt_idx, prompt in enumerate(prompts):
                if len(all_samples) >= num_samples:
                    break
                
                logging.info(f"Round {round_idx+1}/{num_rounds}, Prompt {prompt_idx+1}/{len(prompts)}")
                
                samples = self.generate_samples(prompt, samples_per_call)
                all_samples.extend(samples)
                
                logging.info(f"Total samples: {len(all_samples)}/{num_samples}")
            
            if len(all_samples) >= num_samples:
                break
        
        return all_samples[:num_samples]
    
    def save_to_csv(self, samples: List[Dict[str, Any]], output_file: str):
        """Save samples to CSV file."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['utterance', 'style', 'label'])
            writer.writeheader()
            writer.writerows(samples)
        
        logging.info(f"Saved {len(samples)} samples to {output_file}")
    
    def split_dataset(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split dataset into train/val/test.
        
        Args:
            samples: List of samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        
        Returns:
            Dict with keys: train, val, test
        """
        import random
        random.shuffle(samples)
        
        total = len(samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        return {
            'train': samples[:train_end],
            'val': samples[train_end:val_end],
            'test': samples[val_end:]
        }


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Arabic EOU dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10,000 samples
  python generate_dataset.py --num-samples 10000 --output dataset.csv
  
  # Generate with custom model
  python generate_dataset.py --num-samples 5000 --model gpt-4 --output data.csv
  
  # Generate and split into train/val/test
  python generate_dataset.py --num-samples 10000 --split --output-dir ./data
        """
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of samples to generate (default: 10000)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='dataset.csv',
        help='Output CSV file (default: dataset.csv)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4.1-mini',
        help='LLM model to use (default: gpt-4.1-mini)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    
    parser.add_argument(
        '--samples-per-call',
        type=int,
        default=50,
        help='Samples per API call (default: 50)'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split dataset into train/val/test'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for split datasets (default: ./data)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DatasetGenerator(
        api_key=args.api_key,
        model_name=args.model,
        temperature=args.temperature
    )
    
    # Generate dataset
    logging.info(f"Generating {args.num_samples} samples...")
    samples = generator.generate_dataset(
        num_samples=args.num_samples,
        samples_per_call=args.samples_per_call
    )
    
    # Save or split
    if args.split:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Split dataset
        splits = generator.split_dataset(samples)
        
        # Save splits
        for split_name, split_samples in splits.items():
            output_file = os.path.join(args.output_dir, f'{split_name}.csv')
            generator.save_to_csv(split_samples, output_file)
        
        logging.info(f"Dataset split and saved to {args.output_dir}/")
        logging.info(f"  Train: {len(splits['train'])} samples")
        logging.info(f"  Val: {len(splits['val'])} samples")
        logging.info(f"  Test: {len(splits['test'])} samples")
    else:
        # Save single file
        generator.save_to_csv(samples, args.output)
    
    logging.info("Done!")


if __name__ == '__main__':
    main()
