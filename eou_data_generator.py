#!/usr/bin/env python3
"""
Arabic EOU Synthetic Data Generator - Final Production Version

This script generates synthetic Arabic conversational data for End-of-Utterance (EOU) detection
using Nebius Token Factory API with the Qwen model.

Features:
- Generates realistic Arabic conversations with EOU labels
- Handles multiple JSON output formats from the model
- Normalizes output to consistent format
- Validates and cleans data
- Production-ready error handling

Author: Manus AI
Date: December 2025
"""

import json
import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from openai import OpenAI


# Fix Windows console encoding
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("eou_data_generation_final.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Simplified prompts for better consistency
SIMPLIFIED_PROMPTS = [
    """Generate a realistic Arabic conversation between a customer and a restaurant receptionist in Saudi Arabic dialect.

Customer wants to make a dinner reservation. Include natural hesitations and filler words.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "rest_001",
  "topic": "restaurant_reservation",
  "dialogue": [
    {"speaker": "customer", "text": "السلام عليكم، هل عندكم طاولة متاحة؟", "is_eou": true},
    {"speaker": "receptionist", "text": "وعليكم السلام، نعم متاح. كم عدد الأشخاص؟", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between a customer and a bank representative in Saudi Arabic dialect.

Customer inquires about account balance and recent transactions.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "bank_001",
  "topic": "bank_inquiry",
  "dialogue": [
    {"speaker": "customer", "text": "السلام عليكم، أبغى أتعرف على رصيدي", "is_eou": true},
    {"speaker": "representative", "text": "وعليكم السلام، رقم الحساب من فضلك؟", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between a traveler and a hotel receptionist in Saudi Arabic dialect.

Traveler wants to book a room for 3 nights with special requests.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "hotel_001",
  "topic": "hotel_booking",
  "dialogue": [
    {"speaker": "traveler", "text": "السلام عليكم، أبغى أحجز غرفة لثلاث ليالي", "is_eou": true},
    {"speaker": "receptionist", "text": "وعليكم السلام، متى الوصول؟", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between a patient and a clinic receptionist in Saudi Arabic dialect.

Patient calls to schedule a doctor's appointment and describes symptoms.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "clinic_001",
  "topic": "medical_appointment",
  "dialogue": [
    {"speaker": "patient", "text": "السلام عليكم، أبغى أحجز موعد عند الدكتور", "is_eou": true},
    {"speaker": "receptionist", "text": "وعليكم السلام، متى تفضل؟", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between two friends catching up in Saudi Arabic dialect.

Friends haven't seen each other in months. Include excitement and natural flow.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "friends_001",
  "topic": "catching_up",
  "dialogue": [
    {"speaker": "friend1", "text": "يا إلهي! كيفك أنت؟ شنو أخبارك؟", "is_eou": true},
    {"speaker": "friend2", "text": "الحمد لله، تمام التمام. وأنت كيفك؟", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between family members at dinner in Saudi Arabic dialect.

Family discussing daily events and plans. Include natural banter and opinions.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "family_001",
  "topic": "family_dinner",
  "dialogue": [
    {"speaker": "mother", "text": "كيفك في الشغل اليوم؟", "is_eou": true},
    {"speaker": "father", "text": "الحمد لله، يوم عادي. والأطفال؟", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between a customer and a vendor at a market in Saudi Arabic dialect.

Customer browsing and asking about products. Include price negotiation.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "market_001",
  "topic": "market_shopping",
  "dialogue": [
    {"speaker": "customer", "text": "السلام عليكم، كم سعر هذا؟", "is_eou": true},
    {"speaker": "vendor", "text": "وعليكم السلام، 50 ريال", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between a customer and a car rental agent in Saudi Arabic dialect.

Customer renting a car for a trip. Include insurance and price negotiation.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "rental_001",
  "topic": "car_rental",
  "dialogue": [
    {"speaker": "customer", "text": "السلام عليكم، أبغى أستأجر سيارة", "is_eou": true},
    {"speaker": "agent", "text": "وعليكم السلام، نوع السيارة اللي تفضل؟", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between an interviewer and a job candidate in Saudi Arabic dialect.

Job interview for a professional position. Include questions about experience.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "interview_001",
  "topic": "job_interview",
  "dialogue": [
    {"speaker": "interviewer", "text": "أهلا وسهلا، كيفك؟ تفضل، اجلس", "is_eou": true},
    {"speaker": "candidate", "text": "شكراً، الحمد لله، أنا بخير", "is_eou": true}
  ]
}""",
    """Generate a realistic Arabic conversation between a customer and a restaurant server in Saudi Arabic dialect.

Customer ordering food at a restaurant. Include menu inquiry and recommendations.

Return ONLY valid JSON with this structure:
{
  "conversation_id": "order_001",
  "topic": "restaurant_order",
  "dialogue": [
    {"speaker": "customer", "text": "السلام عليكم، ايش تنصح؟", "is_eou": true},
    {"speaker": "server", "text": "وعليكم السلام، الكبسة حلوة جداً", "is_eou": true}
  ]
}""",
]


class ArabicEOUDataGeneratorFinal:
    """
    Generates synthetic Arabic conversational data using Nebius Token Factory API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ):
        """
        Initialize the data generator with Nebius Token Factory API.

        Args:
            api_key: Nebius API key. If None, uses NEBIUS_API_KEY environment variable.
            model_name: Name of the model to use.
        """
        if api_key is None:
            api_key = os.getenv("NEBIUS_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key not provided and NEBIUS_API_KEY environment variable not set. "
                    "Please set your API key: export NEBIUS_API_KEY='your-api-key'"
                )

        # Initialize OpenAI client pointing to Nebius Token Factory endpoint
        self.client = OpenAI(
            api_key=api_key, base_url="https://api.tokenfactory.nebius.com/v1/"
        )
        self.model_name = model_name
        logger.info(f"Initialized Nebius Token Factory model: {model_name}")

    def _normalize_conversation(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize conversation to standard format.

        Handles multiple input formats and converts to standard format with 'turns'.

        Args:
            data: Raw conversation data from model

        Returns:
            Normalized conversation or None if invalid
        """
        if not isinstance(data, dict):
            return None

        # Already in correct format with 'turns'
        if "turns" in data and isinstance(data["turns"], list):
            return data

        # Convert 'dialogue' format to 'turns' format
        if "dialogue" in data and isinstance(data["dialogue"], list):
            turns = []
            for idx, turn in enumerate(data["dialogue"], 1):
                if isinstance(turn, dict) and "speaker" in turn and "text" in turn:
                    normalized_turn = {
                        "turn_id": idx,
                        "speaker": turn["speaker"],
                        "text": turn["text"],
                        "is_eou": turn.get("is_eou", True),
                        "turn_type": turn.get("turn_type", "dialogue"),
                        "has_continuation_signals": turn.get(
                            "has_continuation_signals", False
                        ),
                    }
                    turns.append(normalized_turn)

            if turns:
                data["turns"] = turns
                del data["dialogue"]
                return data

        return None

    def generate_conversation(
        self, prompt: str, temperature: float = 0.7, max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single conversation using Nebius Token Factory API.

        Args:
            prompt: The conversation generation prompt.
            temperature: Creativity level (0.0-1.0).
            max_retries: Maximum number of retry attempts.

        Returns:
            Normalized conversation dictionary or None if generation fails.
        """
        for attempt in range(max_retries):
            try:
                # Call Nebius Token Factory API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in generating realistic Arabic conversational data. Always return ONLY valid JSON, with no markdown formatting or code blocks.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=2000,
                    top_p=0.9,
                )

                # Extract response text
                response_text = response.choices[0].message.content.strip()

                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    if "```json" in response_text:
                        response_text = (
                            response_text.split("```json")[1].split("```")[0].strip()
                        )
                    else:
                        response_text = (
                            response_text.split("```")[1].split("```")[0].strip()
                        )

                # Parse JSON
                conversation_data = json.loads(response_text)

                # Normalize to standard format
                normalized = self._normalize_conversation(conversation_data)

                if normalized:
                    logger.debug(
                        f"Successfully generated conversation: {normalized.get('conversation_id')}"
                    )
                    return normalized
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries}: Failed to normalize conversation format"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    continue

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries}: JSON parsing failed - {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Error - {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue

        logger.error(f"Failed to generate conversation after {max_retries} attempts")
        return None

    def generate_batch(
        self,
        prompts: List[str],
        output_file: str,
        temperature: float = 0.7,
        delay_between_requests: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate a batch of conversations and save to file.

        Args:
            prompts: List of conversation generation prompts.
            output_file: Path to save the generated conversations.
            temperature: Creativity level.
            delay_between_requests: Delay between API requests in seconds.

        Returns:
            Dictionary with generation statistics.
        """
        conversations = []
        stats = {
            "total_prompts": len(prompts),
            "successful_generations": 0,
            "failed_generations": 0,
            "total_turns": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": 0,
        }

        logger.info(
            f"Starting batch generation with {len(prompts)} prompts using Nebius Token Factory"
        )

        for idx, prompt in enumerate(prompts, 1):
            logger.info(f"Generating conversation {idx}/{len(prompts)}")

            conversation = self.generate_conversation(prompt, temperature=temperature)

            if conversation:
                conversations.append(conversation)
                stats["successful_generations"] += 1

                # Count turns
                if "turns" in conversation:
                    stats["total_turns"] += len(conversation["turns"])

                logger.info(f"[OK] Successfully generated conversation {idx}")
            else:
                stats["failed_generations"] += 1
                logger.warning(f"[FAILED] Failed to generate conversation {idx}")

            # Delay between requests
            if idx < len(prompts):
                time.sleep(delay_between_requests)

        stats["end_time"] = datetime.now().isoformat()

        # Save conversations
        self._save_conversations(conversations, output_file, stats)

        logger.info(
            f"Batch generation complete: {stats['successful_generations']}/{len(prompts)} successful"
        )
        return stats

    def _save_conversations(
        self,
        conversations: List[Dict[str, Any]],
        output_file: str,
        stats: Dict[str, Any],
    ) -> None:
        """
        Save conversations to a JSON file.

        Args:
            conversations: List of conversation dictionaries.
            output_file: Path to save the file.
            stats: Generation statistics.
        """
        output_data = {
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "model": self.model_name,
                "provider": "nebius_token_factory",
                "total_conversations": len(conversations),
                "total_turns": stats["total_turns"],
                "generation_stats": stats,
            },
            "conversations": conversations,
        }

        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(conversations)} conversations to {output_file}")

    def validate_conversations(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate the quality of generated conversations.

        Args:
            conversations: List of conversation dictionaries.

        Returns:
            Validation report dictionary.
        """
        report = {
            "total_conversations": len(conversations),
            "valid_conversations": 0,
            "invalid_conversations": 0,
            "issues": [],
            "statistics": {
                "avg_turns_per_conversation": 0,
                "min_turns": float("inf"),
                "max_turns": 0,
                "conversations_with_eou_labels": 0,
                "avg_eou_ratio": 0,
            },
        }

        total_turns = 0
        total_eou_turns = 0

        for idx, conv in enumerate(conversations):
            is_valid = True

            # Check required fields
            if "turns" not in conv:
                report["issues"].append(f"Conversation {idx}: Missing 'turns' field")
                is_valid = False
            else:
                turns = conv["turns"]
                total_turns += len(turns)

                # Update statistics
                if len(turns) < report["statistics"]["min_turns"]:
                    report["statistics"]["min_turns"] = len(turns)
                if len(turns) > report["statistics"]["max_turns"]:
                    report["statistics"]["max_turns"] = len(turns)

                # Check each turn
                for turn_idx, turn in enumerate(turns):
                    required_fields = ["turn_id", "speaker", "text", "is_eou"]
                    for field in required_fields:
                        if field not in turn:
                            report["issues"].append(
                                f"Conversation {idx}, Turn {turn_idx}: Missing '{field}' field"
                            )
                            is_valid = False

                    if turn.get("is_eou"):
                        total_eou_turns += 1

                # Count conversations with EOU labels
                if any(turn.get("is_eou") for turn in turns):
                    report["statistics"]["conversations_with_eou_labels"] += 1

            if is_valid:
                report["valid_conversations"] += 1
            else:
                report["invalid_conversations"] += 1

        # Calculate averages
        if len(conversations) > 0:
            report["statistics"]["avg_turns_per_conversation"] = total_turns / len(
                conversations
            )

        if total_turns > 0:
            report["statistics"]["avg_eou_ratio"] = total_eou_turns / total_turns

        if report["statistics"]["min_turns"] == float("inf"):
            report["statistics"]["min_turns"] = 0

        return report


def main():
    """Main function to run the data generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Arabic conversational data for EOU detection using Nebius Token Factory"
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=10,
        help="Number of conversations to generate (default: 10)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="synthetic_conversations.json",
        help="Output file path (default: synthetic_conversations.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Model temperature for creativity (0.0-1.0, default: 0.7)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
        help="Nebius model to use (default: Qwen/Qwen3-235B-A22B-Instruct-2507)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate generated conversations"
    )

    args = parser.parse_args()

    try:
        # Initialize generator
        generator = ArabicEOUDataGeneratorFinal(model_name=args.model)

        # Use simplified prompts
        prompts = SIMPLIFIED_PROMPTS[: args.num_conversations]

        if not prompts:
            logger.error("No prompts available!")
            return 1

        # Generate conversations
        logger.info(
            f"Generating {len(prompts)} conversations using Nebius Token Factory..."
        )
        stats = generator.generate_batch(
            prompts,
            args.output_file,
            temperature=args.temperature,
            delay_between_requests=args.delay,
        )

        # Load and validate if requested
        if args.validate:
            with open(args.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            validation_report = generator.validate_conversations(data["conversations"])

            logger.info("")
            logger.info("=== Validation Report ===")
            logger.info(
                f"Total conversations: {validation_report['total_conversations']}"
            )
            logger.info(
                f"Valid conversations: {validation_report['valid_conversations']}"
            )
            logger.info(
                f"Invalid conversations: {validation_report['invalid_conversations']}"
            )
            logger.info(
                f"Average turns per conversation: {validation_report['statistics']['avg_turns_per_conversation']:.2f}"
            )
            logger.info(f"Min turns: {validation_report['statistics']['min_turns']}")
            logger.info(f"Max turns: {validation_report['statistics']['max_turns']}")
            logger.info(
                f"EOU ratio: {validation_report['statistics']['avg_eou_ratio']:.2%}"
            )

            if validation_report["issues"]:
                logger.warning("Issues found:")
                for issue in validation_report["issues"][:10]:
                    logger.warning(f"  - {issue}")
                if len(validation_report["issues"]) > 10:
                    logger.warning(
                        f"  ... and {len(validation_report['issues']) - 10} more issues"
                    )

        logger.info("")
        logger.info(
            f"[SUCCESS] Data generation complete! Output saved to: {args.output_file}"
        )

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
