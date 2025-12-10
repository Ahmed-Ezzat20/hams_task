"""
Prompt Builder Module

Centralized prompt management for Arabic EOU conversation generation.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Represents a conversation prompt template."""

    id: str
    domain: str
    description: str
    scenario: str


class PromptBuilder:
    """Builds prompts for conversation generation."""

    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize the prompt builder.

        Args:
            prompts_file: Path to YAML file containing prompts.
                         If None, uses default prompts.yaml in package.
        """
        if prompts_file is None:
            prompts_file = Path(__file__).parent.parent / "prompts.yaml"

        self.prompts_file = Path(prompts_file)
        self.templates = self._load_templates()

    def _load_templates(self) -> List[PromptTemplate]:
        """Load prompt templates from YAML file."""
        with open(self.prompts_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        templates = []
        for prompt_data in data.get("prompts", []):
            template = PromptTemplate(
                id=prompt_data["id"],
                domain=prompt_data["domain"],
                description=prompt_data["description"],
                scenario=prompt_data["scenario"],
            )
            templates.append(template)

        return templates

    def build_prompt(
        self, template: PromptTemplate, style: str = "clean", num_turns: int = 8
    ) -> str:
        """
        Build a conversation generation prompt.

        Args:
            template: The prompt template to use.
            style: "clean" for standard output, "asr_like" for ASR-style output.
            num_turns: Target number of turns in the conversation.

        Returns:
            Formatted prompt string.
        """
        base_prompt = f"""Generate a realistic Arabic conversation simulating voice conversation in Saudi Arabic Najdi dialect.

SCENARIO:
{template.scenario}

REQUIREMENTS:
- Language: Saudi Arabic Najdi dialect
- Number of turns: {num_turns}-{num_turns+2} turns
- Natural conversational flow
- Clear turn boundaries"""

        if style == "asr_like":
            base_prompt += """
- Style: ASR-like output (simulate automatic speech recognition)
  * Remove most punctuation (،؟.!)
  * Include common ASR mistakes: mis-casing, missing diacritics
  * Insert natural fillers (آه، يعني، خلاص) frequently
  * Some word repetitions and hesitations"""
        else:  # clean
            base_prompt += """
- Style: Clean conversational text
  * Proper punctuation
  * Natural hesitations and filler words
  * Clear and readable"""

        base_prompt += (
            """

OUTPUT FORMAT (JSON only, no markdown):
{
  "conversation_id": "conv_"""
            + template.id
            + """",
  "domain": """
            " + template.domain + "
            """,
  "turns": [
    {"text": "السلام عليكم، ...", "is_eou": true},
    {"text": "وعليكم السلام، ...", "is_eou": true}
  ]
}

Return ONLY valid JSON. No markdown code blocks."""
        )

        return base_prompt

    def get_all_prompts(
        self,
        style: str = "clean",
        num_turns: int = 8,
        domains: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get all prompts, optionally filtered by domain.

        Args:
            style: "clean" or "asr_like"
            num_turns: Target number of turns
            domains: List of domains to include (None = all domains)

        Returns:
            List of formatted prompt strings.
        """
        templates = self.templates

        if domains:
            templates = [t for t in templates if t.domain in domains]

        return [self.build_prompt(t, style, num_turns) for t in templates]

    def get_prompt_by_id(
        self, prompt_id: str, style: str = "clean", num_turns: int = 8
    ) -> Optional[str]:
        """
        Get a specific prompt by ID.

        Args:
            prompt_id: The prompt template ID
            style: "clean" or "asr_like"
            num_turns: Target number of turns

        Returns:
            Formatted prompt string or None if not found.
        """
        for template in self.templates:
            if template.id == prompt_id:
                return self.build_prompt(template, style, num_turns)
        return None

    def get_prompts_by_domain(
        self, domain: str, style: str = "clean", num_turns: int = 8
    ) -> List[str]:
        """
        Get all prompts for a specific domain.

        Args:
            domain: The domain to filter by
            style: "clean" or "asr_like"
            num_turns: Target number of turns

        Returns:
            List of formatted prompt strings.
        """
        return self.get_all_prompts(style, num_turns, domains=[domain])
