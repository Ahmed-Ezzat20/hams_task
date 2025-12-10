"""
EOU-Aware Prompt Builder Module

Uses few-shot prompting with explicit examples to generate balanced EOU datasets.
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


class EOUAwarePromptBuilder:
    """Builds prompts for EOU detection with few-shot examples."""
    
    # System prompt positioning LLM as synthetic data expert
    SYSTEM_PROMPT = """You are a synthetic data expert specializing in Arabic conversational AI and End-of-Utterance (EOU) detection.

Your task is to generate realistic Saudi Arabic conversations with accurate EOU labels.

IMPORTANT: Not every turn is a complete utterance!
- is_eou: true = Speaker has finished their complete thought/turn
- is_eou: false = Speaker is mid-utterance, incomplete, or will continue

Generate conversations with approximately 40% non-EOU turns (is_eou: false) to create a balanced dataset."""

    # Few-shot examples showing both EOU types
    FEW_SHOT_EXAMPLES = """
EXAMPLES OF EOU LABELING:

Example 1 - Restaurant Reservation:
{
  "turns": [
    {"text": "السلام عليكم", "is_eou": true},
    {"text": "وعليكم السلام، أهلاً", "is_eou": true},
    {"text": "أبي أحجز طاولة", "is_eou": false},
    {"text": "لأربعة أشخاص", "is_eou": false},
    {"text": "يوم الخميس الساعة ثمانية", "is_eou": true},
    {"text": "تمام، خلني أشيك", "is_eou": false},
    {"text": "عندنا طاولة متاحة الساعة ثمانية ونص", "is_eou": true},
    {"text": "ممتاز", "is_eou": false},
    {"text": "أحجزها", "is_eou": true}
  ]
}

Example 2 - Bank Inquiry:
{
  "turns": [
    {"text": "صباح الخير", "is_eou": true},
    {"text": "صباح النور، كيف أقدر أساعدك؟", "is_eou": true},
    {"text": "أبي أستفسر عن", "is_eou": false},
    {"text": "رصيد حسابي", "is_eou": true},
    {"text": "طبعاً، ممكن", "is_eou": false},
    {"text": "رقم الحساب؟", "is_eou": true},
    {"text": "الحساب رقم", "is_eou": false},
    {"text": "١٢٣٤٥٦٧٨", "is_eou": true}
  ]
}

Example 3 - Real-world Incomplete Utterances:
{
  "turns": [
    {"text": "طيب، بس لازم تتفق أول", "is_eou": false},
    {"text": "هل تقدر توصلني بكرا؟", "is_eou": true},
    {"text": "أنا حاولت، لكن ما فهمت الأرس", "is_eou": false},
    {"text": "بس أنت ما قلت لي متى بدأ", "is_eou": false},
    {"text": "شكراً، أكثر على المساعدة", "is_eou": true},
    {"text": "طيب، تكمل بعدين؟", "is_eou": true},
    {"text": "يعني أنا حب أنظر منك نرد على", "is_eou": false},
    {"text": "أنا آسف إذا رأيتك", "is_eou": false},
    {"text": "لا تنسى ترجع المفتاح بعدين", "is_eou": true},
    {"text": "هذا الشيء ما توقعت بصير", "is_eou": true},
    {"text": "هو قال لي أو لازم نتته", "is_eou": false},
    {"text": "إيش رأيك نطلب بيتزا؟", "is_eou": true},
    {"text": "أصلاً ما كان المفروض نجي", "is_eou": false},
    {"text": "طيب، تكمل الحين ولا بعدين؟", "is_eou": true},
    {"text": "أنا ما أقدر أقول لحالي", "is_eou": false},
    {"text": "والله ما كنت أقصد", "is_eou": false}
  ]
}

KEY PATTERNS FOR is_eou: false:
- Incomplete sentences: "أبي أحجز..." "أنا حاولت، لكن ما فهمت..."
- Mid-thought pauses: "خلني أشيك..." "طيب، بس لازم تتفق أول..."
- Partial information: "الحساب رقم..." "بس أنت ما قلت لي متى..."
- Hesitations: "يعني..." "أصلاً ما كان المفروض..."
- Continuations: "وكمان..." "هو قال لي أو لازم..."
- Unfinished thoughts: "أنا ما أقدر أقول لحالي" "والله ما كنت أقصد"

KEY PATTERNS FOR is_eou: true:
- Complete greetings: "السلام عليكم"
- Complete questions: "كيف أقدر أساعدك؟"
- Complete statements: "ممتاز، أحجزها"
- Confirmations: "تمام" "طبعاً"
- Complete information: "يوم الخميس الساعة ثمانية"
"""

    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize the EOU-aware prompt builder.
        
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
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        templates = []
        for prompt_data in data.get('prompts', []):
            template = PromptTemplate(
                id=prompt_data['id'],
                domain=prompt_data['domain'],
                description=prompt_data['description'],
                scenario=prompt_data['scenario']
            )
            templates.append(template)
        
        return templates
    
    def build_eou_aware_prompt(
        self,
        template: PromptTemplate,
        style: str = "clean",
        num_turns: int = 10,
        target_non_eou_ratio: float = 0.4
    ) -> str:
        """
        Build an EOU-aware conversation generation prompt with few-shot examples.
        
        Args:
            template: The prompt template to use
            style: "clean" for standard output, "asr_like" for ASR-style output
            num_turns: Target number of turns in the conversation
            target_non_eou_ratio: Target ratio of non-EOU turns (default: 0.4 = 40%)
        
        Returns:
            Formatted prompt string with system prompt and few-shot examples
        """
        target_non_eou_count = int(num_turns * target_non_eou_ratio)
        target_eou_count = num_turns - target_non_eou_count
        
        prompt = f"""{self.SYSTEM_PROMPT}

{self.FEW_SHOT_EXAMPLES}

NOW GENERATE A NEW CONVERSATION:

SCENARIO:
{template.scenario}

REQUIREMENTS:
- Language: Saudi Arabic dialect
- Total turns: {num_turns} turns
- Target EOU distribution:
  * is_eou: true → {target_eou_count} turns (~{int((1-target_non_eou_ratio)*100)}%)
  * is_eou: false → {target_non_eou_count} turns (~{int(target_non_eou_ratio*100)}%)
- Natural conversational flow
- Realistic incomplete utterances"""

        if style == "asr_like":
            prompt += """
- Style: ASR-like output (simulate automatic speech recognition)
  * Minimal punctuation
  * Some word repetitions
  * Natural fillers (آه، يعني، خلاص)
  * Lowercase where appropriate"""
        else:  # clean
            prompt += """
- Style: Clean conversational text
  * Proper punctuation
  * Natural hesitations and fillers
  * Clear and readable"""
        
        prompt += f"""

OUTPUT FORMAT (JSON only, no markdown):
{{
  "conversation_id": "conv_{template.id}",
  "domain": "{template.domain}",
  "turns": [
    {{"text": "...", "is_eou": true}},
    {{"text": "...", "is_eou": false}},
    ...
  ]
}}

CRITICAL: Include approximately {target_non_eou_count} turns with is_eou: false (incomplete utterances).
Return ONLY valid JSON. No markdown code blocks."""
        
        return prompt
    
    def get_all_eou_aware_prompts(
        self,
        style: str = "clean",
        num_turns: int = 10,
        target_non_eou_ratio: float = 0.4,
        domains: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get all EOU-aware prompts, optionally filtered by domain.
        
        Args:
            style: "clean" or "asr_like"
            num_turns: Target number of turns
            target_non_eou_ratio: Target ratio of non-EOU turns
            domains: List of domains to include (None = all domains)
        
        Returns:
            List of formatted prompt strings
        """
        templates = self.templates
        
        if domains:
            templates = [t for t in templates if t.domain in domains]
        
        return [
            self.build_eou_aware_prompt(t, style, num_turns, target_non_eou_ratio)
            for t in templates
        ]
    
    def get_system_message(self) -> str:
        """Get the system message for the LLM."""
        return self.SYSTEM_PROMPT
