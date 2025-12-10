"""
CSV-Based Prompt Builder Module

Uses streamlined prompting to generate CSV format data directly.
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Represents a conversation prompt template."""

    id: str
    domain: str
    description: str
    scenario: str


class EOUAwarePromptBuilder:
    """Builds prompts for EOU detection with CSV output."""

    # System prompt for CSV-based generation
    SYSTEM_PROMPT = """You are a synthetic data expert specializing in Arabic conversational AI, Saudi dialect generation, and End-of-Utterance (EOU) detection.  
Your goal is to generate high-quality, non-redundant Saudi Arabic conversational samples — including ASR-like imperfect transcripts — with accurate binary EOU labels.  
The data will train an EOU text-classification model for a voice agent.

# 1. OUTPUT FORMAT
Produce data as **CSV-like lines** with **no header**, following this schema:

utterance,style,label

Where:
- `utterance` = the user message
- `style` = `informal` OR `formal` OR `asr_like`
- `label` = `1` for end-of-utterance (EOU), `0` for not end-of-utterance

**Important rules for formatting:**
- Do NOT use commas in the utterance.
- If needed, replace commas with: `—` `؛` `-`
- **CRITICAL: Do NOT use `…` or `...` to mark incomplete utterances.**
- Non-EOU utterances should be semantically incomplete WITHOUT punctuation hints.
- Output ONLY the CSV lines. No explanations.

---

# 2. STYLES TO GENERATE

## A. Informal
Use natural Saudi Najdi slang + common Gulf phrasing.  
Casual, spontaneous, spoken tone.

## B. Formal
Use MSA-infused Saudi phrasing.  
Polite, professional, service-oriented style.

## C. ASR-Like (voice transcript style)
Simulate typical ASR imperfections while keeping the meaning understandable:
- Light vowel/hamza drops: "ابي" → "أبي", "هذي الاشياء" → "هذي الاشيا"
- Occasional character swaps: "متى" → "مته"
- Word merges/splits: "على طول" → "علطول"
- Minor omissions: "وش سالفة" → "سالفة"
- Occasional extra pauses like "…"

**Avoid:**
- Heavy corruption
- Nonsense text
- Foreign filler noise ("um", "uh")
- Timestamps or tags
ASR-like text must stay readable and semantically meaningful.

---

# 3. EOU LABELING RULES
Assign:
- **1 (EOU)** → message is complete
- **0 (Not EOU)** → trailing, incomplete, suggests more to come

Examples of EOU (label=1):
- "طيب متى يوصل الطلب؟" → complete question
- "هاليومين بوصل المستندات لكم" → complete statement
- "هل تقدر تساعدني؟" → complete request
- "شكراً على المساعدة" → complete gratitude
- "أنا موافق على الشروط" → complete agreement

Examples of Non-EOU (label=0) - DIVERSE PATTERNS:
- "طيب، بس لازم نتفق أول" → complete middle sentence
- "بس انت ما قلت لي متى نبدأ" → statement needing response
- "يعني أنا أقصد" → mid-thought filler
- "هل تقدر تشوف" → incomplete question
- "أنا كنت بقولك" → trailing verb phrase
- "المشكلة إن الموعد" → incomplete explanation
- "خلاص فهمت، بس" → trailing conjunction
- "لو سمحت ممكن" → incomplete request  

---

# 4. GENERATION CONSTRAINTS
- Generate **as many samples as possible** per batch.
- Ensure **zero repetition** across lines.
- NO hallucinated metadata (no IDs, no explanations).
- Vary:
  - style (informal / formal / asr_like)
  - topics (daily life, work, appointments, deliveries, telecom, bank, shopping, tech support, social chat)
  - lengths (short, medium, long)
- Avoid real personal names or brands (use generic references like "المحل", "الشركة").

---

# 5. OUTPUT EXAMPLE (STRUCTURE REFERENCE ONLY — DO NOT REPEAT)

وش وضع الاجتماع اليوم؟,informal,1  
لحظه بخليك مع المدير,formal,0  
تقدر تتأكد من الطلبيه لو سمحت؟,formal,1  
انا كنت افكر نروح بكرا,informal,0  
ابي اعرف ليه الطلب تاخر مره,asr_like,1  
كنت بسأل عن عرض الانترنت الجديد,asr_like,0  
يعني أنا كنت بقول,informal,0  
هل في مشكلة بالخدمة؟,formal,1  
بس المشكلة إن,asr_like,0  
شكرا على المساعده,informal,1  

---

# 6. START GENERATING NOW
Begin generating a large batch of **new**, **diverse**, **high-quality** samples.  
Output ONLY the CSV-like lines — nothing else."""

    def __init__(self):
        """
        Initialize the EOU-aware prompt builder.
        """
        # Hardcoded templates for common Saudi Arabic conversation domains
        self.templates = [
            PromptTemplate(
                id="restaurant",
                domain="restaurant",
                description="Restaurant reservations and food ordering",
                scenario="Customer making reservations, ordering food, or asking about menu items"
            ),
            PromptTemplate(
                id="banking",
                domain="banking",
                description="Banking inquiries and transactions",
                scenario="Customer inquiring about account balance, transactions, or services"
            ),
            PromptTemplate(
                id="hospitality",
                domain="hospitality",
                description="Hotel bookings and travel arrangements",
                scenario="Traveler booking rooms, asking about amenities, or making special requests"
            ),
            PromptTemplate(
                id="healthcare",
                domain="healthcare",
                description="Medical appointments and health inquiries",
                scenario="Patient scheduling appointments, describing symptoms, or asking health questions"
            ),
            PromptTemplate(
                id="social",
                domain="social",
                description="Social conversations between friends and family",
                scenario="Friends or family discussing daily life, plans, or catching up"
            ),
            PromptTemplate(
                id="retail",
                domain="retail",
                description="Shopping and retail transactions",
                scenario="Customer browsing products, negotiating prices, or making purchases"
            ),
            PromptTemplate(
                id="transportation",
                domain="transportation",
                description="Transportation and travel services",
                scenario="Customer renting cars, booking rides, or asking about travel options"
            ),
            PromptTemplate(
                id="professional",
                domain="professional",
                description="Professional and work-related conversations",
                scenario="Job interviews, work discussions, or professional inquiries"
            )
        ]

    def build_csv_prompt(
        self, template: PromptTemplate, target_samples: int = 50
    ) -> str:
        """
        Build a CSV generation prompt.

        Args:
            template: The prompt template to use
            target_samples: Target number of samples to generate

        Returns:
            Formatted prompt string for CSV generation
        """
        prompt = f"""Generate {target_samples} diverse Saudi Arabic EOU samples for the following scenario:

DOMAIN: {template.domain}
SCENARIO: {template.scenario}

REQUIREMENTS:
- Generate exactly {target_samples} CSV lines
- Mix of informal, formal, and asr_like styles
- Approximately 40% with label=0 (incomplete utterances)
- Approximately 60% with label=1 (complete utterances)
- Vary topics within the {template.domain} domain
- NO repetition
- NO explanations

OUTPUT FORMAT:
utterance,style,label

Generate {target_samples} lines now:"""

        return prompt

    def get_all_csv_prompts(
        self, target_samples_per_domain: int = 50, domains: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get all CSV generation prompts, optionally filtered by domain.

        Args:
            target_samples_per_domain: Target number of samples per domain
            domains: List of domains to include (None = all domains)

        Returns:
            List of formatted prompt strings
        """
        templates = self.templates

        if domains:
            templates = [t for t in templates if t.domain in domains]

        return [self.build_csv_prompt(t, target_samples_per_domain) for t in templates]

    def get_system_message(self) -> str:
        """Get the system message for the LLM."""
        return self.SYSTEM_PROMPT
