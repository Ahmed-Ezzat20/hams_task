"""
Post-processor Module

Normalizes raw JSON to Turn dataclasses and applies ASR-style degradation.
"""

import re
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Turn:
    """Represents a single conversational turn."""
    text: str
    is_eou: bool
    speaker: Optional[str] = None
    turn_id: Optional[int] = None


@dataclass
class Conversation:
    """Represents a complete conversation."""
    conversation_id: str
    domain: str
    turns: List[Turn]
    metadata: Optional[Dict[str, Any]] = None


class PostProcessor:
    """Normalizes and augments conversation data."""
    
    # Common Arabic homophone/mistake mappings for ASR simulation
    NOISY_MAP = {
        "السلام": "السلم",
        "عليكم": "عليكوم",
        "وعليكم": "وعليكوم",
        "الحمد": "الحمدو",
        "لله": "لل",
        "تمام": "تمم",
        "ممتاز": "ممتز",
        "شكراً": "شكرا",
        "من": "مين",
        "هذا": "هاذا",
        "ذلك": "ذاك",
    }
    
    def normalize(self, raw_json: Dict[str, Any]) -> Conversation:
        """
        Normalize raw JSON to Conversation with Turn dataclasses.
        
        Args:
            raw_json: Raw JSON dictionary from the generator
        
        Returns:
            Normalized Conversation object
        
        Raises:
            ValueError: If the JSON structure is invalid
        """
        if not isinstance(raw_json, dict):
            raise ValueError("Invalid JSON: expected dictionary")
        
        conversation_id = raw_json.get('conversation_id', 'unknown')
        domain = raw_json.get('domain', 'unknown')
        
        # Handle both 'turns' and 'dialogue' formats
        raw_turns = raw_json.get('turns') or raw_json.get('dialogue', [])
        
        if not isinstance(raw_turns, list):
            raise ValueError("Invalid JSON: 'turns' or 'dialogue' must be a list")
        
        turns = []
        for idx, raw_turn in enumerate(raw_turns, 1):
            if not isinstance(raw_turn, dict):
                continue
            
            text = raw_turn.get('text', '')
            if not text:
                continue
            
            turn = Turn(
                text=text.strip(),
                is_eou=raw_turn.get('is_eou', True),
                speaker=raw_turn.get('speaker'),
                turn_id=idx
            )
            turns.append(turn)
        
        if not turns:
            raise ValueError("Invalid JSON: no valid turns found")
        
        return Conversation(
            conversation_id=conversation_id,
            domain=domain,
            turns=turns,
            metadata=raw_json.get('metadata')
        )
    
    def add_asr_noise(
        self,
        text: str,
        drop_punct: bool = True,
        prob_swap: float = 0.1,
        prob_filler: float = 0.2,
        prob_repeat: float = 0.1
    ) -> str:
        """
        Add ASR-style noise to text.
        
        Simulates automatic speech recognition imperfections:
        - Removes punctuation
        - Swaps words with common homophones
        - Adds filler words
        - Adds word repetitions
        
        Args:
            text: Clean input text
            drop_punct: Whether to remove punctuation
            prob_swap: Probability of swapping a word (0.0-1.0)
            prob_filler: Probability of adding a filler word (0.0-1.0)
            prob_repeat: Probability of repeating a word (0.0-1.0)
        
        Returns:
            Noisy text simulating ASR output
        """
        # Remove Arabic punctuation
        if drop_punct:
            text = re.sub(r'[،؟.!؛:]', '', text)
        
        words = text.split()
        noisy_words = []
        
        for word in words:
            # Maybe swap with homophone
            if random.random() < prob_swap:
                word = self.NOISY_MAP.get(word, word)
            
            noisy_words.append(word)
            
            # Maybe repeat word (stuttering)
            if random.random() < prob_repeat:
                noisy_words.append(word)
        
        # Maybe add filler word
        fillers = ["يعني", "آه", "خلاص", "تمام", "ممم"]
        if random.random() < prob_filler:
            filler = random.choice(fillers)
            # Insert at random position
            if noisy_words:
                pos = random.randint(0, len(noisy_words))
                noisy_words.insert(pos, filler)
        
        return " ".join(noisy_words)
    
    def augment_conversation(
        self,
        conversation: Conversation,
        add_noise: bool = True,
        **noise_params
    ) -> Conversation:
        """
        Augment a conversation with ASR-style noise.
        
        Args:
            conversation: The conversation to augment
            add_noise: Whether to add ASR noise
            **noise_params: Parameters for add_asr_noise()
        
        Returns:
            New Conversation with augmented turns
        """
        if not add_noise:
            return conversation
        
        noisy_turns = []
        for turn in conversation.turns:
            noisy_text = self.add_asr_noise(turn.text, **noise_params)
            noisy_turn = Turn(
                text=noisy_text,
                is_eou=turn.is_eou,
                speaker=turn.speaker,
                turn_id=turn.turn_id
            )
            noisy_turns.append(noisy_turn)
        
        return Conversation(
            conversation_id=f"{conversation.conversation_id}_noisy",
            domain=conversation.domain,
            turns=noisy_turns,
            metadata={
                **(conversation.metadata or {}),
                'augmented': True,
                'augmentation_type': 'asr_noise'
            }
        )
    
    def to_dict(self, conversation: Conversation) -> Dict[str, Any]:
        """
        Convert Conversation to dictionary.
        
        Args:
            conversation: The conversation to convert
        
        Returns:
            Dictionary representation
        """
        return {
            'conversation_id': conversation.conversation_id,
            'domain': conversation.domain,
            'turns': [asdict(turn) for turn in conversation.turns],
            'metadata': conversation.metadata
        }
