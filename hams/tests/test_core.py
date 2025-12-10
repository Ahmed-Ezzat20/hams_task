"""
Unit tests for core modules.
"""

import pytest
from hams.core.postprocessor import PostProcessor, Turn, Conversation


class TestPostProcessor:
    """Tests for PostProcessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = PostProcessor()
    
    def test_normalize_with_turns(self):
        """Test normalization with 'turns' format."""
        raw_json = {
            'conversation_id': 'test_001',
            'domain': 'test',
            'turns': [
                {'text': 'السلام عليكم', 'is_eou': True},
                {'text': 'وعليكم السلام', 'is_eou': True}
            ]
        }
        
        conversation = self.processor.normalize(raw_json)
        
        assert conversation.conversation_id == 'test_001'
        assert conversation.domain == 'test'
        assert len(conversation.turns) == 2
        assert all(isinstance(turn, Turn) for turn in conversation.turns)
        assert all(turn.is_eou for turn in conversation.turns)
    
    def test_normalize_with_dialogue(self):
        """Test normalization with 'dialogue' format."""
        raw_json = {
            'conversation_id': 'test_002',
            'domain': 'test',
            'dialogue': [
                {'speaker': 'user', 'text': 'مرحبا', 'is_eou': True},
                {'speaker': 'assistant', 'text': 'أهلا', 'is_eou': True}
            ]
        }
        
        conversation = self.processor.normalize(raw_json)
        
        assert len(conversation.turns) == 2
        assert conversation.turns[0].speaker == 'user'
        assert conversation.turns[1].speaker == 'assistant'
    
    def test_normalize_invalid_json(self):
        """Test normalization with invalid JSON."""
        with pytest.raises(ValueError):
            self.processor.normalize({'invalid': 'data'})
    
    def test_add_asr_noise_removes_punctuation(self):
        """Test that ASR noise removes punctuation."""
        text = "السلام عليكم، كيف حالك؟"
        noisy = self.processor.add_asr_noise(text, drop_punct=True)
        
        # Should not contain Arabic punctuation
        assert '،' not in noisy
        assert '؟' not in noisy
    
    def test_add_asr_noise_word_count(self):
        """Test that ASR noise keeps reasonable word count."""
        text = "السلام عليكم كيف حالك"
        original_words = len(text.split())
        
        noisy = self.processor.add_asr_noise(
            text,
            prob_swap=0.5,
            prob_filler=0.5,
            prob_repeat=0.5
        )
        noisy_words = len(noisy.split())
        
        # Word count should be > 0 and <= original + 4 (allowing for filler + repeat)
        assert noisy_words > 0
        assert noisy_words <= original_words + 4
    
    def test_augment_conversation(self):
        """Test conversation augmentation."""
        conversation = Conversation(
            conversation_id='test_003',
            domain='test',
            turns=[
                Turn(text='السلام عليكم', is_eou=True),
                Turn(text='وعليكم السلام', is_eou=True)
            ]
        )
        
        augmented = self.processor.augment_conversation(conversation, add_noise=True)
        
        assert augmented.conversation_id == 'test_003_noisy'
        assert len(augmented.turns) == len(conversation.turns)
        assert augmented.metadata['augmented'] is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        conversation = Conversation(
            conversation_id='test_004',
            domain='test',
            turns=[
                Turn(text='مرحبا', is_eou=True, speaker='user', turn_id=1)
            ]
        )
        
        conv_dict = self.processor.to_dict(conversation)
        
        assert conv_dict['conversation_id'] == 'test_004'
        assert conv_dict['domain'] == 'test'
        assert len(conv_dict['turns']) == 1
        assert conv_dict['turns'][0]['text'] == 'مرحبا'
        assert conv_dict['turns'][0]['is_eou'] is True


class TestTurn:
    """Tests for Turn dataclass."""
    
    def test_turn_creation(self):
        """Test creating a Turn."""
        turn = Turn(text="السلام عليكم", is_eou=True)
        
        assert turn.text == "السلام عليكم"
        assert turn.is_eou is True
        assert turn.speaker is None
        assert turn.turn_id is None
    
    def test_turn_with_all_fields(self):
        """Test creating a Turn with all fields."""
        turn = Turn(
            text="مرحبا",
            is_eou=True,
            speaker="user",
            turn_id=1
        )
        
        assert turn.text == "مرحبا"
        assert turn.is_eou is True
        assert turn.speaker == "user"
        assert turn.turn_id == 1


class TestConversation:
    """Tests for Conversation dataclass."""
    
    def test_conversation_creation(self):
        """Test creating a Conversation."""
        turns = [
            Turn(text="السلام عليكم", is_eou=True),
            Turn(text="وعليكم السلام", is_eou=True)
        ]
        
        conversation = Conversation(
            conversation_id="test_001",
            domain="greeting",
            turns=turns
        )
        
        assert conversation.conversation_id == "test_001"
        assert conversation.domain == "greeting"
        assert len(conversation.turns) == 2
        assert conversation.metadata is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
