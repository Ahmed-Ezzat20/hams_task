"""
CSV Exporter Module

Exports conversations to CSV format optimized for Hugging Face datasets.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from .writer import DatasetWriter
from .postprocessor import Conversation


class CSVExporter:
    """Exports conversations to CSV format for Hugging Face."""
    
    def __init__(self, output_file: str):
        """
        Initialize the CSV exporter.
        
        Args:
            output_file: Path to output CSV file
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def export_turn_level(
        self,
        conversations: List[Conversation],
        include_context: bool = True,
        context_window: int = 3
    ) -> int:
        """
        Export conversations to CSV at turn level (one row per turn).
        
        This format is ideal for EOU detection training where each turn
        is a separate training example.
        
        Args:
            conversations: List of conversations to export
            include_context: Whether to include previous turns as context
            context_window: Number of previous turns to include as context
        
        Returns:
            Number of turns written
        
        CSV Format:
            conversation_id, turn_id, speaker, text, is_eou, context, domain
        """
        fieldnames = [
            'conversation_id',
            'turn_id',
            'speaker',
            'text',
            'is_eou',
            'domain'
        ]
        
        if include_context:
            fieldnames.insert(-1, 'context')
        
        total_turns = 0
        
        with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for conv in conversations:
                for idx, turn in enumerate(conv.turns):
                    row = {
                        'conversation_id': conv.conversation_id,
                        'turn_id': turn.turn_id or idx + 1,
                        'speaker': turn.speaker or 'unknown',
                        'text': turn.text,
                        'is_eou': int(turn.is_eou),  # Convert bool to 0/1
                        'domain': conv.domain
                    }
                    
                    # Add context if requested
                    if include_context and idx > 0:
                        # Get previous turns as context
                        start_idx = max(0, idx - context_window)
                        context_turns = conv.turns[start_idx:idx]
                        context_text = " ".join([t.text for t in context_turns])
                        row['context'] = context_text
                    elif include_context:
                        row['context'] = ""
                    
                    writer.writerow(row)
                    total_turns += 1
        
        return total_turns
    
    def export_conversation_level(
        self,
        conversations: List[Conversation]
    ) -> int:
        """
        Export conversations to CSV at conversation level (one row per conversation).
        
        This format is useful for conversation-level analysis or metadata.
        
        Args:
            conversations: List of conversations to export
        
        Returns:
            Number of conversations written
        
        CSV Format:
            conversation_id, domain, num_turns, full_text, eou_labels
        """
        fieldnames = [
            'conversation_id',
            'domain',
            'num_turns',
            'full_text',
            'eou_labels'
        ]
        
        with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for conv in conversations:
                # Concatenate all turns
                full_text = " ".join([turn.text for turn in conv.turns])
                
                # Get EOU labels as comma-separated string
                eou_labels = ",".join([str(int(turn.is_eou)) for turn in conv.turns])
                
                row = {
                    'conversation_id': conv.conversation_id,
                    'domain': conv.domain,
                    'num_turns': len(conv.turns),
                    'full_text': full_text,
                    'eou_labels': eou_labels
                }
                
                writer.writerow(row)
        
        return len(conversations)
    
    def export_huggingface_format(
        self,
        conversations: List[Conversation],
        split: str = 'train'
    ) -> int:
        """
        Export conversations to Hugging Face recommended format.
        
        This creates a CSV with columns optimized for HF datasets library.
        Each row is a turn with context and label.
        
        Args:
            conversations: List of conversations to export
            split: Dataset split name ('train', 'validation', 'test')
        
        Returns:
            Number of turns written
        
        CSV Format:
            text, label, context, metadata
        """
        fieldnames = [
            'text',           # Current turn text
            'label',          # EOU label (0 or 1)
            'context',        # Previous turns
            'domain',         # Conversation domain
            'conversation_id', # For tracking
            'split'           # train/val/test
        ]
        
        total_turns = 0
        
        with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for conv in conversations:
                context_buffer = []
                
                for idx, turn in enumerate(conv.turns):
                    row = {
                        'text': turn.text,
                        'label': int(turn.is_eou),
                        'context': " ".join(context_buffer),
                        'domain': conv.domain,
                        'conversation_id': conv.conversation_id,
                        'split': split
                    }
                    
                    writer.writerow(row)
                    total_turns += 1
                    
                    # Add current turn to context buffer for next turn
                    context_buffer.append(turn.text)
                    
                    # Keep only last 3 turns as context
                    if len(context_buffer) > 3:
                        context_buffer.pop(0)
        
        return total_turns
    
    @staticmethod
    def create_splits(
        input_file: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        export_format: str = 'huggingface'
    ) -> Dict[str, str]:
        """
        Create train/val/test splits and export to CSV.
        
        Args:
            input_file: Input JSONL file with conversations
            output_dir: Output directory for split CSV files
            train_ratio: Ratio of training data (default: 0.7)
            val_ratio: Ratio of validation data (default: 0.15)
            test_ratio: Ratio of test data (default: 0.15)
            export_format: Export format ('huggingface', 'turn_level', 'conversation_level')
        
        Returns:
            Dictionary with paths to train/val/test CSV files
        """
        # Read conversations
        reader = DatasetWriter(input_file)
        conversations = reader.read_conversations()
        
        if not conversations:
            raise ValueError(f"No conversations found in {input_file}")
        
        # Calculate split sizes
        total = len(conversations)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Split conversations
        train_convs = conversations[:train_size]
        val_convs = conversations[train_size:train_size + val_size]
        test_convs = conversations[train_size + val_size:]
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export each split
        splits = {}
        
        for split_name, split_convs in [
            ('train', train_convs),
            ('validation', val_convs),
            ('test', test_convs)
        ]:
            output_file = output_path / f"{split_name}.csv"
            exporter = CSVExporter(str(output_file))
            
            if export_format == 'huggingface':
                exporter.export_huggingface_format(split_convs, split=split_name)
            elif export_format == 'turn_level':
                exporter.export_turn_level(split_convs)
            elif export_format == 'conversation_level':
                exporter.export_conversation_level(split_convs)
            else:
                raise ValueError(f"Unknown export format: {export_format}")
            
            splits[split_name] = str(output_file)
        
        return splits
