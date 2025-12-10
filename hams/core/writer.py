"""
Dataset Writer Module

Writes conversations to JSONL format for easy streaming and merging.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, TextIO
from .postprocessor import Conversation, PostProcessor


class DatasetWriter:
    """Writes conversations to JSONL files."""
    
    def __init__(self, output_file: str):
        """
        Initialize the dataset writer.
        
        Args:
            output_file: Path to output JSONL file
        """
        self.output_file = Path(output_file)
        self.postprocessor = PostProcessor()
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def write_conversation(
        self,
        conversation: Conversation,
        file_handle: TextIO
    ) -> None:
        """
        Write a single conversation to file.
        
        Args:
            conversation: The conversation to write
            file_handle: Open file handle
        """
        conv_dict = self.postprocessor.to_dict(conversation)
        json_line = json.dumps(conv_dict, ensure_ascii=False)
        file_handle.write(json_line + '\n')
    
    def write_conversations(
        self,
        conversations: List[Conversation],
        append: bool = False
    ) -> int:
        """
        Write multiple conversations to JSONL file.
        
        Args:
            conversations: List of conversations to write
            append: Whether to append to existing file
        
        Returns:
            Number of conversations written
        """
        mode = 'a' if append else 'w'
        
        with open(self.output_file, mode, encoding='utf-8') as f:
            for conversation in conversations:
                self.write_conversation(conversation, f)
        
        return len(conversations)
    
    def read_conversations(self) -> List[Conversation]:
        """
        Read conversations from JSONL file.
        
        Returns:
            List of Conversation objects
        """
        if not self.output_file.exists():
            return []
        
        conversations = []
        
        with open(self.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    conv_dict = json.loads(line)
                    # Reconstruct Conversation object
                    conversation = self.postprocessor.normalize(conv_dict)
                    conversations.append(conversation)
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return conversations
    
    def merge_files(self, input_files: List[str], remove_duplicates: bool = True) -> int:
        """
        Merge multiple JSONL files into this writer's output file.
        
        Args:
            input_files: List of input JSONL file paths
            remove_duplicates: Whether to remove duplicate conversation IDs
        
        Returns:
            Total number of conversations written
        """
        seen_ids = set()
        total_written = 0
        
        with open(self.output_file, 'w', encoding='utf-8') as out_f:
            for input_file in input_files:
                input_path = Path(input_file)
                if not input_path.exists():
                    continue
                
                with open(input_path, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            conv_dict = json.loads(line)
                            conv_id = conv_dict.get('conversation_id', '')
                            
                            # Skip duplicates if requested
                            if remove_duplicates and conv_id in seen_ids:
                                continue
                            
                            seen_ids.add(conv_id)
                            out_f.write(line + '\n')
                            total_written += 1
                            
                        except json.JSONDecodeError:
                            continue
        
        return total_written
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        conversations = self.read_conversations()
        
        if not conversations:
            return {
                'total_conversations': 0,
                'total_turns': 0,
                'avg_turns_per_conversation': 0.0,
                'domains': {}
            }
        
        total_turns = sum(len(conv.turns) for conv in conversations)
        domains = {}
        
        for conv in conversations:
            domain = conv.domain
            domains[domain] = domains.get(domain, 0) + 1
        
        return {
            'total_conversations': len(conversations),
            'total_turns': total_turns,
            'avg_turns_per_conversation': total_turns / len(conversations),
            'min_turns': min(len(conv.turns) for conv in conversations),
            'max_turns': max(len(conv.turns) for conv in conversations),
            'domains': domains
        }
