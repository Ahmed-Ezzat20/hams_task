"""
Test script to verify EOU distribution in generated conversations.
"""

import os
import sys
from hams.core.prompt_builder_v2 import EOUAwarePromptBuilder
from hams.core.generator import ConversationGenerator
from hams.core.postprocessor import PostProcessor

def test_eou_distribution(num_conversations: int = 5):
    """Test EOU distribution in generated conversations."""
    
    # Get API key
    api_key = os.getenv('NEBIUS_API_KEY')
    if not api_key:
        print("Error: NEBIUS_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize components
    print("Initializing components...")
    prompt_builder = EOUAwarePromptBuilder()
    generator = ConversationGenerator(api_key=api_key)
    postprocessor = PostProcessor()
    
    # Get prompts
    prompts = prompt_builder.get_all_eou_aware_prompts(
        style='clean',
        num_turns=10,
        target_non_eou_ratio=0.4
    )
    
    print(f"\nGenerating {num_conversations} test conversations...")
    print("=" * 60)
    
    total_turns = 0
    total_eou_true = 0
    total_eou_false = 0
    
    for i in range(num_conversations):
        prompt = prompts[i % len(prompts)]
        
        try:
            print(f"\nConversation {i+1}/{num_conversations}:")
            
            # Generate
            raw_json = generator.generate(prompt, temperature=0.7)
            
            # Normalize
            conversation = postprocessor.normalize(raw_json)
            
            # Count EOU labels
            eou_true = sum(1 for turn in conversation.turns if turn.is_eou)
            eou_false = len(conversation.turns) - eou_true
            
            total_turns += len(conversation.turns)
            total_eou_true += eou_true
            total_eou_false += eou_false
            
            print(f"  Domain: {conversation.domain}")
            print(f"  Total turns: {len(conversation.turns)}")
            print(f"  EOU=true: {eou_true} ({eou_true/len(conversation.turns)*100:.1f}%)")
            print(f"  EOU=false: {eou_false} ({eou_false/len(conversation.turns)*100:.1f}%)")
            
            # Show sample turns
            print(f"  Sample turns:")
            for turn in conversation.turns[:5]:
                eou_label = "✓" if turn.is_eou else "✗"
                print(f"    [{eou_label}] {turn.text[:50]}...")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS:")
    print("=" * 60)
    print(f"Total turns: {total_turns}")
    print(f"EOU=true: {total_eou_true} ({total_eou_true/total_turns*100:.1f}%)")
    print(f"EOU=false: {total_eou_false} ({total_eou_false/total_turns*100:.1f}%)")
    print(f"\nTarget: 60% EOU=true / 40% EOU=false")
    
    # Check if distribution is acceptable
    eou_false_ratio = total_eou_false / total_turns
    if 0.30 <= eou_false_ratio <= 0.50:
        print("✅ EOU distribution is BALANCED!")
    else:
        print("⚠️  EOU distribution needs adjustment")
    
    print("=" * 60)

if __name__ == '__main__':
    test_eou_distribution(num_conversations=5)
