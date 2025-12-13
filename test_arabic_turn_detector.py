#!/usr/bin/env python3
"""
Test script for Arabic Turn Detector

This script tests the Arabic turn detector plugin to verify:
1. Model loading works correctly
2. Inference produces results
3. Logging outputs EOU probability
"""

import json
import logging
import sys
from pathlib import Path

# Enable DEBUG logging to see eou prediction logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Add the plugin to Python path
plugin_path = Path(__file__).parent / "livekit_plugins_arabic_turn_detector"
sys.path.insert(0, str(plugin_path))

from livekit.plugins.arabic_turn_detector.arabic import _EOURunnerAr

def test_arabic_turn_detector():
    """Test the Arabic turn detector"""
    
    print("=" * 70)
    print("Testing Arabic Turn Detector")
    print("=" * 70)
    
    # Model path
    model_path = Path(__file__).parent / "eou_model" / "models" / "eou_model_quantized.onnx"
    
    if not model_path.exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("\nPlease ensure you have:")
        print("1. Trained your EOU model")
        print("2. Converted it to ONNX format")
        print("3. Quantized it")
        print("4. Placed it in the correct directory")
        return False
    
    print(f"\n✓ Model found at: {model_path}")
    
    # Initialize runner
    print("\n" + "=" * 70)
    print("Initializing Arabic EOU Runner...")
    print("=" * 70)
    
    try:
        runner = _EOURunnerAr(
            model_path=str(model_path),
            unlikely_threshold=0.7
        )
        
        runner.initialize()
        print("\n✓ Runner initialized successfully")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize runner: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test cases
    test_cases = [
        {
            "name": "Incomplete utterance (greeting)",
            "chat_ctx": [
                {"role": "user", "content": "مرحبا"}
            ],
            "expected_eou": False
        },
        {
            "name": "Complete utterance (question)",
            "chat_ctx": [
                {"role": "user", "content": "مرحبا كيف حالك؟"}
            ],
            "expected_eou": True
        },
        {
            "name": "Incomplete response",
            "chat_ctx": [
                {"role": "user", "content": "كيف حالك؟"},
                {"role": "assistant", "content": "أنا بخير"}
            ],
            "expected_eou": False
        },
        {
            "name": "Complete response",
            "chat_ctx": [
                {"role": "user", "content": "كيف حالك؟"},
                {"role": "assistant", "content": "أنا بخير والحمد لله، شكراً لسؤالك."}
            ],
            "expected_eou": True
        },
        {
            "name": "Multi-turn conversation",
            "chat_ctx": [
                {"role": "user", "content": "السلام عليكم"},
                {"role": "assistant", "content": "وعليكم السلام ورحمة الله"},
                {"role": "user", "content": "كيف يمكنني مساعدتك اليوم؟"}
            ],
            "expected_eou": True
        }
    ]
    
    # Run tests
    print("\n" + "=" * 70)
    print("Running Test Cases")
    print("=" * 70)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        print(f"Chat context: {test_case['chat_ctx']}")
        
        # Prepare input
        input_data = json.dumps({"chat_ctx": test_case["chat_ctx"]}).encode()
        
        try:
            # Run inference
            result_bytes = runner.run(input_data)
            result = json.loads(result_bytes.decode())
            
            print(f"\nResult:")
            print(f"  eou_probability: {result.get('eou_probability', 'N/A')}")
            print(f"  is_eou: {result.get('is_eou', 'N/A')}")
            print(f"  threshold: {result.get('threshold', 'N/A')}")
            print(f"  duration: {result.get('duration', 'N/A')}s")
            print(f"  input (truncated): {result.get('input', 'N/A')[:50]}...")
            
            # Check result
            expected = test_case["expected_eou"]
            actual = result.get("is_eou", False)
            
            if actual == expected:
                print(f"\n✓ PASS: Expected {expected}, got {actual}")
                results.append(True)
            else:
                print(f"\n✗ FAIL: Expected {expected}, got {actual}")
                results.append(False)
                
        except Exception as e:
            print(f"\n❌ Error running test: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = test_arabic_turn_detector()
    sys.exit(0 if success else 1)
