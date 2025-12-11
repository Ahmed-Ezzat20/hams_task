"""
Simple Arabic Voice Agent Example
==================================

This example demonstrates how to create a simple Arabic voice agent using
the Arabic Turn Detector plugin with LiveKit.

Requirements:
    - LiveKit server running
    - LIVEKIT_URL and LIVEKIT_API_KEY environment variables set
    - Arabic EOU model trained and converted to ONNX
    - ElevenLabs API key (or other TTS provider)
    - Deepgram API key (or other STT provider)

Usage:
    python simple_agent.py
"""

import asyncio
import logging
import os
from pathlib import Path

from livekit import rtc
from livekit.agents import (
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)

# Import STT/TTS providers
try:
    from livekit.plugins import elevenlabs
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("ElevenLabs not available. Install with: pip install livekit-plugins-elevenlabs")

try:
    from livekit.plugins import deepgram
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    logging.warning("Deepgram not available. Install with: pip install livekit-plugins-deepgram")

# Import Arabic turn detector
from livekit.plugins.arabic_turn_detector import ArabicTurnDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
MODEL_PATH = os.getenv(
    "ARABIC_EOU_MODEL_PATH",
    "./models/eou_model_quantized.onnx"
)
UNLIKELY_THRESHOLD = float(os.getenv("EOU_THRESHOLD", "0.7"))
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "v7UCHHCrHj1KBa4E41gb")


async def entrypoint(ctx: JobContext):
    """
    Agent entrypoint
    
    This function is called when a participant joins a room.
    """
    logger.info("Agent starting...")
    
    # Connect to room
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")
    
    # Create Arabic turn detector
    logger.info(f"Loading Arabic turn detector from: {MODEL_PATH}")
    try:
        arabic_turn_detector = ArabicTurnDetector(
            model_path=MODEL_PATH,
            unlikely_threshold=UNLIKELY_THRESHOLD
        )
        logger.info(f"✓ Arabic turn detector loaded (threshold={UNLIKELY_THRESHOLD})")
    except Exception as e:
        logger.error(f"Failed to load Arabic turn detector: {e}")
        logger.error("Make sure you have:")
        logger.error("1. Trained and converted your model to ONNX")
        logger.error("2. Set ARABIC_EOU_MODEL_PATH environment variable")
        logger.error("3. Model file exists at the specified path")
        raise
    
    # Configure STT
    if STT_AVAILABLE:
        stt = deepgram.STT(
            language="ar",
            model="nova-2",
        )
        logger.info("✓ STT configured (Deepgram, Arabic)")
    else:
        logger.error("STT not available. Please install: pip install livekit-plugins-deepgram")
        return
    
    # Configure TTS
    if TTS_AVAILABLE:
        tts = elevenlabs.TTS(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id="eleven_multilingual_v2",
            language="ar",
        )
        logger.info(f"✓ TTS configured (ElevenLabs, voice={ELEVENLABS_VOICE_ID})")
    else:
        logger.error("TTS not available. Please install: pip install livekit-plugins-elevenlabs")
        return
    
    # Create agent session
    logger.info("Creating agent session...")
    session = AgentSession(
        # Speech recognition
        stt=stt,
        
        # Speech synthesis
        tts=tts,
        
        # Turn detection with Arabic model
        turn_detector=arabic_turn_detector,
        
        # Endpointing configuration
        min_endpointing_delay=0.5,  # Wait 500ms after speech ends
        interrupt_speech_duration=0.3,  # Allow interruption after 300ms
        interrupt_min_words=2,  # Require at least 2 words to interrupt
        
        # Other settings
        allow_interruptions=True,
        max_nested_fnc_calls=5,
    )
    
    logger.info("✓ Agent session created")
    
    # Define agent behavior
    @session.on("user_speech_committed")
    async def on_user_speech(text: str):
        """Handle user speech"""
        logger.info(f"User said: {text}")
        
        # Simple echo response (replace with your logic)
        response = f"سمعتك تقول: {text}"
        
        logger.info(f"Agent responding: {response}")
        await session.say(response)
    
    @session.on("agent_speech_committed")
    async def on_agent_speech(text: str):
        """Handle agent speech"""
        logger.info(f"Agent said: {text}")
    
    @session.on("turn_detected")
    async def on_turn_detected():
        """Handle turn detection"""
        logger.debug("Turn detected - user finished speaking")
    
    # Start the session
    logger.info("Starting agent session...")
    await session.start()
    
    logger.info("✓ Agent is now active and listening")


def main():
    """Main entry point"""
    
    # Check environment variables
    required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set:")
        for var in missing_vars:
            logger.error(f"  export {var}=<your-value>")
        return
    
    # Check model path
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at: {MODEL_PATH}")
        logger.error("Please:")
        logger.error("1. Train your model: cd ../eou_model && python scripts/train.py")
        logger.error("2. Convert to ONNX: python scripts/convert_to_onnx.py")
        logger.error("3. Quantize: python scripts/quantize_model.py")
        logger.error("4. Set ARABIC_EOU_MODEL_PATH environment variable")
        return
    
    logger.info("="*70)
    logger.info("Arabic Voice Agent")
    logger.info("="*70)
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"EOU threshold: {UNLIKELY_THRESHOLD}")
    logger.info(f"LiveKit URL: {os.getenv('LIVEKIT_URL')}")
    logger.info("="*70)
    
    # Run the agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )


if __name__ == "__main__":
    main()
