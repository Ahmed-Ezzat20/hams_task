"""
LiveKit Voice Agent with Arabic Turn Detection
===============================================

Voice assistant with custom Arabic End-of-Utterance detection for improved
conversation flow in Arabic language interactions.

Features:
- Arabic STT (Speech-to-Text) via ElevenLabs
- Custom Arabic EOU turn detection (90% accuracy)
- Google Gemini LLM for responses
- Cartesia Sonic TTS for natural voice output
- Noise cancellation for better audio quality

Requirements:
    pip install livekit-agents livekit-plugins-elevenlabs livekit-plugins-google \\
                livekit-plugins-silero livekit-plugins-noise-cancellation \\
                onnxruntime transformers python-dotenv

Environment Variables (.env.local):
    LIVEKIT_URL=wss://your-server.livekit.cloud
    LIVEKIT_API_KEY=your_api_key
    LIVEKIT_API_SECRET=your_api_secret
    ELEVENLABS_API_KEY=your_elevenlabs_key
    GOOGLE_API_KEY=your_google_key

Usage:
    # Development mode (console)
    python agent.py dev
    
    # Production mode
    python agent.py start
"""

import logging
import os
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
    inference,
)

# Import specific plugins
from livekit.plugins import noise_cancellation, silero, elevenlabs, google

# Import custom Arabic EOU detector plugin
from arabic_turn_detector_plugin import ArabicEOUDetector

# Configure logging to show DEBUG messages (required to see EOU probability logs)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("agent")

# Load environment variables
load_dotenv(".env.local")


class Assistant(Agent):
    """Voice assistant with Arabic turn detection support"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.

You eagerly assist users with their questions by providing information from your extensive knowledge.

Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.

You are curious, friendly, and have a sense of humor.""",
        )


# Create the server instance
server = AgentServer()


def prewarm(proc: JobProcess):
    """Pre-warm the server with necessary components"""
    logger.info("Prewarming models...")
    
    # Pre-load VAD model
    logger.info("Loading VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("✓ VAD model loaded")
    
    # Pre-load Arabic EOU detector
    logger.info("Loading Arabic EOU detector...")
    try:
        # Get model path from environment or use default
        model_path = os.getenv(
            "ARABIC_EOU_MODEL_PATH",
            "./eou_model/models/eou_model_quantized.onnx"
        )
        
        eou_detector = ArabicEOUDetector(
            model_path=model_path,
            confidence_threshold=0.7,  # Adjust based on your needs
        )
        
        proc.userdata["eou_detector"] = eou_detector
        logger.info(f"✓ Arabic EOU detector loaded")
        logger.info(f"  Model info: {eou_detector.get_model_info()}")
        
    except Exception as e:
        logger.warning(f"Failed to pre-warm EOU detector: {e}")
        logger.warning("Agent will run without custom turn detection")


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    """Main agent session handler"""
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Get pre-loaded components
    vad = ctx.proc.userdata.get("vad")
    eou_detector = ctx.proc.userdata.get("eou_detector")
    
    if eou_detector:
        logger.info(f"Using Arabic EOU detector with threshold: {eou_detector.confidence_threshold}")
    else:
        logger.warning("Arabic EOU detector not available, using default turn detection")
    
    # Create the agent session with Arabic EOU detection
    session = AgentSession(
        # Speech-to-Text (ElevenLabs with Arabic support)
        stt=elevenlabs.STT(
            language_code="ar",  # Arabic language
            use_realtime=False,  # ✅ FIXED: False prevents WebSocket errors
        ),
        
        # Large Language Model (Google Gemini)
        llm=google.LLM(
            model="models/gemini-2.5-flash-lite",
        ),
        
        # Text-to-Speech (Cartesia Sonic)
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Update with your preferred voice
        ),
        
        # Custom Arabic End-of-Utterance detection
        turn_detection=eou_detector,  # ✅ ACTIVATED: Using our custom detector
        
        # Voice Activity Detection (still recommended for robustness)
        vad=vad if vad else silero.VAD.load(),
        
        # Enable preemptive generation for faster responses
        preemptive_generation=True,
        
        # Optional: Set minimum endpointing delay for Arabic (slightly longer)
        min_endpointing_delay=0.8,
    )
    
    logger.info("AgentSession initialized with Arabic EOU detector")
    
    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )
    
    # Connect to the room
    await ctx.connect()
    
    logger.info(f"Agent connected to room: {ctx.room.name}")


if __name__ == "__main__":
    cli.run_app(server)
