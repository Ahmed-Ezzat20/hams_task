"""
LiveKit Agent with Custom Arabic Turn Detection

This agent uses the custom Arabic EOU (End-of-Utterance) turn detector
instead of the default multilingual turn detector for improved performance
with Arabic conversations.

Requirements:
    - livekit-agents
    - livekit-plugins-elevenlabs
    - livekit-plugins-google
    - livekit-plugins-silero
    - livekit-plugins-noise-cancellation
    - Custom Arabic turn detector plugin

Environment Variables:
    - LIVEKIT_URL: LiveKit server URL
    - LIVEKIT_API_KEY: LiveKit API key
    - LIVEKIT_API_SECRET: LiveKit API secret
    - ELEVENLABS_API_KEY: ElevenLabs API key
    - GOOGLE_API_KEY: Google API key (for Gemini)
    - ARABIC_EOU_MODEL_PATH: Path to Arabic EOU ONNX model

Usage:
    python agent_with_arabic_turn_detector.py start
"""

import logging
import os
from pathlib import Path
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

# Import custom Arabic turn detector
from livekit_plugins_arabic_turn_detector.livekit.plugins.arabic_turn_detector import (
    ArabicTurnDetector,
)

logger = logging.getLogger("agent")

# Load environment variables
load_dotenv(".env.local")


class ArabicAssistant(Agent):
    """
    Arabic Voice AI Assistant with custom turn detection

    This assistant is optimized for Arabic conversations with:
    - Custom Arabic EOU turn detection
    - Arabic STT (Speech-to-Text)
    - Multilingual LLM (Gemini)
    - High-quality TTS
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""أنت مساعد صوتي ذكي باللغة العربية. المستخدم يتفاعل معك عبر الصوت.
            
You are a helpful Arabic voice AI assistant. The user is interacting with you via voice.
You assist users with their questions by providing information from your extensive knowledge.
Your responses are concise, to the point, and natural for spoken conversation.
You respond in Arabic when the user speaks Arabic, and in English when they speak English.
Avoid complex formatting, emojis, asterisks, or other symbols in your responses.
You are friendly, professional, and helpful.""",
        )


# Increase proc initialization timeout (default is 10 s) to allow model downloads / heavy loads
server = AgentServer()


def prewarm(proc: JobProcess):
    """
    Prewarm function to load models before agent starts

    This loads:
    - VAD (Voice Activity Detection) model
    - Arabic EOU turn detection model (if path provided)
    """
    logger.info("Prewarming models...")

    # Load VAD model
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("✓ VAD model loaded")

    # Load Arabic turn detector if model path is provided
    model_path = os.getenv("ARABIC_EOU_MODEL_PATH")
    if model_path and Path(model_path).exists():
        try:
            proc.userdata["arabic_turn_detector"] = ArabicTurnDetector(
                model_path=model_path,
                unlikely_threshold=0.7,  # Adjust based on your needs
            )
            logger.info(f"✓ Arabic turn detector loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load Arabic turn detector: {e}")
            logger.warning("Will use default turn detector as fallback")
            proc.userdata["arabic_turn_detector"] = None
    else:
        logger.warning("ARABIC_EOU_MODEL_PATH not set or file not found")
        logger.warning("Will use default turn detector as fallback")
        proc.userdata["arabic_turn_detector"] = None


server.setup_fnc = prewarm


@server.rtc_session()
async def arabic_agent(ctx: JobContext):
    """
    Main agent function with Arabic turn detection

    This creates an agent session with:
    - Arabic STT (ElevenLabs with Arabic support)
    - Gemini LLM (multilingual)
    - High-quality TTS (Cartesia Sonic)
    - Custom Arabic turn detector
    - VAD for voice activity detection
    - Noise cancellation
    """
    ctx.log_context_fields = {"room": ctx.room.name}

    # Get Arabic turn detector from userdata (loaded in prewarm)
    arabic_turn_detector = ctx.proc.userdata.get("arabic_turn_detector")

    # Fallback to default if Arabic detector not available
    if arabic_turn_detector is None:
        logger.warning("Using default turn detector (Arabic detector not available)")
        from livekit.plugins.turn_detector.multilingual import MultilingualModel

        turn_detector = MultilingualModel()
    else:
        logger.info("Using custom Arabic turn detector")
        turn_detector = arabic_turn_detector

    # Create agent session
    session = AgentSession(
        # STT: ElevenLabs with Arabic support
        # Note: Changed use_realtime to False to avoid WebSocket connection issues
        stt=elevenlabs.STT(
            language_code="ar",  # Arabic language code
            use_realtime=False,  # Set to False to avoid connection issues
        ),
        # LLM: Google Gemini (supports Arabic)
        llm=google.LLM(
            model="models/gemini-2.5-flash-lite",
        ),
        # TTS: Cartesia Sonic (high quality)
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # Turn Detection: Custom Arabic EOU detector
        turn_detection=turn_detector,
        # VAD: Voice Activity Detection
        vad=ctx.proc.userdata["vad"],
        # Enable preemptive generation for faster responses
        preemptive_generation=True,
    )

    # Start the session
    await session.start(
        agent=ArabicAssistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Connect to the room
    await ctx.connect()

    logger.info(f"Agent connected to room: {ctx.room.name}")


if __name__ == "__main__":
    # Run the agent server
    cli.run_app(server)
