"""
LiveKit Arabic Turn Detector Plugin
====================================

A LiveKit plugin for detecting end-of-utterance in Arabic conversations.

Example:
    ```python
    from livekit.plugins.arabic_turn_detector import ArabicTurnDetector
    from livekit.agents import AgentSession
    
    # Create turn detector
    turn_detector = ArabicTurnDetector(
        model_path="./models/eou_model_quantized.onnx",
        unlikely_threshold=0.7
    )
    
    # Use in agent session
    async def entrypoint(ctx: JobContext):
        await ctx.connect()
        
        session = AgentSession(
            turn_detector=turn_detector,
            # ... other configurations
        )
        
        await session.start()
    ```
"""

from .arabic import ArabicTurnDetector, _EOURunnerAr
from .version import __version__

__all__ = [
    "ArabicTurnDetector",
    "_EOURunnerAr",
    "__version__",
]
