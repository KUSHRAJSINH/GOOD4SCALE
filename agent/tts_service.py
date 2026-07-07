"""
agent/tts_service.py — Custom Pipecat TTS service using Microsoft edge-tts.

No API key or credit card required. Uses Microsoft Edge's neural TTS voices.
Audio pipeline: edge-tts (MP3) → pydub (PCM 16 kHz mono) → Pipecat frames
"""

import asyncio
import io
import logging
from typing import AsyncGenerator

import edge_tts
import os
import sys

# Dynamically add winget ffmpeg to PATH if not present to avoid system restart
winget_packages_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
if os.path.exists(winget_packages_dir):
    for root, dirs, files in os.walk(winget_packages_dir):
        if "ffmpeg.exe" in files:
            bin_dir = root
            if bin_dir not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + bin_dir
            break

from pydub import AudioSegment

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

log = logging.getLogger(__name__)

# Pipecat expects PCM at 16 kHz, 16-bit, mono for standard transport use
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2   # 16-bit = 2 bytes
CHUNK_BYTES = 8192         # ~256ms @ 16kHz 16-bit mono


class EdgeTTSService(TTSService):
    """
    Pipecat TTS service backed by Microsoft edge-tts (free, no CC needed).

    Args:
        voice: One of the Edge TTS neural voices.
               Run `edge-tts --list-voices` to see all options.
               Default: en-US-JennyNeural (warm, clear, customer-service feel)
    """

    def __init__(self, voice: str = "en-US-JennyNeural", **kwargs):
        super().__init__(**kwargs)
        self._voice = voice
        log.info(f"EdgeTTSService initialized with voice: {voice}")

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """
        Convert text → PCM audio frames using edge-tts.
        Yields: TTSStartedFrame, TTSAudioRawFrame(s), TTSStoppedFrame
        """
        log.debug(f"TTS synthesizing: {text!r}")

        try:
            # Collect MP3 audio from edge-tts
            communicate = edge_tts.Communicate(text, self._voice)
            mp3_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bytes += chunk["data"]

            if not mp3_bytes:
                log.warning("edge-tts returned no audio bytes")
                return

            # Convert MP3 → PCM (16 kHz, 16-bit, mono) using pydub
            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            audio_segment = (
                audio_segment
                .set_frame_rate(TARGET_SAMPLE_RATE)
                .set_channels(TARGET_CHANNELS)
                .set_sample_width(TARGET_SAMPLE_WIDTH)
            )
            pcm_data = audio_segment.raw_data

            yield TTSStartedFrame()

            # Stream PCM in chunks for smooth real-time delivery
            for i in range(0, len(pcm_data), CHUNK_BYTES):
                chunk = pcm_data[i : i + CHUNK_BYTES]
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=TARGET_SAMPLE_RATE,
                    num_channels=TARGET_CHANNELS,
                )
                # Yield control so the event loop can process other frames
                await asyncio.sleep(0)

            yield TTSStoppedFrame()

        except Exception as exc:
            log.error(f"EdgeTTSService error: {exc}", exc_info=True)
