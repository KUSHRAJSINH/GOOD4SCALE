import asyncio
import logging
import os
import json
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.processors.audio.vad_processor import VADProcessor

from agent.llm import GeminiConversation
from agent.tts_service import EdgeTTSService

log = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")

GREETING = (
    "Hello! Thank you for calling BrightBox support. "
    "I'm your virtual assistant and I'm here to help with questions about your subscription, "
    "shipping, billing, or anything else BrightBox related. "
    "How can I help you today?"
)

SILENCE_PROMPT = "Are you still there? Feel free to ask me anything about BrightBox."


class BrightBoxRAGProcessor(FrameProcessor):
    def __init__(self, conversation: GeminiConversation, **kwargs):
        super().__init__(**kwargs)
        self._conversation = conversation
        self._silence_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if not user_text:
                return

            log.info(f"STT: {user_text!r}")
            self._silence_count = 0

            try:
                response_text, is_ending = await self._conversation.respond(user_text)
                log.info(f"LLM response: {response_text!r} (ending={is_ending})")

                await self.push_frame(TextFrame(text=response_text), direction)

                if is_ending:
                    await asyncio.sleep(4)
                    await self.push_frame(EndFrame(), direction)

            except Exception as exc:
                log.error(f"RAG/LLM processing error: {exc}", exc_info=True)
                fallback = "I'm sorry, I'm having a little trouble right now. Please try again."
                await self.push_frame(TextFrame(text=fallback), direction)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


async def run_brightbox_pipeline(websocket) -> None:
    conversation = GeminiConversation()

    data = await websocket.receive_text()
    message = json.loads(data)
    
    if message.get("event") == "connected":
        data = await websocket.receive_text()
        message = json.loads(data)
        
    if message.get("event") != "start":
        log.error(f"Expected start event, got: {message.get('event')}")
        return
        
    stream_sid = message["streamSid"]
    call_sid = message["start"]["callSid"]

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN")
    )
    
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    stt = WhisperSTTService(
        device="cpu",
        settings=WhisperSTTService.Settings(
            model=Model(WHISPER_MODEL_SIZE).value,
        ),
    )

    rag_llm = BrightBoxRAGProcessor(conversation=conversation)
    tts = EdgeTTSService(voice=EDGE_TTS_VOICE)

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            rag_llm,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        log.info("Twilio WebSocket connected — sending greeting")
        await task.queue_frame(TextFrame(text=GREETING))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        log.info("Twilio WebSocket disconnected")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)
    log.info("Pipeline completed for this call")
