import asyncio
import inspect
import json
import logging
import uuid
from typing import Dict, List

from fastapi import WebSocketDisconnect
from google import genai
from google.genai import types
from google.genai import live as live_module

from config import settings

logger = logging.getLogger(__name__)


def _patch_websockets_for_headers() -> None:
    target = getattr(live_module, "ws_connect", None)
    if target is None:
        return
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return
    if "additional_headers" in sig.parameters:
        return

    def connect_wrapper(*args, additional_headers=None, **kwargs):
        if additional_headers is not None and "extra_headers" not in kwargs:
            kwargs["extra_headers"] = additional_headers
        return target(*args, **kwargs)
    live_module.ws_connect = connect_wrapper


_patch_websockets_for_headers()


class GeminiLiveSession:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = genai.Client(
            http_options=types.HttpOptions(api_version="v1alpha"),
            api_key=api_key
        )
        self.session = None
        self.websocket = None
        self.conversation_history: List[Dict[str, str]] = []
        self.current_assistant_turn = ""
        self.current_user_turn = ""
        self.session_id = str(uuid.uuid4())

    async def run(self, websocket):
        self.websocket = websocket

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=settings.VOICE_MODEL
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self._get_system_instructions())]
            ),
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="call_human_agent",
                            description="Call this when the user explicitly asks for a human agent."
                        )
                    ]
                )
            ]
        )

        logger.info("Connecting to AI Agent...")
        try:
            async with self.client.aio.live.connect(model=settings.MODEL, config=config) as session:
                self.session = session

                await websocket.send_json({
                    "type": "preflight",
                    "status": 200,
                    "message": "Connected to Agent"
                })

                await asyncio.gather(
                    self._receive_from_ui(),
                    self._send_to_ui(),
                    return_exceptions=False
                )

        except asyncio.CancelledError:
            logger.info("Session cancelled normally")
        except Exception as e:
            logger.error(f"Error in GeminiLiveSession: {e}", exc_info=True)
        finally:
            logger.info("Exiting GeminiLiveSession run loop")

    async def _receive_from_ui(self):
        """Receives audio/text from UI and pushes to Gemini"""
        try:
            while True:
                message = await self.websocket.receive()

                if "bytes" in message:
                    if not self.session:
                        continue

                    await self.session.send_realtime_input(
                        media=types.Blob(
                            data=message["bytes"],
                            mime_type='audio/pcm;rate=16000'
                        )
                    )

                elif "text" in message:
                    data = json.loads(message["text"])
                    if "content" in data and self.session:
                        await self.session.send_client_content(
                            turns=types.Content(
                                role='user',
                                parts=[types.Part(text=data["content"])]
                            ),
                            turn_complete=True
                        )

                    if data.get("type") == "end_session":
                        raise asyncio.CancelledError("User ended session")

        except WebSocketDisconnect:
            logger.info("Client disconnected WebSocket")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error receiving from UI: {e}")
            raise

    async def _send_to_ui(self):
        """Receives from Gemini and pushes to UI"""
        try:
            while True:
                async for response in self.session.receive():
                    if response.tool_call and response.tool_call.function_calls:
                        for call in response.tool_call.function_calls:
                            if call.name == "call_human_agent":
                                await self.websocket.send_json({
                                    "type": "handover",
                                    "reason": "human_agent_requested"
                                })
                                logger.info("Tool call came, so ended")
                                return

                    server_content = getattr(response, "server_content", None)
                    if server_content:
                        if getattr(server_content, "input_transcription", None):
                            text = server_content.input_transcription.text
                            if text:
                                await self.websocket.send_json({
                                    "type": "transcript",
                                    "role": "user",
                                    "text": text
                                })

                        if getattr(server_content, "output_transcription", None):
                            text = server_content.output_transcription.text
                            if text:
                                await self.websocket.send_json({
                                    "type": "transcript",
                                    "role": "assistant",
                                    "text": text
                                })

                        if getattr(server_content, "turn_complete", False):
                            await self.websocket.send_json({"type": "turn_complete"})

                    if response.data:
                        await self.websocket.send_bytes(response.data)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error sending to UI: {e}", exc_info=True)
            raise

    def _get_system_instructions(self) -> str:
        return """You are a customer support agent for Hutch Telecommunications.
        If you cannot answer, call the function 'call_human_agent'."""

    async def cleanup(self):
        self.session = None
