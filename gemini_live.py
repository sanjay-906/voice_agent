import asyncio
import inspect
import json
import logging
import uuid
from typing import Optional, Dict, Any, List

from google import genai
from google.genai import types
from google.genai import live as live_module

from config import settings

logger = logging.getLogger(__name__)


def _patch_websockets_for_headers() -> None:
    """google-genai expects websockets.ws_connect to accept additional_headers."""

    target = getattr(live_module, "ws_connect", None)
    if target is None:
        return

    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return

    if "additional_headers" in sig.parameters:
        return

    def connect_wrapper(*args, additional_headers=None, **kwargs):  # type: ignore[override]
        if additional_headers is not None and "extra_headers" not in kwargs:
            kwargs["extra_headers"] = additional_headers
        return target(*args, **kwargs)

    live_module.ws_connect = connect_wrapper  # type: ignore[assignment]


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

        self.latest_structured: Optional[Dict[str, Any]] = None

        self.session_id = str(uuid.uuid4())
        self.session_log_file = None
        self.end_turn_received = asyncio.Event()

    async def run(self, websocket):
        """Main session loop"""
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
                            description="Call this when you don't know the answer to the user query and want to pass on this conversation to the human agent."
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

                tasks = [
                    asyncio.create_task(self._receive_from_ui()),
                    asyncio.create_task(self._send_to_ui()),
                ]

                await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("Session cancelled")

        except Exception as e:
            logger.error(f"Error in `GeminiLiveSession` ({e.__class__.__name__}):{e}", exc_info=True)
            try:
                await websocket.send_json({
                    "type": "Error in `GeminiLiveSession`",
                    "status": 500,
                    "message": f"{e.__class__.__name__}:{e}"
                })

            except Exception:
                pass

    async def _receive_from_ui(self):
        """Receives audio from UI and sends it directly to the agent."""
        try:
            while True:
                message = await self.websocket.receive()

                if "bytes" in message:
                    if not self.session:
                        return

                    audio_chunk = message["bytes"]
                    await self.session.send_realtime_input(
                        media=types.Blob(
                            data=audio_chunk,
                            mime_type='audio/pcm;rate=16000'
                        )
                    )
                    await asyncio.sleep(0.02)

                elif "text" in message:
                    data = json.loads(message["text"])
                    if "content" in data:
                        await self.session.send_client_content(
                            turns=types.Content(
                                role='user',
                                parts=[types.Part(text=data["content"])]
                            )
                        )
                    if data.get("type") == "interrupt":
                        # TODO: handle interrupt
                        pass

                    elif data.get("type") == "end_session":
                        raise asyncio.CancelledError("User ended")

        except asyncio.CancelledError:
            raise

        except Exception as e:
            logger.error(
                f"Error in `_receive_from_ui` ({e.__class__.__name__}): {e}", exc_info=True
            )
            raise

    async def _send_to_ui(self):
        """Receives output from agent and streams directly to the UI"""
        try:
            async for response in self.session.receive():

                if response.go_away is not None:
                    logger.warning("Agent connection closing soon")

                server = getattr(response, "server_content", None)
                if server:
                    if getattr(server, "input_transcription", None):
                        user_text = getattr(server.input_transcription, "text", "")
                        if user_text:
                            self.current_user_turn += user_text
                            await self.websocket.send_json({
                                "type": "transcript",
                                "role": "user",
                                "text": user_text
                            })

                    if getattr(server, "output_transcription", None):
                        ai_text = getattr(server.output_transcription, "text", "")
                        if ai_text:
                            self.current_assistant_turn += ai_text
                            await self.websocket.send_json({
                                "type": "transcript",
                                "role": "assistant",
                                "text": ai_text
                            })

                    if getattr(server, "turn_complete", False):
                        self._finalize_turn()
                        await self.websocket.send_json({"type": "turn_complete"})

                if response.data:
                    await self.websocket.send_bytes(response.data)

                if response.text:
                    self.current_assistant_turn += response.text
                    await self.websocket.send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "text": response.text
                    })

                if response.tool_call and response.tool_call.function_calls:
                    for call in response.tool_call.function_calls:
                        if call.name == "call_human_agent":
                            await self.websocket.send_json({
                                "type": "handover",
                                "reason": "human_agent_requested"
                            })

        except asyncio.CancelledError:
            logger.info("Stopped receiving from agent")
            raise

        except Exception as e:
            logger.error(
                f"Error in `_receive_from_agent` ({e.__class__.__name__}): {e}",
                exc_info=True
            )
            raise

    def _get_system_instructions(self) -> str:
        """Defines AI personality and conversation flow"""
        return """You are a customer support agent for Hutch telecommunications.

Your role is to resolve user's queries.

CONVERSATION STYLE:
- Be friendly, approachable, and conversational
- Ask ONE clear question at a time
- Use simple language (avoid jargon)
- Provide a short verbal summary at the end

COMPLETION PROTOCOL:
1. Greet the user
2. Ask for his/her queries
3. Try your best resolving them
4. If you are unable to resolve any user queries, you may hand over this conversation to a human agent, with the permission of the user, by calling the `call_human_agent()` tool
5. Summarize the conversation to the user before exiting

Keep responses brief (1-2 sentences) except for the final summary."""

    def _finalize_turn(self):
        """
        Finalizes accumulated text chunks into complete conversation turns
        """
        if self.current_assistant_turn:
            complete_text = self.current_assistant_turn.strip()
            if complete_text:
                self.conversation_history.append({
                    "role": "assistant",
                    "text": complete_text
                })
            self.current_assistant_turn = ""

        if self.current_user_turn:
            complete_text = self.current_user_turn.strip()
            if complete_text:
                self.conversation_history.append({
                    "role": "user",
                    "text": complete_text
                })
            self.current_user_turn = ""

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up session")
        if self.session:
            self.session = None
