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
            http_options={"api_version": "v1beta"},
            api_key=api_key
        )
        self.audio_in_queue = None
        self.audio_out_queue = None
        self.session = None
        self.websocket = None

        self.conversation_history: List[Dict[str, str]] = []
        self.current_assistant_turn = ""
        self.current_user_turn = ""

        self.latest_structured: Optional[Dict[str, Any]] = None

        self.session_id = str(uuid.uuid4())
        self.session_log_file = None

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
                self.audio_in_queue = asyncio.Queue()
                self.audio_out_queue = asyncio.Queue(maxsize=50)

                await websocket.send_json({
                    "type": "preflight",
                    "status": 200,
                    "message": "Connected to Agent"
                })

                tasks = [
                    asyncio.create_task(self._receive_from_ui()),
                    asyncio.create_task(self._send_to_agent()),
                    asyncio.create_task(self._receive_from_agent()),
                    asyncio.create_task(self._send_to_ui())
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
        """Receives audio from UI and puts it into the `audio_out_queue` queue."""
        logger.info("Step 1: Receiving input from the user")
        try:
            while True:
                message = await self.websocket.receive()

                if "bytes" in message:
                    audio_chunk = message["bytes"]
                    await self.audio_out_queue.put({
                        "data": audio_chunk,
                        "mime_type": "audio/pcm"
                    })

                elif "text" in message:
                    data = json.loads(message["text"])

                    if data.get("type") == "interrupt":
                        await self._interrupt()

                    elif data.get("type") == "end_session":
                        raise asyncio.CancelledError("User ended")

        except asyncio.CancelledError:
            raise

        except Exception as e:
            logger.error(f"Error in `_receive_from_ui` ({e.__class__.__name__}):{e}")
            raise

    async def _send_to_agent(self):
        """Takes the audio from `audio_out_queue` and sends it to the agent"""
        logger.info("Step 2: Sending to Agent")

        try:
            while True:
                audio_data = await self.audio_out_queue.get()
                await self.session.send(input=audio_data)

        except asyncio.CancelledError:
            raise

        except Exception as e:
            logger.error(f"Error in `_send_to_agent`({e.__class__.__name__}):{e})")
            raise

    async def _receive_from_agent(self):
        """Takes output from agent and puts it in `audio_in_queue`"""
        logger.info("Step 3: Receiving from agent")
        try:
            while True:
                turn = self.session.receive()
                async for response in turn:
                    if hasattr(response, "server_content") and response.server_content:
                        if hasattr(response.server_content, "input_transcription"):
                            input_transcript = response.server_content.input_transcription
                            if input_transcript:
                                user_text = getattr(input_transcript, "text", "")
                                if user_text:
                                    self.current_user_turn += user_text
                                    await self.audio_in_queue.put({
                                        "type": "text",
                                        "role": "user",
                                        "text": user_text
                                    })

                        if hasattr(response.server_content, "output_transcription"):
                            output_transcript = response.server_content.output_transcription
                            if output_transcript:
                                ai_text = getattr(output_transcript, "text", "")
                                if ai_text:
                                    self.current_assistant_turn += ai_text
                                    await self.audio_in_queue.put({
                                        "type": "text",
                                        "role": "assistant",
                                        "text": ai_text
                                    })

                        if hasattr(response.server_content, "turn_complete") and response.server_content.turn_complete:
                            self._finalize_turn()
                            await self.audio_in_queue.put({
                                "type": "turn_complete"
                            })

                    if data := response.data:
                        await self.audio_in_queue.put({
                            "type": "audio",
                            "data": data
                        })

                    if text := response.text:
                        self.current_assistant_turn += text
                        await self.audio_in_queue.put({
                            "type": "text",
                            "role": "assistant",
                            "text": text
                        })

                    if response.tool_call and response.tool_call.function_calls:
                        for func_call in response.tool_call.function_calls:
                            if func_call.name == "call_human_agent":
                                await self.audio_in_queue.put({
                                    "type": "function_call",
                                    "function_name": "call_human_agent"
                                })

        except asyncio.CancelledError:
            logger.info("Stopped receiving from agent")
            raise

        except Exception as e:
            logger.error(f"Error in `_receive_from_agent`({e.__class__.__name__}):{e})")
            raise

    async def _send_to_ui(self):
        """Streams `audio_in_queue` contents to the UI"""
        logger.info("Step 4: Sending to the UI")

        try:
            while True:
                response = await self.audio_in_queue.get()

                if response["type"] == "audio":
                    await self.websocket.send_bytes(response["data"])

                elif response["type"] == "text":
                    await self.websocket.send_json({
                        "type": "transcript",
                        "role": response["role"],
                        "text": response["text"]
                    })

                elif response["type"] == "function_call" and response["function_name"] == "call_human_agent":
                    # what should i do?
                    pass

                elif response["type"] == "turn_complete":
                    await self.websocket.send_json({"type": "turn_complete"})

        except asyncio.CancelledError:
            raise

        except Exception as e:
            logger.error(f"Error in `_send_to_ui`({e.__class__.__name__}):{e})")
            raise

    async def _interrupt(self):
        """Clears audio queue to stop playback"""

        if self.session:

            while not self.audio_in_queue.empty():
                try:
                    self.audio_in_queue.get_nowait()

                except asyncio.QueueEmpty:
                    break

    def _get_system_instructions(self) -> str:
        """Defines AI personality and conversation flow"""

        greeting_tones = {
            "warm": "Be warm, empathetic, and caring in your tone",
            "professional": "Maintain a professional and clinical tone throughout",
            "friendly": "Be friendly, approachable, and conversational",
        }
        tone = greeting_tones.get(settings.GREETING_STYLE, greeting_tones["warm"])

        return f"""You are a customer support agent for Hutch telecommunications.

Your role is to resolve user's queries.

CONVERSATION STYLE:
- {tone}
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
