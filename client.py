#!/usr/bin/env python3
"""
Gemini Live Test Client
Tests the FastAPI WebSocket endpoint with multiple modes:
- Audio file playback (PCM/WAV)
- Real-time microphone input
- Interactive command controls
- Silence generation for basic connectivity tests
"""

import asyncio
import json
import logging
import argparse
import sys
import wave
import time
from pathlib import Path
from typing import Optional, BinaryIO

# Optional dependencies (only required for mic mode)
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GeminiLiveClient")

# Constants
DEFAULT_SAMPLE_RATE = 24000  # Required by Gemini Live API
DEFAULT_CHUNK_SIZE = 3200    # Bytes per audio chunk
SILENCE_DURATION = 2.0       # Seconds of silence to send
AUDIO_TIMEOUT = 30           # Seconds to wait for audio responses

global audio_out, speaker

audio_out = pyaudio.PyAudio()
speaker = audio_out.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=DEFAULT_SAMPLE_RATE,
    output=True,
    frames_per_buffer=DEFAULT_CHUNK_SIZE // 2
)

logger.info("🔊 Speaker output active")


class AudioFileReader:
    """Reads audio files (WAV/PCM) with format validation"""

    def __init__(self, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.wf: Optional[BinaryIO] = None
        self.is_wav = self.file_path.suffix.lower() == '.wav'
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.bytes_per_sec = DEFAULT_SAMPLE_RATE * 2  # 16-bit mono

    def __enter__(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")

        if self.is_wav:
            self.wf = wave.open(str(self.file_path), 'rb')
            if self.wf.getnchannels() != 1:
                raise ValueError("Only mono audio supported")
            if self.wf.getsampwidth() != 2:
                raise ValueError("Only 16-bit audio supported")
            if self.wf.getframerate() != DEFAULT_SAMPLE_RATE:
                raise ValueError(f"Sample rate must be {DEFAULT_SAMPLE_RATE} Hz")
            self.bytes_per_sec = self.wf.getframerate() * self.wf.getsampwidth()
        else:
            # Raw PCM - assume 16kHz 16-bit mono
            self.wf = open(self.file_path, 'rb')

        logger.info(f"Loaded audio: {self.file_path.name} ({self._get_duration():.1f}s)")
        return self

    def __exit__(self, *args):
        if self.wf:
            self.wf.close()

    def _get_duration(self) -> float:
        if self.is_wav:
            return self.wf.getnframes() / self.wf.getframerate()
        return (self.file_path.stat().st_size) / (DEFAULT_SAMPLE_RATE * 2)

    def read_chunk(self) -> Optional[bytes]:
        if not self.wf:
            return None
        chunk = self.wf.read(self.chunk_size if self.is_wav else self.chunk_size)
        return chunk if chunk else None

    @property
    def chunk_duration(self) -> float:
        """Seconds per chunk for realistic playback timing"""
        return self.chunk_size / self.bytes_per_sec


# class MicAudioStream:
#     """Streams audio from microphone (requires PyAudio)"""

#     def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
#         if not PYAUDIO_AVAILABLE:
#             raise RuntimeError("PyAudio required for mic mode. Install with: pip install pyaudio")

#         self.chunk_size = chunk_size
#         self.p = pyaudio.PyAudio()
#         self.stream = None

#     def __enter__(self):
#         try:
#             self.stream = self.p.open(
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=DEFAULT_SAMPLE_RATE,
#                 input=True,
#                 frames_per_buffer=self.chunk_size // 2,  # 2 bytes per sample
#                 stream_callback=self._callback
#             )
#             self.stream.start_stream()
#             logger.info("🎤 Microphone active (press Ctrl+C to stop)")
#             return self
#         except Exception as e:
#             self.p.terminate()
#             raise RuntimeError(f"Failed to open microphone: {e}")

#     def __exit__(self, *args):
#         if self.stream:
#             self.stream.stop_stream()
#             self.stream.close()
#         self.p.terminate()

#     def _callback(self, in_data, frame_count, time_info, status):
#         # Callback runs in separate thread - we'll use queue in main loop
#         return (in_data, pyaudio.paContinue)

#     async def read_chunk(self) -> Optional[bytes]:
#         """Non-blocking chunk read"""
#         if not self.stream or not self.stream.is_active():
#             return None
#         try:
#             return self.stream.read(self.chunk_size // 2, exception_on_overflow=False)
#         except Exception as e:
#             logger.warning(f"Microphone read error: {e}")
#             return None
class MicAudioStream:
    """Streams audio from microphone (blocking read, no callback)"""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio required for mic mode")

        self.chunk_size = chunk_size
        self.p = pyaudio.PyAudio()
        self.stream = None

    def __enter__(self):
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=DEFAULT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.chunk_size // 2  # frames, not bytes
            )
            logger.info("🎤 Microphone active (press Ctrl+C to stop)")
            return self
        except Exception as e:
            self.p.terminate()
            raise RuntimeError(f"Failed to open microphone: {e}")

    def __exit__(self, *args):
        if self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    async def read_chunk(self) -> Optional[bytes]:
        if not self.stream or not self.stream.is_active():
            return None

        try:
            chunk = self.stream.read(
                self.chunk_size // 2,
                exception_on_overflow=False
            )

            # import struct
            # samples = struct.unpack("<" + "h" * (len(chunk) // 2), chunk)
            # peak = max(abs(s) for s in samples)

            # logger.info(f"🎙 Mic peak: {peak}")  # use INFO so you see it

            return chunk

        except Exception as e:
            logger.warning(f"Microphone read error: {e}")
            return None


async def send_audio_chunks(websocket, audio_source, mode: str):
    """Sends audio chunks with realistic timing"""
    chunk_count = 0
    start_time = time.time()

    if mode == "file":
        with audio_source as reader:
            while True:
                chunk = reader.read_chunk()
                if not chunk:
                    break

                await websocket.send(chunk)
                chunk_count += 1
                # Simulate real-time transmission
                await asyncio.sleep(reader.chunk_duration)

    elif mode == "mic":
        with audio_source as mic:
            logger.info("Sending live microphone audio...")
            try:
                while True:
                    chunk = await mic.read_chunk()
                    if chunk and len(chunk) > 0:
                        await websocket.send(chunk)
                        chunk_count += 1
                        # Maintain real-time flow
                        await asyncio.sleep(mic.chunk_size / (DEFAULT_SAMPLE_RATE * 2))
            except KeyboardInterrupt:
                logger.info("\n🛑 Stopping microphone capture")

    elif mode == "silence":
        total_bytes = int(SILENCE_DURATION * DEFAULT_SAMPLE_RATE * 2)
        bytes_sent = 0
        chunk = b'\x00' * DEFAULT_CHUNK_SIZE

        logger.info(f"Sending {SILENCE_DURATION}s of silence...")
        while bytes_sent < total_bytes:
            send_size = min(DEFAULT_CHUNK_SIZE, total_bytes - bytes_sent)
            await websocket.send(chunk[:send_size])
            bytes_sent += send_size
            chunk_count += 1
            await asyncio.sleep(DEFAULT_CHUNK_SIZE / (DEFAULT_SAMPLE_RATE * 2))

    elapsed = time.time() - start_time
    logger.info(f"✅ Sent {chunk_count} audio chunks ({elapsed:.1f}s)")


async def receive_messages(websocket):
    """Processes all incoming messages from server"""
    audio_bytes_received = 0
    session_active = True

    while session_active:
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=AUDIO_TIMEOUT)

            if isinstance(message, bytes):
                speaker.write(message)
                audio_bytes_received += len(message)
                logger.debug(f"[AUDIO] Played {len(message)} bytes")
                continue

            # Process JSON messages
            try:
                data = json.loads(message)
                msg_type = data.get("type", "unknown")

                if msg_type == "preflight":
                    logger.info(f"🚀 {data.get('message', 'Connected')}")

                elif msg_type == "transcript":
                    role = data.get("role", "unknown").upper()
                    text = data.get("text", "")
                    logger.info(f"[{role}] {text}")

                elif msg_type == "extracted_data":
                    logger.info("💊 MEDICAL DATA EXTRACTED:")
                    print(json.dumps(data["data"], indent=2, ensure_ascii=False))

                elif msg_type == "intake_complete":
                    logger.info(f"✅ INTAKE COMPLETE: {data.get('message')}")
                    session_active = False

                elif msg_type == "turn_complete":
                    logger.debug("🔄 Turn completed")

                elif "error" in msg_type.lower() or data.get("status") == 500:
                    logger.error(f"❌ SERVER ERROR: {data}")
                    session_active = False

                else:
                    logger.info(f"[{msg_type}] {data}")

            except json.JSONDecodeError:
                logger.warning(f"Non-JSON message: {message[:100]}")

        except asyncio.TimeoutError:
            logger.warning("⚠️  Audio timeout - server may be processing")
        except Exception as e:
            if "code = 1000" not in str(e):  # Normal closure
                logger.error(f"Receive error: {e}")
            session_active = False

    logger.info(f"Session ended. Total audio received: {audio_bytes_received} bytes")
    return audio_bytes_received > 0


async def run_client(args):
    """Main client workflow"""
    uri = f"ws://{args.host}:{args.port}/ws"
    logger.info(f"🔌 Connecting to {uri}")

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
            # Wait for preflight
            try:
                preflight = await asyncio.wait_for(websocket.recv(), timeout=10)
                preflight_data = json.loads(preflight)
                if preflight_data.get("status") != 200:
                    logger.error(f"Preflight failed: {preflight_data}")
                    return False
                logger.info("✅ Preflight successful")
            except Exception as e:
                logger.error(f"Preflight error: {e}")
                return False

            # Start receiver task
            receiver_task = asyncio.create_task(receive_messages(websocket))

            # Prepare audio source
            try:
                if args.mode == "file":
                    if not args.audio_file:
                        raise ValueError("Audio file required for file mode")
                    audio_source = AudioFileReader(args.audio_file, DEFAULT_CHUNK_SIZE)
                elif args.mode == "mic":
                    if not PYAUDIO_AVAILABLE:
                        raise ValueError("PyAudio not installed. Use file mode or install pyaudio")
                    audio_source = MicAudioStream(DEFAULT_CHUNK_SIZE)
                else:  # silence mode
                    audio_source = None

                # Send audio in background
                if args.mode in ["file", "mic"]:
                    asyncio.create_task(send_audio_chunks(websocket, audio_source, args.mode))
                else:
                    # Silence mode: send after short delay
                    await asyncio.sleep(0.5)
                    await send_audio_chunks(websocket, None, "silence")

                    # Send interrupt after silence
                    await asyncio.sleep(1.0)
                    await websocket.send(json.dumps({"type": "interrupt"}))
                    logger.info("⏸️  Sent INTERRUPT command")

                    # End session after brief wait
                    await asyncio.sleep(2.0)
                    await websocket.send(json.dumps({"type": "end_session"}))
                    logger.info("⏹️  Sent END SESSION command")

            except Exception as e:
                logger.error(f"Audio source error: {e}")
                await websocket.send(json.dumps({"type": "end_session"}))
                receiver_task.cancel()
                return False

            # Wait for session completion
            try:
                await asyncio.wait_for(receiver_task, timeout=60)
            except asyncio.TimeoutError:
                logger.warning("⚠️  Session timeout exceeded")
                receiver_task.cancel()

            return True

    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Gemini Live API Test Client")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--file", dest="mode", action="store_const", const="file", help="Send audio from file (specify with --audio-file)")
    mode_group.add_argument("--mic", dest="mode", action="store_const", const="mic", help="Stream from microphone (requires PyAudio)")
    mode_group.add_argument("--silence", dest="mode", action="store_const", const="silence", help="Send silence + interrupt commands (basic connectivity test)")

    parser.add_argument("--audio-file", help="Path to WAV/PCM audio file (16kHz, 16-bit, mono)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def validate_dependencies(args):
    """Check required dependencies based on mode"""
    errors = []

    if not WEBSOCKETS_AVAILABLE:
        errors.append("❌ 'websockets' package required. Install with: pip install websockets")

    if args.mode == "mic" and not PYAUDIO_AVAILABLE:
        errors.append("❌ 'pyaudio' required for mic mode. Install with: pip install pyaudio")

    if args.mode == "file" and not args.audio_file:
        errors.append("❌ --audio-file required for file mode")

    if args.audio_file and not Path(args.audio_file).exists():
        errors.append(f"❌ Audio file not found: {args.audio_file}")

    if errors:
        for err in errors:
            logger.error(err)
        sys.exit(1)


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate setup
    validate_dependencies(args)

    # Run client
    logger.info(f"Starting client in {args.mode.upper()} mode")
    success = asyncio.run(run_client(args))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Quick dependency check before imports
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)

    main()
