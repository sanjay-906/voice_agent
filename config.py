import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import List
load_dotenv()


class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000"
    ]
    LOG_LEVEL: str = "INFO"

    CLINIC_NAME: str = "Medical Center"
    SPECIALTY: str = "Primary Care"
    GREETING_STYLE: str = "warm"  # "warm", "professional", "friendly"
    VOICE_MODEL: str = "Puck"  # "Charon", "Kore", "Fenrir", "Aoede"

    FORMAT: str = "pcm"
    CHANNELS: int = 1
    SEND_SAMPLE_RATE: int = 16000
    RECEIVE_SAMPLE_RATE: int = 24000

    MODEL: str = "models/gemini-2.5-flash-native-audio-preview-09-2025"
    SUMMARY_MODEL: str = "models/gemini-2.5-flash"

    class Config:
        env_file: str = ".env"


settings = Settings()
