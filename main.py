from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from gemini_live import GeminiLiveSession
from config import settings
from services import service, version, description

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=service,
    description=description,
    version=version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def root():
    """Shows service information"""
    return {
        "service": service,
        "version": version,
        "status": "active",
        "description": description
    }


@app.get("/health")
async def health_check():
    """For monitoring"""
    return {
        "status": "connected"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    main function for websocket_endpoint
    """
    await websocket.accept()
    logger.info("Websocket connection accepted from client")

    google_api_key = settings.GEMINI_API_KEY

    if not google_api_key:
        logger.error("Google API key is missing")
        await websocket.send_json({
            "type": "error",
            "status": 404,
            "message": "API key NOT found"
        })
        await websocket.close()
        return

    session = GeminiLiveSession(api_key=google_api_key)

    try:
        logger.info("Starting live session")
        await session.run(websocket)

    except WebSocketDisconnect:
        logger.info("Websocket disconnected by client")

    except Exception as e:
        logger.error(f"Error in `websocket_endpoint` ({e.__class__.__name__}):{e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "Error in `websocket_endpoint`",
                "status": 500,
                "message": f"{e.__class__.__name__}:{e}"
            })

        except Exception:
            pass

    finally:
        await session.cleanup()
        logger.info("Websocket connection closed")


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
        loop="asyncio"
    )
