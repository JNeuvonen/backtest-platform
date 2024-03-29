import asyncio

from fastapi import APIRouter, WebSocket
from log import get_logger

logger = get_logger()
router = APIRouter()


@router.websocket("/subscribe-log")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.add_websocket_connection(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            await websocket.send_text("health")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.remove_websocket_connection(websocket)
        await websocket.close()
        logger.info("WebSocket connection closed")
