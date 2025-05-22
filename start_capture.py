import asyncio
import logging
from screen_capture_handler import ScreenCaptureService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/capture_service.log'),
        logging.StreamHandler()
    ]
)

async def main():
    queue = asyncio.Queue()
    service = ScreenCaptureService(queue)
    await service.start()
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
