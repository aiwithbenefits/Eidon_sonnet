#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the FastAPI app
echo "Starting FastAPI server..."
python3 -m uvicorn app_backend:app --host 0.0.0.0 --port 8000 > logs/fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "FastAPI server started with PID: $FASTAPI_PID"

# Start the capture service in a separate terminal window
echo "Starting capture service..."
cat > start_capture.py << 'EOF'
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
EOF

python3 start_capture.py > logs/capture_service.log 2>&1 &
CAPTURE_PID=$!
echo "Capture service started with PID: $CAPTURE_PID"

# Start Cloudflare tunnel
echo "Starting Cloudflare tunnel..."
cloudflared tunnel --config cloudflared.yml run > logs/cloudflared.log 2>&1 &
CLOUDFLARE_PID=$!
echo "Cloudflare tunnel started with PID: $CLOUDFLARE_PID"

# Function to clean up processes
cleanup() {
    echo "Shutting down processes..."
    kill $FASTAPI_PID $CAPTURE_PID $CLOUDFLARE_PID 2>/dev/null
    exit 0
}

# Set up trap to catch termination signals
trap cleanup INT TERM

# Keep script running
echo "\nAll services started!"
echo "FastAPI PID: $FASTAPI_PID"
echo "Capture Service PID: $CAPTURE_PID"
echo "Cloudflare Tunnel PID: $CLOUDFLARE_PID"
echo "\nPress Ctrl+C to stop all services\n"

# Keep script running until interrupted
echo "Services are running. Press Ctrl+C to stop all services."

# Simple infinite loop that can be interrupted
while true; do
    sleep 1
done

cleanup
