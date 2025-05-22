import asyncio
import io
import logging
import os
import subprocess
import time
from PIL import Image, ImageGrab, UnidentifiedImageError
import io
import time
from typing import Optional, Dict, Any, Tuple

# Platform-specific imports
MACOS_LIBS_AVAILABLE = False
if __import__('platform').system() == "Darwin":
    try:
        from AppKit import NSWorkspace, NSScreen
        from Foundation import NSAppleScript
        import Quartz  # For screen coordinates, mouse location, and window info
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID
        )
        MACOS_LIBS_AVAILABLE = True
        logging.debug("Successfully imported all required macOS libraries")
    except ImportError as e:
        logging.warning(f"Failed to import required macOS libraries: {e}")
        logging.warning("macOS AppKit/Foundation/Quartz libraries not found. macOS specific features will be limited.")
        logging.info("To enable full macOS functionality, install pyobjc: pip install pyobjc pyobjc-framework-Cocoa pyobjc-framework-Quartz")

try:
    from pynput import mouse, keyboard # Cross-platform idle detection
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logging.warning("pynput library not found. Idle detection will be disabled.")


import config # Application configuration
from utils import get_current_timestamp_ms, is_macos, is_windows, is_linux

logger = logging.getLogger(__name__)

# --- macOS Specific Helper Functions ---
# Constants for window list options
kCGWindowListOptionAll = 0
kCGWindowListOptionOnScreenOnly = 1 << 0
kCGWindowListOptionOnScreenAboveWindow = 1 << 1
kCGWindowListOptionOnScreenBelowWindow = 1 << 2
kCGWindowListOptionIncludingWindow = 1 << 3
kCGWindowListExcludeDesktopElements = 1 << 4
kCGNullWindowID = 0

def _get_active_window_info_macos() -> Optional[Dict[str, Any]]:
    """Gets active application name and window title on macOS."""
    if not MACOS_LIBS_AVAILABLE:
        return {"app_name": "UnknownOSXApp", "window_title": "AppKit/Foundation Unavailable", "url": None}
    
    try:
        active_app = NSWorkspace.sharedWorkspace().activeApplication()
        app_name = active_app.get('NSApplicationName', 'UnknownApp') if active_app else "UnknownApp"
        pid = active_app.get('NSApplicationProcessIdentifier', None) if active_app else None

        if not pid:
            return {"app_name": app_name, "window_title": "No PID", "url": None}

        # Get all on-screen windows
        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID
        )
        
        if not window_list:
            return {"app_name": app_name, "window_title": "No Windows Found", "url": None}
        
        # Try to find the frontmost window for our app
        active_window_title = None
        for window in window_list:
            window_pid = window.get('kCGWindowOwnerPID')
            if window_pid == pid and window.get('kCGWindowLayer') == 0:  # Layer 0 is normal windows
                window_name = window.get('kCGWindowName', '')
                if window_name:  # Prefer windows with names
                    active_window_title = window_name
                    break
        
        # If we didn't find a window with a name, just use the first one
        if active_window_title is None and window_list:
            active_window_title = window_list[0].get('kCGWindowName', 'Untitled Window')
        
        # Fallback if we still don't have a title
        active_window_title = active_window_title or "No Window Title"
        
        # Try to get URL for browsers
        url = None
        if app_name.lower() in ['safari', 'google chrome', 'microsoft edge', 'firefox']:
            url = _get_browser_url_macos(app_name)
            
        return {
            "app_name": app_name,
            "window_title": active_window_title,
            "url": url
        }
    except Exception as e:
        logger.error(f"Error getting active window info (macOS): {e}", exc_info=True)
        return {"app_name": "ErrorApp", "window_title": "ErrorTitle", "url": None}


def _get_browser_url_macos(app_name: str) -> Optional[str]:
    """Gets URL from active tab of supported browsers on macOS."""
    if not MACOS_LIBS_AVAILABLE: return None
    if not app_name: return None
    supported_browsers = {
        "Safari": 'tell application "Safari" to get URL of front document',
        "Google Chrome": 'tell application "Google Chrome" to get URL of active tab of front window',
        "Chromium": 'tell application "Chromium" to get URL of active tab of front window',
        "Microsoft Edge": 'tell application "Microsoft Edge" to get URL of active tab of front window',
        "Brave Browser": 'tell application "Brave Browser" to get URL of active tab of front window',
        "Arc": 'tell application "Arc" to get URL of active tab of front window',
        "Firefox": 'tell application "Firefox" to get URL of active tab of front window'
    }
    script_source = supported_browsers.get(app_name)
    if not script_source: return None
    try:
        script = NSAppleScript.alloc().initWithSource_(script_source)
        result, error = script.executeAndReturnError_(None)
        if error: return None
        return result.stringValue() if result and result.stringValue() else None
    except Exception as e:
        logger.warning(f"Failed to get URL for {app_name} via AppleScript: {e}")
        return None

def _capture_screen_macos() -> Optional[bytes]:
    """Captures the main screen on macOS using multiple methods. Returns PNG image bytes."""
    
    # Method 1: Try using screencapture CLI first (most reliable on macOS)
    try:
        command = ["screencapture", "-x", "-t", "png", "-"]
        env = os.environ.copy()
        env["DISPLAY"] = ":0"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        stdout, stderr = process.communicate(timeout=5)
        logger.debug(f"screencapture stdout length: {len(stdout)} bytes")

        if process.returncode == 0 and stdout:
            logger.debug("Successfully captured screen using screencapture CLI")
            return stdout

        # Fallback debug path
        debug_file = "/tmp/debug_screenshot.png"
        debug_command = ["screencapture", "-x", "-t", "png", debug_file]
        process = subprocess.Popen(debug_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        process.communicate(timeout=5)

        if os.path.exists(debug_file):
            with open(debug_file, "rb") as f:
                img_data = f.read()
            os.remove(debug_file)
            if img_data:
                logger.debug("Successfully captured screen using screencapture debug file fallback")
                return img_data

        logger.warning(f"screencapture CLI failed. RC: {process.returncode}. Stderr: {stderr.decode(errors='ignore')}")
    except FileNotFoundError:
        logger.warning("screencapture command not found. Falling back to alternative methods.")
    except subprocess.TimeoutExpired:
        logger.warning("screencapture CLI command timed out. Falling back to alternative methods.")
        if 'process' in locals() and process.poll() is None: 
            process.kill()
    except Exception as e:
        logger.warning(f"Error with screencapture CLI: {e}")
    
    # Method 2: Try using PIL.ImageGrab as fallback
    try:
        logger.debug("Attempting to capture screen using PIL.ImageGrab")
        screenshot = ImageGrab.grab(all_screens=True)
        if screenshot:
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            screenshot.save(img_byte_arr, format='PNG')
            logger.debug("Successfully captured screen using PIL.ImageGrab")
            return img_byte_arr.getvalue()
    except Exception as e:
        logger.warning(f"Failed to capture screen using PIL.ImageGrab: {e}")
    
    # Method 3: Try using osascript with screencapture
    try:
        logger.debug("Attempting to capture screen using osascript")
        temp_file = "/tmp/eidon_screenshot_" + str(int(time.time())) + ".png"
        process = subprocess.Popen(["osascript", "-e", f'tell application "System Events" to tell application "System Events" to (set theFile to (POSIX file "{temp_file}" as text)) \
                                   -e "do shell script \"screencapture -x \" & (quoted form of POSIX path of theFile)'], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.communicate(timeout=10)
        
        if os.path.exists(temp_file):
            with open(temp_file, 'rb') as f:
                img_data = f.read()
            os.remove(temp_file)
            if img_data:
                logger.debug("Successfully captured screen using osascript")
                return img_data
    except Exception as e:
        logger.warning(f"Failed to capture screen using osascript: {e}")
        if os.path.exists(temp_file):
            try: os.remove(temp_file)
            except: pass
    
    logger.error("All screen capture methods failed")
    return None

# --- Idle Detection (using pynput) ---
class IdleMonitor:
    def __init__(self, idle_threshold_seconds: float, loop: Optional[asyncio.AbstractEventLoop] = None): # Add loop parameter
        self.idle_threshold_seconds = idle_threshold_seconds
        self.last_activity_time = time.monotonic()
        self.is_idle_state = False
        self._mouse_listener = None
        self._keyboard_listener = None
        self._lock = asyncio.Lock()
        self._loop = loop # Store the asyncio event loop
        self._pynput_available = 'PYNPUT_AVAILABLE' in globals() and PYNPUT_AVAILABLE

    async def _on_activity(self, *args):
        async with self._lock:
            current_time = time.monotonic()
            # Optional: Add a small debounce if events are too rapid
            # if current_time - self.last_activity_time < 0.1: # 100ms debounce
            #     return
            self.last_activity_time = current_time
            if self.is_idle_state:
                logger.info("User activity detected, Eidon capture resuming.")
                self.is_idle_state = False

    async def check_is_user_idle(self) -> bool:
        async with self._lock:
            idle_duration = time.monotonic() - self.last_activity_time
            if idle_duration > self.idle_threshold_seconds:
                if not self.is_idle_state:
                    logger.info(f"User idle for > {self.idle_threshold_seconds:.1f}s. Eidon capture pausing.")
                    self.is_idle_state = True
                return True
            return False

    def start_monitoring(self):
        if not self._pynput_available:
            logger.warning("pynput not available, cannot start idle monitoring.")
            return
        
        # Get the current running loop if not provided, crucial for run_coroutine_threadsafe
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError: # Should not happen if called from an asyncio context (e.g., service start)
                logger.error("Failed to get running asyncio loop for IdleMonitor. Activity callbacks might fail.")
                return # Cannot proceed without a loop

        logger.info(f"Starting idle monitoring (threshold: {self.idle_threshold_seconds}s).")

        def sync_on_activity_wrapper(*args):
            if self._loop and self._loop.is_running(): # Ensure loop is valid and running
                # Schedule the async _on_activity to run in the main asyncio event loop
                asyncio.run_coroutine_threadsafe(self._on_activity(*args), self._loop)
            # else: logger.warning("IdleMonitor: asyncio loop not available or not running for activity callback.")


        self._mouse_listener = mouse.Listener(on_move=sync_on_activity_wrapper, on_click=sync_on_activity_wrapper, on_scroll=sync_on_activity_wrapper)
        self._keyboard_listener = keyboard.Listener(on_press=sync_on_activity_wrapper)

        try:
            self._mouse_listener.start()
            self._keyboard_listener.start()
        except Exception as e: # Catch errors if listeners fail to start (e.g. on some headless systems)
            logger.error(f"Failed to start pynput listeners: {e}", exc_info=True)
            self._pynput_available = False # Mark as unavailable if start fails
            self._mouse_listener = None # Clear listeners
            self._keyboard_listener = None


    def stop_monitoring(self):
        if not self._pynput_available or (not self._mouse_listener and not self._keyboard_listener) : return # Ensure they were started
        logger.info("Stopping idle monitoring.")
        try:
            if self._mouse_listener: self._mouse_listener.stop()
            if self._keyboard_listener: self._keyboard_listener.stop()
        except Exception as e:
            logger.error(f"Error stopping pynput listeners: {e}", exc_info=True)
        # No need to join here as they are daemon threads by default or managed by pynput
        logger.info("Idle monitoring listeners signalled to stop.")


# --- Main Capture Service ---
class ScreenCaptureService:
    def __init__(self, output_queue: asyncio.Queue, loop: Optional[asyncio.AbstractEventLoop] = None): # Accept loop
        self.output_queue = output_queue
        self.is_running_flag = asyncio.Event()
        self._capture_task: Optional[asyncio.Task] = None
        # Pass the loop to IdleMonitor
        self._idle_monitor = IdleMonitor(config.IDLE_THRESHOLD_SECONDS, loop=loop or asyncio.get_event_loop())
        self.last_successful_capture_processing_time = 0
        self.is_capturing_now = False

    # ... _get_platform_specific_capture_data method remains the same ...
    async def _get_platform_specific_capture_data(self) -> Optional[Dict[str, Any]]:
        """Captures screen and gathers metadata based on platform."""
        timestamp_ms = get_current_timestamp_ms()
        image_bytes: Optional[bytes] = None
        active_info: Optional[Dict[str, Any]] = {"app_name": "Unknown", "window_title": "Unknown", "url": None}

        if is_macos():
            image_bytes = _capture_screen_macos()
            current_active_info = _get_active_window_info_macos()
            if current_active_info:
                active_info.update(current_active_info)
                if active_info.get("app_name"):
                    active_info["url"] = _get_browser_url_macos(active_info["app_name"])
        elif is_windows():
            # logger.debug("Attempting screen capture on Windows (ImageGrab).") # Can be noisy
            try:
                im = ImageGrab.grab()
                buffer = io.BytesIO()
                im.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                im.close()
                active_info = {"app_name": "WindowsApp", "window_title": "WindowsTitle", "url": None}
            except Exception as e: logger.error(f"Error grabbing screen on Windows: {e}", exc_info=True)
        elif is_linux():
            # logger.debug("Attempting screen capture on Linux (ImageGrab/X11).")
            try:
                im = ImageGrab.grab()
                buffer = io.BytesIO()
                im.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                im.close()
                active_info = {"app_name": "LinuxApp", "window_title": "LinuxTitle", "url": None}
            except Exception as e: logger.error(f"Error grabbing screen on Linux with Pillow/ImageGrab: {e}", exc_info=True)
        else:
            logger.error(f"Unsupported platform for screen capture: {config.PLATFORM_SYSTEM}")
            return None

        if not image_bytes: return None

        try:
            img = Image.open(io.BytesIO(image_bytes))
            original_size = img.size
            needs_resize = (config.MAX_SCREENSHOT_WIDTH > 0 and img.width > config.MAX_SCREENSHOT_WIDTH) or \
                           (config.MAX_SCREENSHOT_HEIGHT > 0 and img.height > config.MAX_SCREENSHOT_HEIGHT)
            if needs_resize:
                target_width = config.MAX_SCREENSHOT_WIDTH if config.MAX_SCREENSHOT_WIDTH > 0 else img.width
                target_height = config.MAX_SCREENSHOT_HEIGHT if config.MAX_SCREENSHOT_HEIGHT > 0 else img.height
                img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
            img.close()
        except UnidentifiedImageError:
            logger.error("Captured image data is not a valid image format. Skipping.")
            return None
        except Exception as e:
            logger.error(f"Error during image open/resize: {e}", exc_info=True)

        return {
            "timestamp_ms": timestamp_ms, "image_bytes": image_bytes,
            "app_name": active_info.get("app_name"), "window_title": active_info.get("window_title"),
            "page_url": active_info.get("page_url"), "source": "automatic_capture"
        }

    # ... _capture_and_queue_cycle method remains the same ...
    async def _capture_and_queue_cycle(self):
        if self.is_capturing_now: return 
        self.is_capturing_now = True
        try:
            capture_data = await self._get_platform_specific_capture_data()
            if capture_data:
                try:
                    await asyncio.wait_for(self.output_queue.put(capture_data), timeout=2.0)
                    self.last_successful_capture_processing_time = time.monotonic()
                except asyncio.QueueFull: logger.warning("Output queue is full. Capture data dropped.")
                except asyncio.TimeoutError: logger.warning("Timeout putting capture data on queue.")
                except Exception as e: logger.error(f"Error putting capture data on queue: {e}", exc_info=True)
        except Exception as e: logger.error(f"Unexpected error in _get_platform_specific_capture_data: {e}", exc_info=True)
        finally: self.is_capturing_now = False

    # ... _main_loop, start, stop, is_running methods remain largely the same ...
    async def _main_loop(self):
        self.is_running_flag.set() 
        pynput_available = hasattr(self, '_pynput_available') and self._pynput_available
        if pynput_available:
            self._idle_monitor.start_monitoring() # Idle monitor will get loop from its constructor
        logger.info("Screen capture service main loop started.")
        while self.is_running_flag.is_set():
            try:
                is_user_idle = await self._idle_monitor.check_is_user_idle() if pynput_available else False
                current_time_monotonic = time.monotonic()
                if is_user_idle:
                    await asyncio.sleep(min(config.IDLE_THRESHOLD_SECONDS / 2, 5.0)) 
                    continue
                time_since_last_good_process = current_time_monotonic - self.last_successful_capture_processing_time
                if time_since_last_good_process < config.MIN_CAPTURE_INTERVAL_DURING_ACTIVITY_SECONDS:
                    sleep_duration = config.MIN_CAPTURE_INTERVAL_DURING_ACTIVITY_SECONDS - time_since_last_good_process
                    await asyncio.sleep(sleep_duration)
                    continue 
                await self._capture_and_queue_cycle()
                await asyncio.sleep(config.CAPTURE_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                logger.info("Capture service main loop was cancelled.")
                break 
            except Exception as e:
                logger.error(f"Unhandled critical error in capture service main loop: {e}", exc_info=True)
                await asyncio.sleep(config.CAPTURE_INTERVAL_SECONDS * 3)
        if pynput_available:
            self._idle_monitor.stop_monitoring()
        logger.info("Screen capture service main loop finished.")

    async def start(self):
        logger.info(f"Starting screen capture service. ENABLE_AUTOMATIC_SCREEN_CAPTURE={config.ENABLE_AUTOMATIC_SCREEN_CAPTURE}")
        if not config.ENABLE_AUTOMATIC_SCREEN_CAPTURE:
            logger.info("Automatic screen capture is DISABLED by server configuration.")
            return False 
        if self._capture_task and not self._capture_task.done():
            logger.warning("Capture service task already exists and appears to be running.")
            return True 
        if not self.output_queue:
            logger.error("Output queue not provided to ScreenCaptureService. Cannot start.")
            return False
        logger.info("Screen capture service configuration looks good. Initializing...")
        self.is_running_flag.clear() 
        # Store pynput availability as an instance variable
        self._pynput_available = 'PYNPUT_AVAILABLE' in globals() and PYNPUT_AVAILABLE
        logger.info(f"pynput available: {self._pynput_available}")
        # Ensure the idle monitor has the correct loop when the service starts
        try:
            self._idle_monitor._loop = asyncio.get_event_loop() # Explicitly set/update loop here
            logger.info(f"Event loop set for idle monitor: {self._idle_monitor._loop}")
            self._capture_task = asyncio.create_task(self._main_loop())
            logger.info("Capture task created and main loop started.")
            return True
        except Exception as e:
            logger.error(f"Failed to start screen capture service: {e}", exc_info=True)
            return False
        logger.info("ScreenCaptureService background task created and initiated.")
        return True

    async def stop(self):
        if not config.ENABLE_AUTOMATIC_SCREEN_CAPTURE: return
        if not self._capture_task or self._capture_task.done():
            logger.info("Capture service not running or task already completed.")
            self.is_running_flag.clear() 
            return
        logger.info("Attempting to gracefully stop screen capture service...")
        self.is_running_flag.clear() 
        if self._capture_task:
            try:
                await asyncio.wait_for(self._capture_task, timeout=10.0)
                logger.info("Capture service task successfully awaited and completed.")
            except asyncio.CancelledError: 
                logger.info("Capture service task was cancelled as part of shutdown.")
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for capture service task to stop. Forcing cancellation.")
                self._capture_task.cancel() 
                try: await self._capture_task 
                except asyncio.CancelledError: logger.info("Forced cancellation of capture task processed.")
            except Exception as e: 
                logger.error(f"Error during capture service task shutdown: {e}", exc_info=True)
        self._capture_task = None 
        logger.info("ScreenCaptureService stopped.")

    @property
    def is_running(self) -> bool:
        """Check if the capture service is running."""
        if not hasattr(self, 'is_running_flag') or not hasattr(self, '_capture_task'):
            return False
            
        try:
            return (
                self.is_running_flag.is_set() and 
                self._capture_task is not None and 
                not self._capture_task.done()
            )
        except Exception as e:
            logger.warning(f"Error checking if capture service is running: {e}", exc_info=True)
            return False

# ... (Self-test code `if __name__ == "__main__":` remains the same, ensure IdleMonitor gets the loop if run standalone)
if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d|%(threadName)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    async def queue_consumer_for_test(queue: asyncio.Queue, stop_event: asyncio.Event):
        logger.info("Test queue consumer started.")
        count = 0
        while not stop_event.is_set() or not queue.empty():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
                ts = item.get('timestamp_ms'); app = item.get('app_name', 'N/A'); title = item.get('window_title', 'N/A')
                url = item.get('page_url', 'N/A'); img_size = len(item.get('image_bytes', b'')) // 1024; count +=1
                logger.info(f"CONSUMED CAPTURE #{count}: TS={ts}, App='{app[:20]}', Title='{title[:30]}', ImgSize={img_size}KB, QSize: {queue.qsize()}")
                queue.task_done()
            except asyncio.TimeoutError:
                if stop_event.is_set() and queue.empty(): break
            except Exception as e:
                logger.error(f"Error in test consumer: {e}", exc_info=True)
                if stop_event.is_set(): break
        logger.info(f"Test queue consumer finished after processing {count} items.")

    async def run_capture_service_test():
        if not config.ENABLE_AUTOMATIC_SCREEN_CAPTURE:
            logger.warning("Automatic screen capture is DISABLED in config. Test will not run the capture service.")
            return
        if not is_macos() and not is_windows() and not is_linux():
             logger.warning(f"Test running on unsupported platform '{config.PLATFORM_SYSTEM}' for full capture. Limited test.")

        current_loop = asyncio.get_event_loop() # Get the loop for the test context
        test_queue = asyncio.Queue(maxsize=50)
        service = ScreenCaptureService(test_queue, loop=current_loop) # Pass the loop
        
        stop_consumer_event = asyncio.Event()
        consumer_task = asyncio.create_task(queue_consumer_for_test(test_queue, stop_consumer_event))

        logger.info("Starting ScreenCaptureService for test...")
        started_successfully = await service.start()
        if not started_successfully:
            logger.error("Failed to start ScreenCaptureService for test. Aborting.")
            stop_consumer_event.set(); await asyncio.sleep(1)
            if not consumer_task.done(): consumer_task.cancel()
            return
        run_duration = 20 
        logger.info(f"ScreenCaptureService running. Test will last for {run_duration} seconds...")
        try: await asyncio.sleep(run_duration)
        except KeyboardInterrupt: logger.info("Test interrupted by user.")
        finally:
            logger.info("Stopping ScreenCaptureService from test...")
            await service.stop()
            logger.info("Signaling consumer to stop and drain queue...")
            stop_consumer_event.set()
            try: await asyncio.wait_for(consumer_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Consumer task timed out during shutdown.")
                if not consumer_task.done(): consumer_task.cancel()
            except asyncio.CancelledError: logger.info("Consumer task was cancelled.")
        logger.info("ScreenCaptureService test finished.")
    asyncio.run(run_capture_service_test())