"""Enhanced OCR processing using Apple Vision framework with advanced features."""
import asyncio
import io
import logging
import platform
import warnings
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage, ImageDraw
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Check if we're on macOS and can import Vision
HAS_VISION = False
VISION_IMPORT_ERROR = None

if platform.system() == 'Darwin':
    try:
        import Vision
        from CoreFoundation import NSData
        from Quartz import (
            CGImageSourceCreateWithData,
            CGImageSourceCreateImageAtIndex,
            kCGImageSourceTypeIdentifierHint,
            kCGImageSourceShouldCache,
            CGImageGetWidth,
            CGImageGetHeight
        )
        from Foundation import NSURL, NSUTF8StringEncoding
        from objc import nil
        # objc_objectRef is not needed in newer pyobjc versions
        try:
            from objc import objc_objectRef  # For older pyobjc versions
        except ImportError:
            pass
        HAS_VISION = True
    except ImportError as e:
        VISION_IMPORT_ERROR = str(e)
        logger.warning(f"Failed to import Vision framework: {VISION_IMPORT_ERROR}")
    except Exception as e:
        VISION_IMPORT_ERROR = str(e)
        logger.error(f"Error initializing Vision framework: {VISION_IMPORT_ERROR}")
else:
    VISION_IMPORT_ERROR = "Not running on macOS"

class OCR:
    """OCR class to extract text from images using Apple Vision with advanced features."""
    
    HAS_VISION = HAS_VISION
    VISION_IMPORT_ERROR = VISION_IMPORT_ERROR
    
    def __init__(self, image: Union[PILImage.Image, bytes], format: str = "PNG"):
        """
        Initialize the OCR processor.
        
        Args:
            image: Input image as PIL Image or bytes
            format: Image format (default: "PNG")
        """
        self.image = image
        self.format = format
        self.res = None
        self.data = []
        self.dataframe = None
        
        if isinstance(image, PILImage.Image):
            self.image_bytes = self.image_to_buffer(image)
        else:
            self.image_bytes = image
    
    def image_to_buffer(self, image: PILImage.Image) -> bytes:
        """
        Convert a PIL image to bytes.
        
        Args:
            image: PIL Image to convert
            
        Returns:
            Image data as bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format=self.format)
        return buffer.getvalue()
    
    def dealloc(self, request: Any, request_handler: Any) -> None:
        """
        Clean up and deallocate resources.
        
        Args:
            request: The text recognition request object
            request_handler: The image request handler object
        """
        try:
            if request:
                request.dealloc()
            if request_handler:
                request_handler.dealloc()
        except Exception as e:
            logger.error(f"Error during deallocation: {e}")
    
    def cluster(self, dataframe: pd.DataFrame, num_clusters: int = 3) -> list:
        """
        Perform K-Means clustering on the text data.
        
        Args:
            dataframe: DataFrame containing text data with 'x' and 'y' columns
            num_clusters: Number of clusters to create
            
        Returns:
            List of cluster labels
        """
        try:
            array = np.array([(x, y) for x, y in zip(dataframe["x"], dataframe["y"])])
            tensor = torch.tensor(array, dtype=torch.float32)
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            kmeans.fit(tensor.numpy())
            labels = kmeans.labels_
            self.dataframe["Cluster"] = labels
            return labels
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            return []
    
    async def extract_text(self) -> str:
        """
        Extract text from the image using Apple Vision.
        
        Returns:
            Extracted text as a single string
        """
        if not HAS_VISION:
            logger.warning("Apple Vision framework not available")
            return ""
        
        def _extract_sync():
            request, request_handler = None, None
            try:
                # Convert image data to NSData
                ns_data = NSData.dataWithData_(self.image_bytes)
                
                # Create a new image-request handler
                request_handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                    ns_data, None
                )
                
                # Create a new request to recognize text
                request = Vision.VNRecognizeTextRequest.alloc().init()
                request.setRecognitionLevel_(1)  # 0 = fast, 1 = accurate
                
                # Perform the request
                success = request_handler.performRequests_error_([request], None)
                
                if not success:
                    logger.error("Vision request failed")
                    return ""
                
                # Process results
                texts = []
                for observation in request.results() or []:
                    try:
                        bbox = observation.boundingBox()
                        w, h = bbox.size.width, bbox.size.height
                        x, y = bbox.origin.x, bbox.origin.y
                        
                        recognized_text = observation.topCandidates_(1)
                        if recognized_text and recognized_text.count() > 0:
                            text = recognized_text[0].string()
                            confidence = recognized_text[0].confidence()
                            self.data.append((text, confidence, [x, y, w, h]))
                            texts.append(text)
                    except Exception as e:
                        logger.error(f"Error processing observation: {e}")
                
                return "\n".join(texts) if texts else ""
                
            except Exception as e:
                logger.error(f"Error in Vision OCR: {e}", exc_info=True)
                return ""
            finally:
                try:
                    self.dealloc(request, request_handler)
                except:
                    pass
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _extract_sync)
            
            # If we have data, create a DataFrame
            if self.data:
                content, confidences, bbox = zip(*self.data)
                w = [b[2] for b in bbox]
                h = [b[3] for b in bbox]
                x = [b[0] for b in bbox]
                y = [b[1] for b in bbox]
                
                # Calculate text density
                text_areas = np.array(w) * np.array(h)
                total_area = 1 * 1
                densities = text_areas / total_area
                
                # Calculate centroids
                cx = np.array(x) + np.array(w) / 2
                cy = np.array(y) + np.array(h) / 2
                
                self.dataframe = pd.DataFrame({
                    "Content": content,
                    "Confidence": confidences,
                    "x": x,
                    "y": y,
                    "Width": w,
                    "Height": h,
                    "Density": densities,
                    "Centroid x": cx,
                    "Centroid y": cy
                })
                logger.info(f"Extracted {len(self.data)} OCR blocks, avg confidence: {np.mean(confidences):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async extraction: {e}", exc_info=True)
            return ""

# For backward compatibility
class VisionOCR:
    """Legacy wrapper for backward compatibility."""
    
    HAS_VISION = HAS_VISION
    VISION_IMPORT_ERROR = VISION_IMPORT_ERROR
    
    @staticmethod
    async def extract_text(image_data: bytes) -> str:
        """Legacy method to extract text from image data."""
        ocr = OCR(image_data)
        return await ocr.extract_text()

# Global instance for backward compatibility
vision_ocr = VisionOCR()

def convert_pil_to_bytes(image: PILImage.Image, format: str = "PNG", quality: int = 80) -> bytes:
    """
    Convert a PIL Image to bytes in the specified format.
    
    Args:
        image: PIL Image to convert
        format: Output format (WEBP, PNG, JPEG)
        quality: Quality setting for lossy formats (1-100)
        
    Returns:
        Bytes of the converted image
    """
    buffer = io.BytesIO()
    save_kwargs = {"format": format.upper()}
    
    # Set quality for lossy formats
    if format.upper() in ["WEBP", "JPEG"]:
        save_kwargs["quality"] = quality
        if format.upper() == "WEBP":
            save_kwargs["method"] = 6  # Best quality/speed tradeoff
    
    # Convert to RGB if saving as JPEG and image has alpha channel
    if format.upper() == "JPEG" and image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
    
    try:
        image.save(buffer, **save_kwargs)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error converting image to {format}: {e}")
        raise

# For backward compatibility
async def extract_text_from_image_vision(image_data: bytes, input_mime_type: str = "image/png") -> str:
    """Legacy function for backward compatibility."""
    return await vision_ocr.extract_text(image_data)

# Set this as the default extractor if on macOS
if platform.system() == 'Darwin' and HAS_VISION:
    extract_text_from_image = extract_text_from_image_vision
else:
    # Fallback implementation if not on macOS or Vision not available
    async def extract_text_from_image(image_data: bytes, input_mime_type: str = "image/png") -> str:
        logger.warning("Apple Vision not available, returning empty string")
        return ""
