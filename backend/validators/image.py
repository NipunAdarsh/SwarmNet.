"""
Image validation for SwarmNet.

Provides security validation for uploaded images before processing.
Validates format, size, and dimensions to prevent malicious uploads.
"""

import os
import io
import logging
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Configuration from environment
MAX_IMAGE_SIZE_MB: int = int(os.environ.get("MAX_IMAGE_SIZE_MB", 10))
MAX_IMAGE_SIZE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024

MAX_IMAGE_WIDTH: int = int(os.environ.get("MAX_IMAGE_WIDTH", 4096))
MAX_IMAGE_HEIGHT: int = int(os.environ.get("MAX_IMAGE_HEIGHT", 4096))

# Parse allowed formats from environment
_formats_raw = os.environ.get("ALLOWED_IMAGE_FORMATS", "jpg,jpeg,png,webp,gif,bmp")
ALLOWED_IMAGE_FORMATS: Set[str] = {fmt.strip().lower() for fmt in _formats_raw.split(",") if fmt.strip()}

# Map PIL format names to common extensions
PIL_FORMAT_MAP = {
    "JPEG": {"jpg", "jpeg"},
    "PNG": {"png"},
    "GIF": {"gif"},
    "WEBP": {"webp"},
    "BMP": {"bmp"},
    "TIFF": {"tiff", "tif"},
}


@dataclass
class ImageValidationResult:
    """Result of image validation."""
    valid: bool
    error_code: Optional[int] = None  # HTTP status code
    error_message: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None


class ImageValidator:
    """
    Validates images for security and format compliance.

    Validates:
    - Size in bytes (before decoding)
    - Image format (via PIL detection)
    - Image dimensions (after decoding)
    """

    def __init__(
        self,
        max_size_bytes: int = MAX_IMAGE_SIZE_BYTES,
        max_width: int = MAX_IMAGE_WIDTH,
        max_height: int = MAX_IMAGE_HEIGHT,
        allowed_formats: Set[str] = None,
    ):
        self.max_size_bytes = max_size_bytes
        self.max_width = max_width
        self.max_height = max_height
        self.allowed_formats = allowed_formats or ALLOWED_IMAGE_FORMATS

    def _is_format_allowed(self, pil_format: str) -> bool:
        """Check if the PIL format is in the allowed list."""
        if not pil_format:
            return False

        pil_format_upper = pil_format.upper()

        # Check if PIL format maps to any allowed extension
        if pil_format_upper in PIL_FORMAT_MAP:
            mapped_extensions = PIL_FORMAT_MAP[pil_format_upper]
            return bool(mapped_extensions & self.allowed_formats)

        # Fallback: check if PIL format name itself is allowed (lowercase)
        return pil_format.lower() in self.allowed_formats

    def validate(self, image_bytes: bytes) -> ImageValidationResult:
        """
        Validate image bytes.

        Returns:
            ImageValidationResult with validation status and details.
            error_code will be:
            - 413 for size violations
            - 415 for format violations
            - 400 for other processing errors
        """
        size_bytes = len(image_bytes)

        # Check 1: Size limit (before attempting to decode)
        if size_bytes > self.max_size_bytes:
            return ImageValidationResult(
                valid=False,
                error_code=413,
                error_message=f"Image size ({size_bytes / 1024 / 1024:.2f} MB) exceeds limit ({self.max_size_bytes / 1024 / 1024:.0f} MB)",
                size_bytes=size_bytes,
            )

        # Check 2: Minimum size (empty or too small to be valid)
        if size_bytes < 10:
            return ImageValidationResult(
                valid=False,
                error_code=400,
                error_message="Image data too small to be a valid image",
                size_bytes=size_bytes,
            )

        # Check 3: Try to open with PIL to validate format
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_format = img.format
            width, height = img.size
        except Exception as exc:
            logger.warning("Image validation failed: cannot open image: %s", exc)
            return ImageValidationResult(
                valid=False,
                error_code=415,
                error_message=f"Cannot decode image: {str(exc)}",
                size_bytes=size_bytes,
            )

        # Check 4: Format whitelist
        if not self._is_format_allowed(img_format):
            return ImageValidationResult(
                valid=False,
                error_code=415,
                error_message=f"Image format '{img_format}' not allowed. Allowed formats: {', '.join(sorted(self.allowed_formats))}",
                size_bytes=size_bytes,
                format=img_format,
                width=width,
                height=height,
            )

        # Check 5: Dimension limits
        if width > self.max_width or height > self.max_height:
            return ImageValidationResult(
                valid=False,
                error_code=413,
                error_message=f"Image dimensions ({width}x{height}) exceed limit ({self.max_width}x{self.max_height})",
                size_bytes=size_bytes,
                format=img_format,
                width=width,
                height=height,
            )

        # All checks passed
        logger.debug(
            "Image validation passed: format=%s, size=%d bytes, dimensions=%dx%d",
            img_format, size_bytes, width, height
        )
        return ImageValidationResult(
            valid=True,
            size_bytes=size_bytes,
            format=img_format,
            width=width,
            height=height,
        )

    def validate_or_raise(self, image_bytes: bytes) -> Tuple[int, int, str]:
        """
        Validate image and raise HTTPException on failure.

        Returns:
            Tuple of (width, height, format) if valid.

        Raises:
            HTTPException with appropriate status code on failure.
        """
        from fastapi import HTTPException

        result = self.validate(image_bytes)

        if not result.valid:
            raise HTTPException(
                status_code=result.error_code or 400,
                detail=result.error_message or "Image validation failed",
            )

        return result.width, result.height, result.format


# Default validator instance
default_validator = ImageValidator()


def validate_image(image_bytes: bytes) -> ImageValidationResult:
    """Validate image using default validator settings."""
    return default_validator.validate(image_bytes)


def validate_image_or_raise(image_bytes: bytes) -> Tuple[int, int, str]:
    """Validate image and raise HTTPException on failure."""
    return default_validator.validate_or_raise(image_bytes)
