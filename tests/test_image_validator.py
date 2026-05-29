import base64
import io
from PIL import Image
import pytest
from fastapi import HTTPException
from backend.validators.image import validate_image_or_raise, ImageValidationResult

def _create_image_bytes(format="PNG", size=(100, 100)):
    img = Image.new("RGB", size, color="red")
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def test_valid_image():
    img_bytes = _create_image_bytes()
    # Should not raise
    validate_image_or_raise(img_bytes)

def test_invalid_format():
    img_bytes = _create_image_bytes(format="BMP")  # Assuming BMP is allowed, pick a truly unsupported format like TIFF
    img_bytes = _create_image_bytes(format="TIFF")
    with pytest.raises(HTTPException):
        validate_image_or_raise(img_bytes)
