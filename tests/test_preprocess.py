import pytest
from backend.preprocess import _preprocess_image

# Simple dummy image (RGB 224x224)
def _dummy_image_bytes():
    from PIL import Image
    import io
    img = Image.new("RGB", (224, 224), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_preprocess_returns_correct_shape():
    raw = _dummy_image_bytes()
    arr = _preprocess_image(raw)
    # Expect shape (1, 3, 224, 224)
    assert arr.shape == (1, 3, 224, 224)
