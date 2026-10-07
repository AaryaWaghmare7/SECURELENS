"""Bounded image decoding for uploads, paths, and research scripts."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_INPUT_PIXELS = 40_000_000
MAX_ANALYSIS_SIDE = 1600
SUPPORTED_FORMATS = {"JPEG", "PNG"}


class ImageAnalysisError(ValueError):
    """An image cannot safely be decoded or analyzed."""


@dataclass
class LoadedImage:
    image: Image.Image
    metadata: dict
    notes: list[str]


def load_image(source, max_side=MAX_ANALYSIS_SIDE):
    """Validate actual format, normalize orientation/color, and limit processing size."""
    if max_side < 1:
        raise ValueError("Analysis size must be positive.")
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            if path.stat().st_size > MAX_UPLOAD_BYTES:
                raise ImageAnalysisError("Use an image smaller than 16 MB.")
            data = path.read_bytes()
        except OSError as error:
            raise ImageAnalysisError("The image file could not be read. Check its path and permissions.") from error
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif hasattr(source, "getvalue"):
        data = source.getvalue()
    else:
        raise ImageAnalysisError("Provide a JPG, JPEG, or PNG image.")
    if not data:
        raise ImageAnalysisError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageAnalysisError("Use an image smaller than 16 MB.")

    notes = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as original:
                if original.format not in SUPPORTED_FORMATS:
                    raise ImageAnalysisError("Only JPG, JPEG, and PNG files are supported.")
                width, height = original.size
                if width * height > MAX_INPUT_PIXELS:
                    raise ImageAnalysisError("Use an image with at most 40 megapixels.")
                metadata = {
                    "width": width, "height": height,
                    "channels": len(original.getbands()), "mode": original.mode,
                    "format": original.format, "file_size_bytes": len(data),
                    "has_exif": bool(original.getexif()),
                }
                original.load()
                oriented = ImageOps.exif_transpose(original)
                if oriented.size != original.size:
                    notes.append("EXIF orientation was applied before analysis.")
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    normalized = Image.alpha_composite(background, rgba).convert("RGB")
                    notes.append("Transparency was composited on white for JPEG analysis.")
                else:
                    normalized = oriented.convert("RGB")
                if max(normalized.size) > max_side:
                    normalized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    notes.append("Large image resized for bounded memory use; metrics describe the resized image.")
                metadata["analysis_width"], metadata["analysis_height"] = normalized.size
                metadata["analysis_channels"] = 3
                return LoadedImage(normalized, metadata, notes)
    except ImageAnalysisError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageAnalysisError("The image could not be decoded. Try exporting it again as JPG or PNG.") from error
