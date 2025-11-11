# resize_images.py (fixed)
import argparse
import sys
from pathlib import Path
import tempfile
import os

try:
    from PIL import Image, ImageOps
except ImportError:
    print("This script requires Pillow. Install with: pip install pillow")
    sys.exit(1)

SUPPORTED_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"
}
EXT_TO_FORMAT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
    ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF", ".gif": "GIF"
}

def positive_float(x: str) -> float:
    try:
        v = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("Scale must be a number.")
    if v <= 0:
        raise argparse.ArgumentTypeError("Scale must be > 0.")
    return v

def ensure_jpeg_compatible(im: Image.Image) -> Image.Image:
    """Convert/flatten to a JPEG-safe mode (RGB or L)."""
    if im.mode in ("RGBA", "LA"):
        # Flatten transparency onto white
        bg = Image.new("RGB", im.size, (255, 255, 255))
        alpha = im.getchannel("A")
        bg.paste(im.convert("RGB"), mask=alpha)
        return bg
    if im.mode == "P":
        return im.convert("RGB")
    if im.mode not in ("RGB", "L"):
        return im.convert("RGB")
    return im

def resize_image_inplace(path: Path, scale: float) -> tuple[int, int, int, int]:
    """
    Resize a single (non-animated) image at `path` by `scale`, overwriting it.
    Returns (old_w, old_h, new_w, new_h).
    """
    with Image.open(path) as im_opened:
        # Keep original meta before transforms
        orig_format = im_opened.format
        orig_info = dict(getattr(im_opened, "info", {}))
        # Skip animated to avoid breaking them
        if getattr(im_opened, "is_animated", False):
            raise RuntimeError("animated image detected (skipped)")

        # Respect EXIF orientation and preserve EXIF (Orientation reset)
        im = ImageOps.exif_transpose(im_opened)
        exif = im.getexif()
        try:
            if exif is not None:
                exif[274] = 1  # Orientation
                exif_bytes = exif.tobytes()
            else:
                exif_bytes = None
        except Exception:
            exif_bytes = None

        old_w, old_h = im.width, im.height
        new_w = max(1, int(round(old_w * scale)))
        new_h = max(1, int(round(old_h * scale)))
        if new_w == old_w and new_h == old_h:
            return old_w, old_h, new_w, new_h

        im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # Choose output by extension (keep original type)
        ext = path.suffix.lower()
        out_format = EXT_TO_FORMAT.get(ext, im.format or "PNG")

        # JPEG safety: ensure compatible mode
        if out_format == "JPEG":
            im = ensure_jpeg_compatible(im)

        # Save to temp then atomically replace
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.stem + "_tmp_", suffix=ext, delete=False
        ) as tf:
            tmp_path = Path(tf.name)

        save_kwargs = {}
        if out_format == "JPEG":
            save_kwargs.update({
                "quality": 95,
                "optimize": True,
            })
            # If the original looked progressive, preserve that *boolean*
            if bool(orig_info.get("progressive", False)):
                save_kwargs["progressive"] = True
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
        elif out_format == "PNG":
            save_kwargs.update({"optimize": True})
        elif out_format == "WEBP":
            save_kwargs.update({"quality": 95, "method": 6})
        # TIFF/GIF/BMP: defaults are fine

        try:
            im.save(tmp_path, format=out_format, **save_kwargs)
            os.replace(tmp_path, path)  # atomic on same volume
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            finally:
                raise e

        return old_w, old_h, new_w, new_h

def main():
    parser = argparse.ArgumentParser(
        description="Resize all images in the current directory (non-recursive) by a scale factor and overwrite them."
    )
    parser.add_argument("scale", type=positive_float, help="Scale factor (e.g., 0.5 halves size, 2.0 doubles).")
    args = parser.parse_args()
    scale = args.scale

    cwd = Path(".").resolve()
    files = [p for p in cwd.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    if not files:
        print("No supported images found in the current directory.")
        return

    processed = 0
    skipped = 0
    errors = 0

    for p in files:
        try:
            old_w, old_h, new_w, new_h = resize_image_inplace(p, scale)
            if (old_w, old_h) == (new_w, new_h):
                print(f"[skip] {p.name}: size unchanged ({old_w}x{old_h}).")
                skipped += 1
            else:
                print(f"[ok]   {p.name}: {old_w}x{old_h} -> {new_w}x{new_h}")
                processed += 1
        except RuntimeError as re:
            print(f"[skip] {p.name}: {re}")
            skipped += 1
        except Exception as e:
            print(f"[err]  {p.name}: {e}")
            errors += 1

    print("\nDone.")
    print(f"  Resized: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors : {errors}")

if __name__ == "__main__":
    main()
