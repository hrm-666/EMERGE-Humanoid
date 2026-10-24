"""Load a local image into the main agent's multimodal tool context."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from Emerge.agent.tools.filesystem import _FsTool
from Emerge.subagents import ImageContent, SubagentToolResult


class ViewImageTool(_FsTool):
    @property
    def name(self) -> str:
        return "view_image"

    @property
    def description(self) -> str:
        return (
            "View a local image: attach its pixels to your context for visual inspection. "
            "Paths are relative to the workspace or absolute. Large images are scaled "
            "to at most 1536 pixels per side; animated images show their first frame."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "description": "Path to the image file"},
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(self, path: str) -> SubagentToolResult:
        return await asyncio.to_thread(self._load_image, self._resolve(path))

    def _load_image(self, path: Path) -> SubagentToolResult:
        # Decode and encode off the event loop, keeping the original file unchanged.
        with Image.open(path) as source:
            first_frame = getattr(source, "is_animated", False)
            image = ImageOps.exif_transpose(source)
            original_size = image.size
            image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            if "A" in image.getbands() or "transparency" in image.info:
                rgba = image.convert("RGBA")
                image = Image.new("RGB", rgba.size, "white")
                image.paste(rgba, mask=rgba.getchannel("A"))
            else:
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            description = (
                f"Image: {path}\n"
                f"Original size (after orientation): {original_size[0]} x {original_size[1]} pixels. "
                f"Attached size: {image.width} x {image.height} pixels (JPEG)."
            )
            if first_frame:
                description += " First frame of an animated image."
        return SubagentToolResult(
            text=description,
            content=(ImageContent(f"data:image/jpeg;base64,{data}", detail="high"),),
        )
