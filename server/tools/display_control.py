import base64
import binascii
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)

SCREEN_WIDTH = 135
SCREEN_HEIGHT = 240
STATUS_BAR_Y = 10

FONT_PIXELS_BY_SIZE = {
    1: 14,
    2: 20,
    3: 28,
    4: 36,
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/NotoSansCJKJP-Regular.otf",
    "/Library/Fonts/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
]

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


def _rgb888_to_rgb565_be(rgb888: bytes) -> bytes:
    out = bytearray((len(rgb888) // 3) * 2)
    j = 0
    for i in range(0, len(rgb888), 3):
        r = rgb888[i]
        g = rgb888[i + 1]
        b = rgb888[i + 2]
        pixel = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        out[j] = (pixel >> 8) & 0xFF
        out[j + 1] = pixel & 0xFF
        j += 2
    return bytes(out)


def _fit_size(raw_w: int, raw_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    if raw_w <= 0 or raw_h <= 0 or max_w <= 0 or max_h <= 0:
        return 0, 0
    scale = min(max_w / raw_w, max_h / raw_h, 1.0)
    w = max(1, int(round(raw_w * scale)))
    h = max(1, int(round(raw_h * scale)))
    return w, h


@lru_cache(maxsize=1)
def _resolve_font_path() -> str | None:
    env_path = os.environ.get("AICHATTER_DISPLAY_FONT", "").strip()
    if env_path and Path(env_path).exists():
        return env_path

    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 0x7F for ch in text)


def _measure_text_width(draw, text: str, font) -> int:
    if not text:
        return 0
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return max(0, right - left)


def _split_wrapped_lines(
    draw,
    text: str,
    font,
    max_width: int,
    max_lines: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.replace("\r", "").split("\n"):
        if len(lines) >= max_lines:
            break

        if paragraph == "":
            lines.append("")
            continue

        current = ""
        for ch in paragraph:
            trial = current + ch
            if not current or _measure_text_width(draw, trial, font) <= max_width:
                current = trial
                continue

            lines.append(current)
            if len(lines) >= max_lines:
                return lines
            current = ch

        lines.append(current)

    return lines[:max_lines]


def _render_text_to_rgb565(
    text: str,
    size: int,
    max_width: int,
    max_height: int,
) -> tuple[bytes, int, int] | ToolResult:
    if Image is None or ImageDraw is None or ImageFont is None:
        return ToolResult(content="Pillow未インストールのためテキスト描画できません。", is_error=True)

    if max_width <= 0 or max_height <= 0:
        return ToolResult(content="描画可能な領域がありません。", is_error=True)

    font_path = _resolve_font_path()
    if _contains_non_ascii(text) and not font_path:
        return ToolResult(
            content=(
                "日本語フォントが見つかりません。"
                "環境変数 AICHATTER_DISPLAY_FONT でフォントファイルを指定してください。"
            ),
            is_error=True,
        )

    font_px = FONT_PIXELS_BY_SIZE.get(size, FONT_PIXELS_BY_SIZE[1])
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_px)
        else:
            font = ImageFont.load_default()
    except Exception as e:
        return ToolResult(content=f"フォント読み込みエラー: {e}", is_error=True)

    canvas = Image.new("RGB", (max_width, max_height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        ascent, descent = font.getmetrics()
        line_height = max(font_px, ascent + descent)
    except Exception:
        line_height = font_px
    line_height += max(2, font_px // 6)

    max_lines = max(1, max_height // line_height)
    wrapped = _split_wrapped_lines(draw, text, font, max_width, max_lines)

    y = 0
    used_w = 0
    for line in wrapped:
        if y + line_height > max_height:
            break
        if line:
            draw.text((0, y), line, font=font, fill=(255, 255, 255))
            used_w = max(used_w, _measure_text_width(draw, line, font))
        y += line_height

    used_h = max(1, min(max_height, y if y > 0 else line_height))
    used_w = max(1, min(max_width, used_w if used_w > 0 else max_width))

    rendered = canvas.crop((0, 0, used_w, used_h))
    rgb565 = _rgb888_to_rgb565_be(rendered.tobytes())
    return rgb565, rendered.width, rendered.height


class DisplayTextTool(ToolBase):
    name = "display_text"
    description = (
        "M5StickS3のディスプレイにテキストを表示します。"
        "日本語対応。size(1-4)で文字サイズを指定できます。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "表示するテキスト (改行可)",
            },
            "size": {
                "type": "integer",
                "description": "文字サイズ倍率 (1-4)",
                "minimum": 1,
                "maximum": 4,
            },
            "x": {
                "type": "integer",
                "description": "描画開始X座標 (0-134)",
                "minimum": 0,
                "maximum": 134,
            },
            "y": {
                "type": "integer",
                "description": "描画開始Y座標 (0-239)",
                "minimum": 0,
                "maximum": 239,
            },
            "clear": {
                "type": "boolean",
                "description": "trueなら本文領域を消去してから描画",
            },
        },
        "required": ["text"],
    }

    def __init__(self, get_pipelines: Callable) -> None:
        self._get_pipelines = get_pipelines

    async def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text")
        if not isinstance(text, str):
            return ToolResult(content="text は文字列で指定してください。", is_error=True)

        size = kwargs.get("size", 1)
        x = kwargs.get("x", 0)
        y = kwargs.get("y", STATUS_BAR_Y)
        clear = kwargs.get("clear", True)

        if not isinstance(size, int) or size < 1 or size > 4:
            return ToolResult(content="size は1-4の整数で指定してください。", is_error=True)
        if not isinstance(x, int) or x < 0 or x >= SCREEN_WIDTH:
            return ToolResult(content="x は0-134の整数で指定してください。", is_error=True)
        if not isinstance(y, int) or y < 0 or y >= SCREEN_HEIGHT:
            return ToolResult(content="y は0-239の整数で指定してください。", is_error=True)
        if not isinstance(clear, bool):
            return ToolResult(content="clear は true/false で指定してください。", is_error=True)

        pipelines = self._get_pipelines()
        if not pipelines:
            return ToolResult(content="接続中のデバイスがありません。", is_error=True)

        payload = None
        if text != "":
            rendered = _render_text_to_rgb565(
                text=text,
                size=size,
                max_width=SCREEN_WIDTH - x,
                max_height=SCREEN_HEIGHT - y,
            )
            if isinstance(rendered, ToolResult):
                return rendered
            payload = rendered

        sent = 0
        for pipeline in list(pipelines):
            try:
                if clear:
                    await pipeline.send_display_text(
                        text="",
                        size=1,
                        x=0,
                        y=STATUS_BAR_Y,
                        clear=True,
                    )
                if payload is not None:
                    rgb565, width, height = payload
                    await pipeline.send_display_image(
                        rgb565=rgb565,
                        width=width,
                        height=height,
                        x=x,
                        y=y,
                    )
                sent += 1
            except Exception as e:
                logger.error(f"display_text送信エラー: {e}", exc_info=True)
                return ToolResult(content=f"テキスト表示送信エラー: {e}", is_error=True)

        logger.info(f"テキスト表示送信: devices={sent} size={size} x={x} y={y}")
        return ToolResult(content=f"テキストを{sent}台のデバイスに表示しました。")


class DisplayImageTool(ToolBase):
    name = "display_image"
    description = (
        "M5StickS3のディスプレイに画像を表示します。"
        "image_path または rgb565_base64 を指定してください。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "表示するローカル画像ファイルのパス (PNG/JPEG等)",
            },
            "rgb565_base64": {
                "type": "string",
                "description": (
                    "RGB565生データをbase64化した文字列。"
                    "長さは width*height*2 バイトに一致する必要があります。"
                ),
            },
            "width": {
                "type": "integer",
                "description": "画像幅 (1-135)。省略時は元画像サイズを使用。",
                "minimum": 1,
                "maximum": 135,
            },
            "height": {
                "type": "integer",
                "description": "画像高 (1-240)。省略時は元画像サイズを使用。",
                "minimum": 1,
                "maximum": 240,
            },
            "x": {
                "type": "integer",
                "description": "描画先X座標 (0-134)",
                "minimum": 0,
                "maximum": 134,
            },
            "y": {
                "type": "integer",
                "description": "描画先Y座標 (0-239)",
                "minimum": 0,
                "maximum": 239,
            },
            "clear": {
                "type": "boolean",
                "description": "trueなら画像描画前に本文領域をクリア",
            },
        },
        "required": [],
    }

    def __init__(self, get_pipelines: Callable) -> None:
        self._get_pipelines = get_pipelines

    @staticmethod
    def _validate_position(x: int, y: int) -> str | None:
        if not isinstance(x, int) or x < 0 or x >= SCREEN_WIDTH:
            return "x は0-134の整数で指定してください。"
        if not isinstance(y, int) or y < 0 or y >= SCREEN_HEIGHT:
            return "y は0-239の整数で指定してください。"
        return None

    @staticmethod
    def _validate_dimension(name: str, value: int | None, limit: int) -> str | None:
        if value is None:
            return None
        if not isinstance(value, int) or value < 1 or value > limit:
            return f"{name} は1-{limit}の整数で指定してください。"
        return None

    def _load_from_path(
        self,
        image_path: str,
        req_width: int | None,
        req_height: int | None,
        x: int,
        y: int,
    ) -> tuple[bytes, int, int] | ToolResult:
        if Image is None:
            return ToolResult(
                content="Pillow未インストールのため image_path は利用できません。",
                is_error=True,
            )

        path = Path(image_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_file():
            return ToolResult(content=f"画像ファイルが見つかりません: {path}", is_error=True)

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                raw_w, raw_h = img.size

                if req_width is not None and req_height is not None:
                    width, height = req_width, req_height
                elif req_width is not None:
                    width = req_width
                    height = max(1, int(round(raw_h * (req_width / raw_w))))
                elif req_height is not None:
                    height = req_height
                    width = max(1, int(round(raw_w * (req_height / raw_h))))
                else:
                    width, height = raw_w, raw_h

                max_w = SCREEN_WIDTH - x
                max_h = SCREEN_HEIGHT - y
                if width > max_w or height > max_h:
                    width, height = _fit_size(width, height, max_w, max_h)

                if width <= 0 or height <= 0:
                    return ToolResult(
                        content="描画可能な画像サイズを計算できませんでした。",
                        is_error=True,
                    )

                if (width, height) != img.size:
                    resampling = getattr(Image, "Resampling", Image)
                    img = img.resize((width, height), resampling.LANCZOS)

                rgb565 = _rgb888_to_rgb565_be(img.tobytes())
                return rgb565, width, height
        except Exception as e:
            return ToolResult(content=f"画像読み込みエラー: {e}", is_error=True)

    async def execute(self, **kwargs) -> ToolResult:
        image_path = kwargs.get("image_path")
        rgb565_base64 = kwargs.get("rgb565_base64")
        width = kwargs.get("width")
        height = kwargs.get("height")
        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        clear = kwargs.get("clear", False)

        pos_err = self._validate_position(x, y)
        if pos_err:
            return ToolResult(content=pos_err, is_error=True)

        w_err = self._validate_dimension("width", width, SCREEN_WIDTH)
        if w_err:
            return ToolResult(content=w_err, is_error=True)
        h_err = self._validate_dimension("height", height, SCREEN_HEIGHT)
        if h_err:
            return ToolResult(content=h_err, is_error=True)
        if not isinstance(clear, bool):
            return ToolResult(content="clear は true/false で指定してください。", is_error=True)

        has_path = isinstance(image_path, str) and image_path.strip() != ""
        has_base64 = isinstance(rgb565_base64, str) and rgb565_base64 != ""
        if not has_path and not has_base64:
            return ToolResult(
                content="image_path か rgb565_base64 のいずれかを指定してください。",
                is_error=True,
            )
        if has_path and has_base64:
            return ToolResult(
                content="image_path と rgb565_base64 は同時に指定できません。",
                is_error=True,
            )

        if has_path:
            loaded = self._load_from_path(image_path, width, height, x, y)
            if isinstance(loaded, ToolResult):
                return loaded
            rgb565, width, height = loaded
        else:
            if width is None or height is None:
                return ToolResult(
                    content="rgb565_base64 を使う場合は width と height が必須です。",
                    is_error=True,
                )
            try:
                rgb565 = base64.b64decode(rgb565_base64, validate=True)
            except (ValueError, binascii.Error):
                return ToolResult(content="rgb565_base64 のデコードに失敗しました。", is_error=True)

            expected_len = width * height * 2
            if len(rgb565) != expected_len:
                return ToolResult(
                    content=(
                        "RGB565データ長が不正です。"
                        f" expected={expected_len} bytes, actual={len(rgb565)} bytes"
                    ),
                    is_error=True,
                )

        if x + width > SCREEN_WIDTH or y + height > SCREEN_HEIGHT:
            return ToolResult(
                content="描画領域が画面外です。x+width<=135, y+height<=240 を満たしてください。",
                is_error=True,
            )

        pipelines = self._get_pipelines()
        if not pipelines:
            return ToolResult(content="接続中のデバイスがありません。", is_error=True)

        sent = 0
        for pipeline in list(pipelines):
            try:
                if clear:
                    await pipeline.send_display_text(text="", size=1, x=0, y=STATUS_BAR_Y, clear=True)
                await pipeline.send_display_image(
                    rgb565=rgb565,
                    width=width,
                    height=height,
                    x=x,
                    y=y,
                )
                sent += 1
            except Exception as e:
                logger.error(f"display_image送信エラー: {e}", exc_info=True)
                return ToolResult(content=f"画像表示送信エラー: {e}", is_error=True)

        logger.info(f"画像表示送信: devices={sent} width={width} height={height} x={x} y={y}")
        return ToolResult(content=f"画像を{sent}台のデバイスに表示しました。")
