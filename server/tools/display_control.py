import base64
import binascii
import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import httpx

from tools.base import ToolBase, ToolResult

logger = logging.getLogger(__name__)

SCREEN_WIDTH = 135
SCREEN_HEIGHT = 240
STATUS_BAR_Y = 10
HTTP_TIMEOUT_SEC = 20.0

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

try:
    import resvg_py
    _resvg_error = ""
except Exception as e:  # pragma: no cover
    resvg_py = None
    _resvg_error = str(e)


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


def _calc_target_size(
    raw_w: int,
    raw_h: int,
    req_width: int | None,
    req_height: int | None,
    max_w: int,
    max_h: int,
) -> tuple[int, int]:
    if raw_w <= 0 or raw_h <= 0:
        return 0, 0

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

    if width > max_w or height > max_h:
        width, height = _fit_size(width, height, max_w, max_h)

    return width, height


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


def _render_pil_to_rgb565(
    image,
    req_width: int | None,
    req_height: int | None,
    x: int,
    y: int,
) -> tuple[bytes, int, int] | ToolResult:
    if Image is None:
        return ToolResult(content="Pillow未インストールのため画像描画できません。", is_error=True)

    raw_w, raw_h = image.size
    max_w = SCREEN_WIDTH - x
    max_h = SCREEN_HEIGHT - y
    width, height = _calc_target_size(raw_w, raw_h, req_width, req_height, max_w, max_h)

    if width <= 0 or height <= 0:
        return ToolResult(content="描画可能な画像サイズを計算できませんでした。", is_error=True)

    if (width, height) != image.size:
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((width, height), resampling.LANCZOS)

    image = image.convert("RGB")
    rgb565 = _rgb888_to_rgb565_be(image.tobytes())
    return rgb565, width, height


def _load_raster_bytes(
    data: bytes,
    req_width: int | None,
    req_height: int | None,
    x: int,
    y: int,
) -> tuple[bytes, int, int] | ToolResult:
    if Image is None:
        return ToolResult(content="Pillow未インストールのため画像表示できません。", is_error=True)

    try:
        with Image.open(io.BytesIO(data)) as img:
            rgb = img.convert("RGB")
            return _render_pil_to_rgb565(rgb, req_width, req_height, x, y)
    except Exception as e:
        return ToolResult(content=f"画像デコードエラー: {e}", is_error=True)


def _load_svg_text(
    svg_text: str,
    req_width: int | None,
    req_height: int | None,
    x: int,
    y: int,
) -> tuple[bytes, int, int] | ToolResult:
    if resvg_py is None:
        detail = f" ({_resvg_error})" if _resvg_error else ""
        return ToolResult(
            content=(
                "SVG表示には resvg-py が必要です。"
                f"`cd server && uv sync` を実行してください。{detail}"
            ),
            is_error=True,
        )

    try:
        png_bytes = resvg_py.svg_to_bytes(
            svg_string=svg_text,
            width=req_width,
            height=req_height,
        )
    except Exception as e:
        return ToolResult(content=f"SVGレンダリングエラー: {e}", is_error=True)

    # SVG側でサイズ指定済みなので、ここでは画面内フィットのみ行う
    return _load_raster_bytes(png_bytes, None, None, x, y)


def _fetch_url_bytes(url: str) -> tuple[bytes, str] | ToolResult:
    if not url.lower().startswith(("http://", "https://")):
        return ToolResult(content="image_url は http:// または https:// を指定してください。", is_error=True)

    try:
        response = httpx.get(url, follow_redirects=True, timeout=HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        return response.content, content_type
    except Exception as e:
        return ToolResult(content=f"URL取得エラー: {e}", is_error=True)


def _load_from_url(
    image_url: str,
    req_width: int | None,
    req_height: int | None,
    x: int,
    y: int,
) -> tuple[bytes, int, int] | ToolResult:
    fetched = _fetch_url_bytes(image_url)
    if isinstance(fetched, ToolResult):
        return fetched
    data, content_type = fetched

    looks_like_svg = (
        "svg" in content_type
        or image_url.split("?", 1)[0].lower().endswith(".svg")
        or data.lstrip().startswith(b"<svg")
    )

    if looks_like_svg:
        try:
            svg_text = data.decode("utf-8")
        except UnicodeDecodeError:
            svg_text = data.decode("utf-8", errors="replace")
        return _load_svg_text(svg_text, req_width, req_height, x, y)

    return _load_raster_bytes(data, req_width, req_height, x, y)


def _render_mermaid_png(mermaid: str) -> bytes | ToolResult:
    try:
        response = httpx.post(
            "https://kroki.io/mermaid/png",
            content=mermaid.encode("utf-8"),
            headers={"content-type": "text/plain; charset=utf-8"},
            timeout=HTTP_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return response.content
    except Exception as first_error:
        try:
            encoded = base64.urlsafe_b64encode(mermaid.encode("utf-8")).decode("ascii")
            response = httpx.get(
                f"https://mermaid.ink/img/{encoded}",
                timeout=HTTP_TIMEOUT_SEC,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content
        except Exception as second_error:
            return ToolResult(
                content=f"Mermaidレンダリングエラー: {first_error} / {second_error}",
                is_error=True,
            )


def _load_from_mermaid(
    mermaid: str,
    req_width: int | None,
    req_height: int | None,
    x: int,
    y: int,
) -> tuple[bytes, int, int] | ToolResult:
    rendered = _render_mermaid_png(mermaid)
    if isinstance(rendered, ToolResult):
        return rendered
    return _load_raster_bytes(rendered, req_width, req_height, x, y)


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
        "image_path/image_url/svg/mermaid/rgb565_base64 のいずれか1つを指定してください。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "表示するローカル画像ファイルのパス (PNG/JPEG/SVG等)",
            },
            "image_url": {
                "type": "string",
                "description": "表示する画像URL (http/https)。SVG URLも可。",
            },
            "url": {
                "type": "string",
                "description": "image_url のエイリアス。",
            },
            "svg": {
                "type": "string",
                "description": "SVG文字列をそのまま指定して表示します。",
            },
            "mermaid": {
                "type": "string",
                "description": "Mermaid構文を指定して図を表示します。",
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
                "description": "画像幅 (1-135)。省略時は元画像サイズ/比率を使用。",
                "minimum": 1,
                "maximum": 135,
            },
            "height": {
                "type": "integer",
                "description": "画像高 (1-240)。省略時は元画像サイズ/比率を使用。",
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
        path = Path(image_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_file():
            return ToolResult(content=f"画像ファイルが見つかりません: {path}", is_error=True)

        if path.suffix.lower() == ".svg":
            try:
                svg_text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                svg_text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return ToolResult(content=f"SVG読み込みエラー: {e}", is_error=True)
            return _load_svg_text(svg_text, req_width, req_height, x, y)

        try:
            data = path.read_bytes()
        except Exception as e:
            return ToolResult(content=f"画像ファイル読み込みエラー: {e}", is_error=True)
        return _load_raster_bytes(data, req_width, req_height, x, y)

    async def execute(self, **kwargs) -> ToolResult:
        image_path = kwargs.get("image_path")
        image_url = kwargs.get("image_url")
        url_alias = kwargs.get("url")
        svg = kwargs.get("svg")
        mermaid = kwargs.get("mermaid")
        rgb565_base64 = kwargs.get("rgb565_base64")

        width = kwargs.get("width")
        height = kwargs.get("height")
        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        clear = kwargs.get("clear", False)

        if (not image_url) and isinstance(url_alias, str) and url_alias.strip() != "":
            image_url = url_alias

        if isinstance(image_path, str) and image_path.lower().startswith(("http://", "https://")):
            if not image_url:
                image_url = image_path
                image_path = None

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
        has_url = isinstance(image_url, str) and image_url.strip() != ""
        has_svg = isinstance(svg, str) and svg.strip() != ""
        has_mermaid = isinstance(mermaid, str) and mermaid.strip() != ""
        has_base64 = isinstance(rgb565_base64, str) and rgb565_base64 != ""

        source_count = sum((has_path, has_url, has_svg, has_mermaid, has_base64))
        if source_count == 0:
            return ToolResult(
                content=(
                    "image_path/image_url/svg/mermaid/rgb565_base64 のいずれか1つを指定してください。"
                ),
                is_error=True,
            )
        if source_count > 1:
            return ToolResult(
                content=(
                    "入力ソースは1つだけ指定してください。"
                    "(image_path, image_url, svg, mermaid, rgb565_base64 は排他的)"
                ),
                is_error=True,
            )

        if has_path:
            loaded = self._load_from_path(image_path, width, height, x, y)
            if isinstance(loaded, ToolResult):
                return loaded
            rgb565, width, height = loaded
        elif has_url:
            loaded = _load_from_url(image_url, width, height, x, y)
            if isinstance(loaded, ToolResult):
                return loaded
            rgb565, width, height = loaded
        elif has_svg:
            loaded = _load_svg_text(svg, width, height, x, y)
            if isinstance(loaded, ToolResult):
                return loaded
            rgb565, width, height = loaded
        elif has_mermaid:
            loaded = _load_from_mermaid(mermaid, width, height, x, y)
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
