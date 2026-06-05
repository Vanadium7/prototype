from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.fftpack import dct, idct


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
RESULT_DIR = ROOT / "results"

SECRET = b"aigc-trace-demo-secret"
PLATFORM_ID = "AIGC-DEMO"
MODEL_ID = "GEN-IMG-v1"
TIMESTAMP = "2026-05-31T21:00:00+08:00"
USER_ID = "student-user-001"

WATERMARK_BITS = 64
REPEAT = 7
BASELINE_REPEAT = 1
BLOCK = 8
COEFF_A = (3, 4)
COEFF_B = (4, 3)
THRESHOLD = 42.0
PHASH_SIZE = 8
PHASH_HIGH_SIMILAR = 8
PHASH_MEDIUM_SIMILAR = 12


@dataclass
class TraceRecord:
    platform_id: str
    model_id: str
    timestamp: str
    user_hash: str
    image_phash: str
    watermark_id: str
    signature: str


def ensure_dirs() -> None:
    for d in (INPUT_DIR, OUTPUT_DIR, LOG_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def make_demo_images() -> list[Path]:
    """Create deterministic image-like fixtures when no real AIGC images exist."""
    ensure_dirs()
    paths: list[Path] = []
    size = 512
    rng = np.random.default_rng(20260531)

    # Image 1: smooth gradient + abstract geometry.
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = (210 * xx + 40 * yy).astype(np.uint8)
    arr[..., 1] = (80 + 130 * yy).astype(np.uint8)
    arr[..., 2] = (180 - 100 * xx + 60 * yy).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse((72, 92, 290, 310), fill=(255, 240, 160, 70), outline=(35, 50, 90, 180), width=4)
    draw.rectangle((260, 160, 452, 360), fill=(40, 90, 140, 80), outline=(255, 255, 255, 160), width=3)
    p = INPUT_DIR / "demo_gradient_geometry.png"
    img.save(p)
    paths.append(p)

    # Image 2: textured synthetic scene.
    base = rng.normal(128, 38, (size, size, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base, "RGB").filter(ImageFilter.GaussianBlur(radius=1.1))
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(14):
        x0 = int(rng.integers(0, size - 90))
        y0 = int(rng.integers(0, size - 90))
        x1 = x0 + int(rng.integers(40, 130))
        y1 = y0 + int(rng.integers(40, 130))
        color = tuple(int(v) for v in rng.integers(40, 240, 3)) + (85,)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=color)
    p = INPUT_DIR / "demo_textured_scene.png"
    img.save(p)
    paths.append(p)

    # Image 3: high-frequency details + flat areas.
    arr = np.full((size, size, 3), 235, dtype=np.uint8)
    stripe = ((np.indices((size, size)).sum(axis=0) // 12) % 2) * 42
    arr[..., 0] -= stripe.astype(np.uint8)
    arr[..., 1] -= (stripe // 2).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.polygon([(60, 420), (230, 80), (420, 420)], fill=(40, 130, 110, 120), outline=(20, 60, 70, 200))
    draw.line((40, 70, 470, 120), fill=(220, 60, 70, 190), width=8)
    draw.line((70, 470, 480, 300), fill=(50, 70, 180, 180), width=8)
    p = INPUT_DIR / "demo_pattern_composite.png"
    img.save(p)
    paths.append(p)

    # Images 4-10: deterministic abstract "AIGC-like" fixtures with varied frequency profiles.
    palettes = [
        ((26, 40, 82), (96, 165, 250), (248, 250, 252)),
        ((49, 46, 129), (236, 72, 153), (253, 224, 71)),
        ((20, 83, 45), (132, 204, 22), (240, 253, 244)),
        ((127, 29, 29), (251, 146, 60), (255, 247, 237)),
        ((30, 41, 59), (45, 212, 191), (226, 232, 240)),
        ((88, 28, 135), (192, 132, 252), (250, 245, 255)),
        ((3, 105, 161), (125, 211, 252), (240, 249, 255)),
    ]
    for idx, (c0, c1, c2) in enumerate(palettes, start=4):
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        wave = 0.5 + 0.5 * np.sin((idx + 1) * np.pi * xx + (idx - 2) * np.pi * yy)
        radial = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        mix = np.clip(1 - radial * 1.7, 0, 1)
        arr = np.zeros((size, size, 3), dtype=np.float32)
        for ch in range(3):
            arr[..., ch] = (
                c0[ch] * (1 - xx) * (1 - mix)
                + c1[ch] * yy * wave
                + c2[ch] * mix
            )
        noise = rng.normal(0, 9 + idx, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=0.35))
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(6):
            cx = int(rng.integers(70, size - 70))
            cy = int(rng.integers(70, size - 70))
            rr = int(rng.integers(24, 82))
            color = tuple(int(v) for v in rng.integers(20, 250, 3)) + (70,)
            draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=color, outline=color[:3] + (135,), width=2)
        for _ in range(4):
            x0 = int(rng.integers(20, size - 80))
            y0 = int(rng.integers(20, size - 80))
            x1 = x0 + int(rng.integers(50, 170))
            y1 = y0 + int(rng.integers(30, 150))
            color = tuple(int(v) for v in rng.integers(30, 230, 3)) + (45,)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=color)
        p = INPUT_DIR / f"demo_aigc_like_{idx:02d}.png"
        img.save(p)
        paths.append(p)

    return paths


def make_additional_inputs(start_idx: int, target_count: int) -> None:
    """Create deterministic AIGC-like fixtures to reach the target input count."""
    ensure_dirs()
    size = 512
    rng = np.random.default_rng(20260602)
    for idx in range(start_idx, target_count + 1):
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        base_hue = (idx * 37) % 255
        arr = np.zeros((size, size, 3), dtype=np.float32)
        arr[..., 0] = (base_hue + 120 * xx + 40 * np.sin(idx * np.pi * yy)) % 255
        arr[..., 1] = (80 + 150 * yy + 35 * np.cos((idx + 2) * np.pi * xx)) % 255
        arr[..., 2] = (210 - 95 * xx + 75 * yy + 45 * np.sin((idx + 1) * np.pi * (xx + yy))) % 255
        texture = rng.normal(0, 12 + (idx % 5) * 3, arr.shape)
        arr = np.clip(arr + texture, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=0.25 + (idx % 3) * 0.15))
        draw = ImageDraw.Draw(img, "RGBA")
        for _ in range(8):
            cx = int(rng.integers(40, size - 40))
            cy = int(rng.integers(40, size - 40))
            rx = int(rng.integers(25, 110))
            ry = int(rng.integers(20, 95))
            color = tuple(int(v) for v in rng.integers(20, 245, 3)) + (int(rng.integers(45, 95)),)
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color, outline=color[:3] + (125,), width=2)
        for _ in range(5):
            x0 = int(rng.integers(20, size - 120))
            y0 = int(rng.integers(20, size - 120))
            x1 = x0 + int(rng.integers(70, 190))
            y1 = y0 + int(rng.integers(35, 155))
            color = tuple(int(v) for v in rng.integers(30, 230, 3)) + (int(rng.integers(35, 80)),)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=color)
        img.save(INPUT_DIR / f"aigc_{idx:02d}_synthetic.png")


def load_or_create_inputs() -> list[Path]:
    ensure_dirs()
    existing = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        existing.extend(INPUT_DIR.glob(suffix))
    if len(existing) < 10:
        existing = make_demo_images()
    if len(existing) < 20:
        make_additional_inputs(len(existing) + 1, 20)
        existing = []
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            existing.extend(INPUT_DIR.glob(suffix))
    return sorted(existing)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hmac_hex(data: str) -> str:
    return hmac.new(SECRET, data.encode("utf-8"), hashlib.sha256).hexdigest()


def hex_to_bits(hex_str: str, n_bits: int) -> np.ndarray:
    raw = bytes.fromhex(hex_str)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits[:n_bits].astype(np.uint8)


def bits_to_hex(bits: np.ndarray) -> str:
    padded_len = int(math.ceil(len(bits) / 8) * 8)
    padded = np.zeros(padded_len, dtype=np.uint8)
    padded[: len(bits)] = bits
    return np.packbits(padded).tobytes().hex()[: math.ceil(len(bits) / 4)]


def phash(image: Image.Image, hash_size: int = PHASH_SIZE, highfreq_factor: int = 4) -> str:
    size = hash_size * highfreq_factor
    img = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    coeff = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeff[:hash_size, :hash_size].copy()
    low[0, 0] = 0
    median = np.median(low)
    bits = (low > median).astype(np.uint8).flatten()
    return bits_to_hex(bits)


def hamming_hex(a: str, b: str) -> int:
    ba = np.unpackbits(np.frombuffer(bytes.fromhex(a), dtype=np.uint8))
    bb = np.unpackbits(np.frombuffer(bytes.fromhex(b), dtype=np.uint8))
    n = min(len(ba), len(bb))
    return int(np.sum(ba[:n] != bb[:n]) + abs(len(ba) - len(bb)))


def block_dct(block: np.ndarray) -> np.ndarray:
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def block_idct(block: np.ndarray) -> np.ndarray:
    return idct(idct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def choose_blocks(width: int, height: int, n: int, key: str) -> list[tuple[int, int]]:
    cols = width // BLOCK
    rows = height // BLOCK
    candidates = [(r, c) for r in range(rows) for c in range(cols)]
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[int(i)] for i in idx]


def adjust_pair(c1: float, c2: float, bit: int, threshold: float) -> tuple[float, float]:
    diff = c1 - c2
    if bit == 1 and diff < threshold:
        delta = (threshold - diff) / 2.0
        c1 += delta
        c2 -= delta
    elif bit == 0 and -diff < threshold:
        delta = (threshold + diff) / 2.0
        c1 -= delta
        c2 += delta
    return c1, c2


def embed_watermark(image: Image.Image, bits: np.ndarray, key: str, repeat: int = REPEAT) -> Image.Image:
    img = image.convert("YCbCr")
    y, cb, cr = img.split()
    y_arr = np.asarray(y, dtype=np.float32)
    h, w = y_arr.shape
    repeated = np.repeat(bits, repeat)
    positions = choose_blocks(w, h, len(repeated), key)
    out = y_arr.copy()

    for bit, (r, c) in zip(repeated, positions):
        y0, x0 = r * BLOCK, c * BLOCK
        coeff = block_dct(out[y0 : y0 + BLOCK, x0 : x0 + BLOCK])
        a = coeff[COEFF_A]
        b = coeff[COEFF_B]
        coeff[COEFF_A], coeff[COEFF_B] = adjust_pair(a, b, int(bit), THRESHOLD)
        out[y0 : y0 + BLOCK, x0 : x0 + BLOCK] = block_idct(coeff)

    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    merged = Image.merge("YCbCr", (Image.fromarray(out, "L"), cb, cr))
    return merged.convert("RGB")


def extract_watermark(image: Image.Image, n_bits: int, key: str, repeat: int = REPEAT) -> np.ndarray:
    img = image.convert("YCbCr")
    y = np.asarray(img.split()[0], dtype=np.float32)
    h, w = y.shape
    total = n_bits * repeat
    positions = choose_blocks(w, h, total, key)
    raw_bits = []
    for r, c in positions:
        y0, x0 = r * BLOCK, c * BLOCK
        coeff = block_dct(y[y0 : y0 + BLOCK, x0 : x0 + BLOCK])
        raw_bits.append(1 if coeff[COEFF_A] > coeff[COEFF_B] else 0)
    raw = np.array(raw_bits, dtype=np.uint8).reshape(n_bits, repeat)
    return (np.sum(raw, axis=1) >= (repeat / 2)).astype(np.uint8)


def psnr(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((arr_a - arr_b) ** 2))
    if mse == 0:
        return float("inf")
    return 10 * math.log10((255.0**2) / mse)


def ssim_global(a: Image.Image, b: Image.Image) -> float:
    """Simple global SSIM approximation; enough for a prototype table."""
    x = np.asarray(a.convert("L"), dtype=np.float64)
    y = np.asarray(b.convert("L"), dtype=np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))


def bit_metrics(original: np.ndarray, extracted: np.ndarray) -> tuple[float, float, float]:
    ber = float(np.mean(original != extracted))
    acc = 1.0 - ber
    a = original.astype(np.float64) * 2 - 1
    b = extracted.astype(np.float64) * 2 - 1
    nc = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return acc, ber, nc


def jpeg_attack(img: Image.Image, quality: int) -> Image.Image:
    tmp = OUTPUT_DIR / f"_tmp_q{quality}.jpg"
    img.save(tmp, "JPEG", quality=quality)
    out = Image.open(tmp).convert("RGB")
    tmp.unlink(missing_ok=True)
    return out


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(42)
    noisy = arr + rng.normal(0, sigma * 255, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGB")


def resize_attack(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((int(w * factor), int(h * factor)), Image.Resampling.BICUBIC)
    return small.resize((w, h), Image.Resampling.BICUBIC)


def crop_attack(img: Image.Image, pct: float) -> Image.Image:
    w, h = img.size
    dx, dy = int(w * pct), int(h * pct)
    cropped = img.crop((dx, dy, w - dx, h - dy))
    return cropped.resize((w, h), Image.Resampling.BICUBIC)


def occlusion_attack(img: Image.Image) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    draw.rectangle((int(w * 0.56), int(h * 0.12), int(w * 0.88), int(h * 0.44)), fill=(15, 15, 15, 235))
    draw.rectangle((int(w * 0.18), int(h * 0.62), int(w * 0.48), int(h * 0.82)), fill=(250, 250, 250, 210))
    return out


def brightness_attack(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def blur_attack(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def classify_risk(bit_acc: float, signature_ok: bool, log_ok: bool, phash_dist: int) -> str:
    watermark_ok = bit_acc >= 0.985
    watermark_partial = bit_acc >= 0.90
    if log_ok and not signature_ok:
        return "疑似水印伪造或来源冒用"
    if watermark_ok and signature_ok and log_ok and phash_dist <= PHASH_HIGH_SIMILAR:
        return "高可信来源一致"
    if watermark_ok and signature_ok and log_ok and phash_dist <= PHASH_MEDIUM_SIMILAR:
        return "来源可信，存在正常后处理或轻微编辑"
    if watermark_ok and signature_ok and log_ok and phash_dist > PHASH_MEDIUM_SIMILAR:
        return "来源记录存在，但内容疑似大幅篡改"
    if watermark_partial and signature_ok and log_ok:
        return "来源证据较强，但水印受损"
    if (not log_ok) and phash_dist <= PHASH_HIGH_SIMILAR:
        return "内容相似，但水印可能被破坏或移除"
    if phash_dist > PHASH_MEDIUM_SIMILAR:
        return "内容变化较大，无法可信溯源"
    return "证据不足，需要人工复核"


def make_record(image: Image.Image) -> TraceRecord:
    image_phash = phash(image)
    user_hash = hmac_hex(USER_ID)[:32]
    record_no_sig = "|".join([PLATFORM_ID, MODEL_ID, TIMESTAMP, user_hash, image_phash])
    signature = hmac_hex(record_no_sig)
    watermark_id = sha256_hex((record_no_sig + "|" + signature).encode("utf-8"))[:16]
    return TraceRecord(PLATFORM_ID, MODEL_ID, TIMESTAMP, user_hash, image_phash, watermark_id, signature)


def record_valid(record: TraceRecord) -> bool:
    record_no_sig = "|".join([record.platform_id, record.model_id, record.timestamp, record.user_hash, record.image_phash])
    return hmac.compare_digest(hmac_hex(record_no_sig), record.signature)


def run() -> None:
    ensure_dirs()
    image_paths = load_or_create_inputs()
    attacks: list[tuple[str, Callable[[Image.Image], Image.Image]]] = [
        ("无攻击", lambda im: im.copy()),
        ("JPEG Q=90", lambda im: jpeg_attack(im, 90)),
        ("JPEG Q=70", lambda im: jpeg_attack(im, 70)),
        ("JPEG Q=50", lambda im: jpeg_attack(im, 50)),
        ("高斯噪声 sigma=0.01", lambda im: gaussian_noise(im, 0.01)),
        ("缩放 0.5x", lambda im: resize_attack(im, 0.5)),
        ("裁剪 10%", lambda im: crop_attack(im, 0.10)),
        ("局部遮挡篡改", occlusion_attack),
        ("亮度 +20%", lambda im: brightness_attack(im, 1.20)),
        ("模糊 r=1.2", lambda im: blur_attack(im, 1.2)),
    ]

    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []

    for path in image_paths:
        original = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        record = make_record(original)
        key = record.watermark_id
        wm_bits = hex_to_bits(record.watermark_id, WATERMARK_BITS)
        watermarked = embed_watermark(original, wm_bits, key)
        baseline_watermarked = embed_watermark(original, wm_bits, key, repeat=BASELINE_REPEAT)

        base_name = path.stem
        watermarked_path = OUTPUT_DIR / f"{base_name}_watermarked.png"
        watermarked.save(watermarked_path)
        log_path = LOG_DIR / f"{base_name}_record.json"
        log_path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")

        for attack_name, attack_fn in attacks:
            attacked = attack_fn(watermarked)
            row = evaluate_case(base_name, attack_name, original, attacked, record, wm_bits, key)
            all_rows.append(row)

            baseline_attacked = attack_fn(baseline_watermarked)
            baseline_row = evaluate_case(
                base_name,
                attack_name,
                original,
                baseline_attacked,
                record,
                wm_bits,
                key,
                repeat=BASELINE_REPEAT,
            )
            baseline_row["method"] = "DCT 单次嵌入"
            baseline_rows.append(baseline_row)

            safe = safe_name(attack_name)
            attacked_path = OUTPUT_DIR / f"{base_name}_{safe}.png"
            attacked.save(attacked_path)

        # Signature/log forgery scenario: image and watermark are intact, but the log signature is tampered.
        tampered = TraceRecord(**asdict(record))
        tampered.signature = "0" * len(tampered.signature)
        row = evaluate_case(base_name, "日志签名篡改", original, watermarked, tampered, wm_bits, key)
        all_rows.append(row)
        baseline_row = evaluate_case(
            base_name,
            "日志签名篡改",
            original,
            baseline_watermarked,
            tampered,
            wm_bits,
            key,
            repeat=BASELINE_REPEAT,
        )
        baseline_row["method"] = "DCT 单次嵌入"
        baseline_rows.append(baseline_row)

    make_contact_sheet(image_paths[0])

    # Aggregate by attack.
    attack_order = [name for name, _ in attacks] + ["日志签名篡改"]
    for attack_name in attack_order:
        rows = [r for r in all_rows if r["attack"] == attack_name]
        if not rows:
            continue
        summary_rows.append(
            {
                "attack": attack_name,
                "psnr": round(float(np.mean([r["psnr"] for r in rows])), 3),
                "ssim": round(float(np.mean([r["ssim"] for r in rows])), 4),
                "bit_accuracy": round(float(np.mean([r["bit_accuracy"] for r in rows])), 4),
                "ber": round(float(np.mean([r["ber"] for r in rows])), 4),
                "nc": round(float(np.mean([r["nc"] for r in rows])), 4),
                "phash_distance": round(float(np.mean([r["phash_distance"] for r in rows])), 2),
                "log_match_rate": round(float(np.mean([1 if r["log_match"] == "是" else 0 for r in rows])), 3),
                "dominant_risk": max(set(r["risk"] for r in rows), key=[r["risk"] for r in rows].count),
            }
        )

    write_csv(RESULT_DIR / "detailed_results.csv", all_rows)
    write_csv(RESULT_DIR / "summary_results.csv", summary_rows)
    write_markdown(RESULT_DIR / "summary_results.md", summary_rows, all_rows, len(image_paths))
    write_json_samples(all_rows)
    write_baseline_comparison(baseline_rows, summary_rows)
    write_ablation_results(all_rows)


def safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("=", "").replace("%", "pct").replace("+", "plus").replace("/", "_")


def evaluate_case(
    base_name: str,
    attack_name: str,
    original: Image.Image,
    attacked: Image.Image,
    verify_record: TraceRecord,
    wm_bits: np.ndarray,
    key: str,
    repeat: int = REPEAT,
) -> dict[str, object]:
    extracted = extract_watermark(attacked, WATERMARK_BITS, key, repeat=repeat)
    extracted_hex = bits_to_hex(extracted)
    bit_acc, ber, nc = bit_metrics(wm_bits, extracted)
    phash_dist = hamming_hex(verify_record.image_phash, phash(attacked))
    log_match = extracted_hex == verify_record.watermark_id
    signature_ok = record_valid(verify_record) and log_match
    risk = classify_risk(bit_acc, signature_ok, log_match, phash_dist)
    return {
        "image": base_name,
        "attack": attack_name,
        "psnr": round(psnr(original, attacked), 3),
        "ssim": round(ssim_global(original, attacked), 4),
        "bit_accuracy": round(bit_acc, 4),
        "ber": round(ber, 4),
        "nc": round(nc, 4),
        "phash_distance": phash_dist,
        "log_match": "是" if log_match else "否",
        "signature_ok": "是" if signature_ok else "否",
        "risk": risk,
    }


def make_contact_sheet(first_input: Path) -> None:
    base = first_input.stem
    names = [
        ("Original", INPUT_DIR / f"{base}.png"),
        ("Watermarked", OUTPUT_DIR / f"{base}_watermarked.png"),
        ("JPEG Q=50", OUTPUT_DIR / f"{base}_JPEG_Q50.png"),
        ("Resize 0.5x", OUTPUT_DIR / f"{base}_缩放_0.5x.png"),
        ("Crop 10%", OUTPUT_DIR / f"{base}_裁剪_10pct.png"),
        ("Occlusion", OUTPUT_DIR / f"{base}_局部遮挡篡改.png"),
    ]
    thumb_w, thumb_h = 220, 220
    label_h = 30
    sheet = Image.new("RGB", (thumb_w * 3, (thumb_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, p) in enumerate(names):
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (idx % 3) * thumb_w
        y = (idx // 3) * (thumb_h + label_h)
        sheet.paste(img, (x, y))
        draw.text((x + 8, y + thumb_h + 7), label, fill=(0, 0, 0))
    sheet.save(RESULT_DIR / "visual_comparison.png")


def write_json_samples(rows: list[dict[str, object]]) -> None:
    interesting = [
        r for r in rows
        if r["attack"] in {"无攻击", "JPEG Q=50", "缩放 0.5x", "裁剪 10%", "局部遮挡篡改", "日志签名篡改"}
        and r["image"] == rows[0]["image"]
    ]
    (RESULT_DIR / "verification_trace_samples.json").write_text(
        json.dumps(interesting, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    attack_order = [
        "无攻击",
        "JPEG Q=90",
        "JPEG Q=70",
        "JPEG Q=50",
        "高斯噪声 sigma=0.01",
        "缩放 0.5x",
        "裁剪 10%",
        "局部遮挡篡改",
        "亮度 +20%",
        "模糊 r=1.2",
        "日志签名篡改",
    ]
    out: list[dict[str, object]] = []
    for attack_name in attack_order:
        group = [r for r in rows if r["attack"] == attack_name]
        if not group:
            continue
        out.append(
            {
                "attack": attack_name,
                "bit_accuracy": round(float(np.mean([r["bit_accuracy"] for r in group])), 4),
                "ber": round(float(np.mean([r["ber"] for r in group])), 4),
                "nc": round(float(np.mean([r["nc"] for r in group])), 4),
                "log_match_rate": round(float(np.mean([1 if r["log_match"] == "是" else 0 for r in group])), 3),
            }
        )
    return out


def write_baseline_comparison(baseline_rows: list[dict[str, object]], proposed_summary: list[dict[str, object]]) -> None:
    baseline_summary = summarize_rows(baseline_rows)
    by_attack = {r["attack"]: r for r in baseline_summary}
    rows: list[dict[str, object]] = []
    for proposed in proposed_summary:
        attack = proposed["attack"]
        baseline = by_attack.get(attack)
        if not baseline:
            continue
        rows.append(
            {
                "attack": attack,
                "proposed_bit_accuracy": proposed["bit_accuracy"],
                "baseline_bit_accuracy": baseline["bit_accuracy"],
                "proposed_ber": proposed["ber"],
                "baseline_ber": baseline["ber"],
                "proposed_log_match_rate": proposed["log_match_rate"],
                "baseline_log_match_rate": baseline["log_match_rate"],
            }
        )
    write_csv(RESULT_DIR / "baseline_comparison.csv", rows)
    lines = [
        "# Baseline 对比结果",
        "",
        "| 攻击类型 | 本文方案 Bit Accuracy | Baseline Bit Accuracy | 本文方案 BER | Baseline BER | 本文方案日志匹配率 | Baseline 日志匹配率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['attack']} | {r['proposed_bit_accuracy']} | {r['baseline_bit_accuracy']} | {r['proposed_ber']} | {r['baseline_ber']} | {r['proposed_log_match_rate']} | {r['baseline_log_match_rate']} |"
        )
    (RESULT_DIR / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def state_from_attack(attack: str) -> str:
    if attack in {"无攻击", "JPEG Q=90", "JPEG Q=70", "高斯噪声 sigma=0.01", "亮度 +20%", "模糊 r=1.2"}:
        return "正常传播"
    if attack in {"JPEG Q=50", "缩放 0.5x"}:
        return "水印受损但内容相似"
    if attack in {"裁剪 10%", "局部遮挡篡改"}:
        return "内容编辑或变化"
    if attack == "日志签名篡改":
        return "来源冒用或记录伪造"
    return "其他"


def state_from_risk(risk: str) -> str:
    if risk == "高可信来源一致":
        return "正常传播"
    if "水印可能" in risk:
        return "水印受损但内容相似"
    if "内容" in risk and ("篡改" in risk or "变化" in risk):
        return "内容编辑或变化"
    if "伪造" in risk or "冒用" in risk:
        return "来源冒用或记录伪造"
    return "其他"


def write_ablation_results(rows: list[dict[str, object]]) -> None:
    categories = ["正常传播", "水印受损但内容相似", "内容编辑或变化", "来源冒用或记录伪造"]

    def log_ok(row: dict[str, object]) -> bool:
        return row["log_match"] == "是"

    def sig_ok(row: dict[str, object]) -> bool:
        return row["signature_ok"] == "是"

    def phash_sim(row: dict[str, object]) -> bool:
        return int(row["phash_distance"]) <= PHASH_HIGH_SIMILAR

    def phash_changed(row: dict[str, object]) -> bool:
        return int(row["phash_distance"]) > PHASH_MEDIUM_SIMILAR

    def pred_watermark(row: dict[str, object]) -> str:
        return "正常传播" if log_ok(row) else "其他"

    def pred_watermark_phash(row: dict[str, object]) -> str:
        if log_ok(row) and phash_sim(row):
            return "正常传播"
        if log_ok(row) and phash_changed(row):
            return "内容编辑或变化"
        if (not log_ok(row)) and phash_sim(row):
            return "水印受损但内容相似"
        if phash_changed(row):
            return "内容编辑或变化"
        return "其他"

    def pred_watermark_signed_log(row: dict[str, object]) -> str:
        if log_ok(row) and sig_ok(row):
            return "正常传播"
        if log_ok(row) and not sig_ok(row):
            return "来源冒用或记录伪造"
        return "其他"

    def pred_all(row: dict[str, object]) -> str:
        return state_from_risk(str(row["risk"]))

    methods = [
        ("仅水印/日志匹配", pred_watermark),
        ("水印 + pHash", pred_watermark_phash),
        ("水印 + 签名日志", pred_watermark_signed_log),
        ("水印 + pHash + 签名日志", pred_all),
    ]
    result_rows: list[dict[str, object]] = []
    for name, fn in methods:
        correct = 0
        counts = {cat: [0, 0] for cat in categories}
        for row in rows:
            expected = state_from_attack(str(row["attack"]))
            predicted = fn(row)
            if expected in counts:
                counts[expected][1] += 1
            if predicted == expected:
                correct += 1
                if expected in counts:
                    counts[expected][0] += 1
        total = len(rows)
        result_rows.append(
            {
                "evidence": name,
                "correct_total": f"{correct}/{total}",
                "accuracy": f"{100 * correct / total:.1f}%",
                "normal": f"{counts['正常传播'][0]}/{counts['正常传播'][1]}",
                "watermark_damaged": f"{counts['水印受损但内容相似'][0]}/{counts['水印受损但内容相似'][1]}",
                "content_changed": f"{counts['内容编辑或变化'][0]}/{counts['内容编辑或变化'][1]}",
                "source_forgery": f"{counts['来源冒用或记录伪造'][0]}/{counts['来源冒用或记录伪造'][1]}",
            }
        )
    lines = [
        "# 消融对比结果",
        "",
        f"本消融实验比较不同证据组合在 {len(rows)} 个测试样本上的风险状态识别能力。",
        "",
        "| 证据组合 | 正确数 / 总数 | 准确率 | 正常传播 | 水印受损但内容相似 | 内容编辑 / 变化 | 来源冒用 / 记录伪造 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result_rows:
        lines.append(
            f"| {r['evidence']} | {r['correct_total']} | {r['accuracy']} | {r['normal']} | {r['watermark_damaged']} | {r['content_changed']} | {r['source_forgery']} |"
        )
    (RESULT_DIR / "ablation_results.md").write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, object]], details: list[dict[str, object]], n_images: int) -> None:
    headers = ["attack", "psnr", "ssim", "bit_accuracy", "ber", "nc", "phash_distance", "log_match_rate", "dominant_risk"]
    lines = [
        "# AIGC Trace Prototype Results",
        "",
        f"- Images: {n_images}",
        f"- Watermark bits: {WATERMARK_BITS}",
        f"- Repetition factor: {REPEAT}",
        f"- DCT threshold: {THRESHOLD}",
        "",
        "## Summary",
        "",
        "| 攻击类型 | PSNR | SSIM | Bit Accuracy | BER | NC | pHash距离 | 日志匹配率 | 主要风险判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['attack']} | {r['psnr']} | {r['ssim']} | {r['bit_accuracy']} | {r['ber']} | {r['nc']} | {r['phash_distance']} | {r['log_match_rate']} | {r['dominant_risk']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `log_match_rate` is strict exact matching between extracted WatermarkID and the platform log.",
            "- Partial watermark recovery is still reported through Bit Accuracy, BER, and NC.",
            "- Risk labels combine watermark recovery, signature/log matching, and pHash distance.",
            "",
            "## Detailed CSV",
            "",
            "- `detailed_results.csv` contains per-image rows.",
            "- `summary_results.csv` contains attack-level averages.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
