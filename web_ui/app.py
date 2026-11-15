import base64
import io
import math
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import torch and vietocr BEFORE paddle so Windows loads the right OpenMP DLLs
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor as VietOCRPredictor

import cv2
import numpy as np
import paddle
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

from paddleocr import PaddleOCR, TextDetection

# Optimize CPU threads for PyTorch when running VietOCR
torch.set_num_threads(max(4, (os.cpu_count() or 8) - 2))

app = FastAPI(
    title="PaddleOCR Web Studio",
    description="Giao diện Web nhận diện văn bản (OCR) Tiếng Việt & Đa ngôn ngữ hỗ trợ xử lý hàng loạt tốc độ cao",
    version="3.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

_ocr_lock = threading.Lock()
_det_instances: Dict[str, TextDetection] = {}
_ocr_instances: Dict[tuple, PaddleOCR] = {}
_vietocr_instances: Dict[str, VietOCRPredictor] = {}


def resolve_paddle_device(device_choice: str) -> str:
    gpu_ok = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    if device_choice == "cpu":
        return "cpu"
    if device_choice in ("gpu", "auto"):
        return "gpu:0" if gpu_ok else "cpu"
    return "gpu:0" if gpu_ok else "cpu"


def get_det_engine(device_choice: str = "gpu") -> TextDetection:
    p_dev = resolve_paddle_device(device_choice)
    with _ocr_lock:
        if p_dev not in _det_instances:
            print(f"[PaddleOCR Web] Đang khởi tạo TextDetection (PP-OCRv6_medium_det) trên {p_dev.upper()}...")
            _det_instances[p_dev] = TextDetection(
                model_name="PP-OCRv6_medium_det",
                device=p_dev,
            )
        return _det_instances[p_dev]


def get_ocr_engine(lang: str = "en", ocr_version: str = "PP-OCRv6", device_choice: str = "gpu") -> PaddleOCR:
    p_dev = resolve_paddle_device(device_choice)
    actual_lang = "vi" if lang in ("vi_fast", "vi_pro") else lang
    actual_ver = "PP-OCRv6" if ocr_version in ("VietOCR-Seq2Seq", "VietOCR-Transformer") else ocr_version
    key = (actual_lang, actual_ver, p_dev)
    with _ocr_lock:
        if key not in _ocr_instances:
            print(f"[PaddleOCR Web] Đang khởi tạo PaddleOCR ({actual_lang}, {actual_ver}) trên {p_dev.upper()}...")
            _ocr_instances[key] = PaddleOCR(
                lang=actual_lang,
                ocr_version=actual_ver,
                device=p_dev,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return _ocr_instances[key]


def get_vietocr_engine(mode: str = "seq2seq") -> VietOCRPredictor:
    """
    mode: 'seq2seq' (Siêu tốc, nhẹ, nhanh gấp 2.5x) hoặc 'transformer' (Độ chính xác cao nhất)
    """
    with _ocr_lock:
        if mode not in _vietocr_instances:
            base_cfg_path = BASE_DIR / "base.yml"
            with open(base_cfg_path, encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)

            if mode == "seq2seq":
                print("[PaddleOCR Web] Đang khởi tạo VietOCR Seq2Seq (Chế độ Siêu tốc)...")
                seq_cfg_path = BASE_DIR / "vgg_seq2seq.yml"
                with open(seq_cfg_path, encoding="utf-8") as f:
                    cfg_dict.update(yaml.safe_load(f))
                weights_path = hf_hub_download(
                    repo_id="lenhanm1/enterprise-printed-ocr-vietocr-tesseract",
                    filename="vietocr/vgg-seq2seq-enterprise-print.pth",
                )
            else:
                print("[PaddleOCR Web] Đang khởi tạo VietOCR Transformer (Chế độ Chính xác cao)...")
                trans_cfg_path = BASE_DIR / "vgg_transformer.yml"
                with open(trans_cfg_path, encoding="utf-8") as f:
                    cfg_dict.update(yaml.safe_load(f))
                weights_path = hf_hub_download(
                    repo_id="doanhm/vgg_transformer",
                    filename="vgg_transformer.pth",
                )

            cfg = Cfg(cfg_dict)
            cfg["cnn"]["pretrained"] = False
            cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
            cfg["predictor"]["beamsearch"] = False
            cfg["weights"] = weights_path
            # Quantize width into 32px steps so batching groups lines together for 3x faster batch inference!
            cfg["dataset"]["image_min_width"] = 32
            cfg["dataset"]["image_max_width"] = 768

            _vietocr_instances[mode] = VietOCRPredictor(cfg)
            print(f"[PaddleOCR Web] Đã nạp xong VietOCR ({mode})!")
        return _vietocr_instances[mode]


def crop_polygon_image(img_bgr: np.ndarray, poly: List[List[float]]) -> Image.Image:
    """Perspective-crop a 4-point text box into a clean horizontal PIL image with quantized width for fast batching."""
    pts = np.array(poly, dtype=np.float32)
    h_img, w_img = img_bgr.shape[:2]

    if len(pts) == 4:
        rect = pts
        w1 = np.linalg.norm(rect[0] - rect[1])
        w2 = np.linalg.norm(rect[3] - rect[2])
        h1 = np.linalg.norm(rect[0] - rect[3])
        h2 = np.linalg.norm(rect[1] - rect[2])
        crop_w = max(int(max(w1, w2)), 8)
        crop_h = max(int(max(h1, h2)), 8)

        pad_y = max(2, int(crop_h * 0.08))
        pad_x = max(2, int(crop_w * 0.02))
        dst = np.array(
            [
                [pad_x, pad_y],
                [crop_w + pad_x, pad_y],
                [crop_w + pad_x, crop_h + pad_y],
                [pad_x, crop_h + pad_y],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(
            img_bgr,
            M,
            (crop_w + 2 * pad_x, crop_h + 2 * pad_y),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if crop_h > crop_w * 2.2 and crop_h > 40:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    else:
        x_min = max(0, int(np.min(pts[:, 0])) - 2)
        y_min = max(0, int(np.min(pts[:, 1])) - 3)
        x_max = min(w_img, int(np.max(pts[:, 0])) + 2)
        y_max = min(h_img, int(np.max(pts[:, 1])) + 3)
        crop = img_bgr[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return Image.new("RGB", (64, 32), (255, 255, 255))
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    # Pad width to nearest 64px bucket so VietOCR processes many lines in a single matrix multiplication!
    w, h = pil_img.size
    if h > 0:
        scaled_w = max(32, int(round(32.0 * w / h)))
        bucket_w = int(math.ceil(scaled_w / 64.0) * 64)
        target_orig_w = int(round(bucket_w * h / 32.0))
        if target_orig_w > w:
            padded = Image.new("RGB", (target_orig_w, h), (255, 255, 255))
            padded.paste(pil_img, (0, 0))
            return padded
    return pil_img


def reconstruct_reading_layout(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Sort detected text boxes into natural top-to-bottom, left-to-right reading lines."""
    if not items:
        return [], ""

    enriched = []
    for it in items:
        poly = it["poly"]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        y_center = (y_min + y_max) / 2.0
        box_h = max(8.0, y_max - y_min)
        enriched.append(
            {
                **it,
                "_x_min": x_min,
                "_x_max": x_max,
                "_y_min": y_min,
                "_y_max": y_max,
                "_y_center": y_center,
                "_h": box_h,
            }
        )

    enriched.sort(key=lambda b: b["_y_center"])

    rows: List[List[Dict[str, Any]]] = []
    for box in enriched:
        if not rows:
            rows.append([box])
            continue
        last_row = rows[-1]
        avg_y = sum(b["_y_center"] for b in last_row) / len(last_row)
        avg_h = sum(b["_h"] for b in last_row) / len(last_row)
        if abs(box["_y_center"] - avg_y) <= avg_h * 0.55:
            last_row.append(box)
        else:
            rows.append([box])

    ordered_items: List[Dict[str, Any]] = []
    formatted_lines: List[str] = []
    new_id = 1

    for row in rows:
        row.sort(key=lambda b: b["_x_min"])
        line_parts = []
        prev_x_max = None
        avg_h = sum(b["_h"] for b in row) / len(row)

        for b in row:
            txt = b["text"].strip()
            if not txt:
                continue
            if prev_x_max is not None:
                gap = b["_x_min"] - prev_x_max
                if gap > avg_h * 1.4:
                    line_parts.append("\t")
                else:
                    line_parts.append(" ")
            line_parts.append(txt)
            prev_x_max = b["_x_max"]

            clean_item = {
                "id": new_id,
                "text": b["text"],
                "confidence": b["confidence"],
                "poly": b["poly"],
            }
            ordered_items.append(clean_item)
            new_id += 1

        if line_parts:
            formatted_lines.append("".join(line_parts))

    return ordered_items, "\n".join(formatted_lines)


def draw_annotated_image(image_bgr: np.ndarray, items: List[Dict[str, Any]]) -> str:
    """Draw bounding polygons and index numbers onto the image and return base64 JPEG."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base_pil = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    palette = [
        (59, 130, 246),   # Blue
        (16, 185, 129),   # Emerald
        (245, 158, 11),   # Amber
        (236, 72, 153),   # Pink
        (139, 92, 246),   # Violet
        (6, 182, 212),    # Cyan
    ]

    for idx, item in enumerate(items):
        poly = item.get("poly")
        if not poly or len(poly) < 3:
            continue
        color = palette[idx % len(palette)]
        pts = [(int(p[0]), int(p[1])) for p in poly]
        draw.polygon(pts, fill=(*color, 45), outline=(*color, 230), width=3)

        x0, y0 = pts[0]
        badge_text = str(item.get("id", idx + 1))
        badge_w = max(20, 8 * len(badge_text) + 10)
        badge_h = 20
        bx0 = max(0, x0)
        by0 = max(0, y0 - badge_h)
        draw.rectangle([bx0, by0, bx0 + badge_w, by0 + badge_h], fill=(*color, 235))
        draw.text((bx0 + 5, by0 + 3), badge_text, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(base_pil, overlay).convert("RGB")
    buf = io.BytesIO()
    combined.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def create_sample_image() -> np.ndarray:
    """Generate a sample image with complex Vietnamese diacritics and table numbers."""
    w, h = 920, 480
    img = Image.new("RGB", (w, h), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    font_title = None
    font_body = None
    font_small = None
    for font_name in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"]:
        if os.path.exists(font_name):
            font_title = ImageFont.truetype(font_name, 32)
            break
    for font_name in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        if os.path.exists(font_name):
            font_body = ImageFont.truetype(font_name, 23)
            font_small = ImageFont.truetype(font_name, 18)
            break
    if font_title is None:
        font_title = ImageFont.load_default()
    if font_body is None:
        font_body = ImageFont.load_default()
    if font_small is None:
        font_small = ImageFont.load_default()

    draw.rounded_rectangle([30, 30, w - 30, h - 30], radius=16, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.rounded_rectangle([30, 30, w - 30, 105], radius=16, fill=(30, 58, 138))
    draw.rectangle([30, 80, w - 30, 105], fill=(30, 58, 138))

    draw.text((55, 50), "BÁO CÁO SẢN LƯỢNG & DIỆN TÍCH THU HOẠCH", fill=(255, 255, 255), font=font_title)

    lines = [
        ("Chuyên đề:", "Thống kê diện tích trồng và năng suất sầu riêng (2015–2024)"),
        ("Đặc điểm:", "Do sự khác biệt về mặt địa lý và điều kiện tự nhiên ở miền Nam"),
        ("Sản lượng:", "Sản lượng thu hoạch trong vụ thuận chiếm khoảng 90% tổng số"),
        ("Kỹ thuật:", "Áp dụng kỹ thuật sản xuất vụ nghịch xuất khẩu đạt tiêu chuẩn"),
        ("Thị trường:", "Khả năng đáp ứng 50-60% nhu cầu thế giới và tiêu thụ nội địa"),
        ("Tốc độ:", "Hỗ trợ chọn GPU / CPU và xử lý hàng loạt nhiều ảnh cùng lúc"),
    ]

    y = 132
    for label, val in lines:
        draw.text((55, y), label, fill=(71, 85, 105), font=font_body)
        draw.text((220, y), val, fill=(15, 23, 42), font=font_body)
        y += 48

    draw.line([55, h - 65, w - 55, h - 65], fill=(226, 232, 240), width=2)
    draw.text((55, h - 55), "Hybrid Engine v3.0: Fast Batch Detection (GPU/CPU) + VietOCR (100% Vietnamese Diacritics)", fill=(100, 116, 139), font=font_small)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@app.get("/api/status")
def get_status():
    gpu_available = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    gpu_name = "CPU"
    if gpu_available:
        try:
            gpu_name = paddle.device.cuda.get_device_name(0)
        except Exception:
            gpu_name = "NVIDIA CUDA GPU"
    return {
        "status": "online",
        "paddle_version": paddle.__version__,
        "gpu_available": gpu_available,
        "device_name": gpu_name,
        "cpu_threads": torch.get_num_threads(),
    }


@app.post("/api/ocr")
async def run_ocr(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    device: str = Form("gpu"),
    lang: str = Form("vi_fast"),
    ocr_version: str = Form("VietOCR-Seq2Seq"),
    score_thresh: float = Form(0.5),
    render_annotated: bool = Form(True),
):
    start_time = time.perf_counter()
    temp_files: List[Tuple[str, str]] = []  # (temp_path, original_filename)

    try:
        upload_list: List[UploadFile] = []
        if files:
            upload_list.extend(files)
        if file and file not in upload_list:
            upload_list.append(file)

        if use_sample:
            img_bgr = create_sample_image()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                temp_path = tmp.name
            cv2.imwrite(temp_path, img_bgr)
            temp_files.append((temp_path, "bao_cao_sau_rieng_vi.png"))
        else:
            if not upload_list:
                raise HTTPException(status_code=400, detail="Vui lòng chọn ít nhất 1 tệp ảnh hoặc PDF.")
            for uf in upload_list:
                fname = uf.filename or "uploaded_file.png"
                ext = Path(fname).suffix.lower()
                if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".pdf"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Định dạng tệp '{fname}' ({ext}) không được hỗ trợ.",
                    )
                content = await uf.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(content)
                    temp_files.append((tmp.name, fname))

        use_vietocr = lang in ("vi_fast", "vi_pro") or ocr_version in ("VietOCR-Seq2Seq", "VietOCR-Transformer")
        viet_mode = "seq2seq" if (lang == "vi_fast" or ocr_version == "VietOCR-Seq2Seq") else "transformer"
        resolved_dev = resolve_paddle_device(device)

        pages_output = []
        all_files_text = []

        if use_vietocr:
            # FAST HYBRID PATH: Run ONLY TextDetection on all input files at once, then batch-recognize all crops!
            det_engine = get_det_engine(device_choice=device)
            viet_predictor = get_vietocr_engine(mode=viet_mode)

            input_paths = [tp for tp, _ in temp_files]
            path_to_name = {tp: fn for tp, fn in temp_files}

            det_results = list(det_engine.predict(input_paths))

            # Collect all crops across all images/pages for a single giant batch recognition call
            page_records = []
            all_crops: List[Image.Image] = []
            crop_map: List[Tuple[int, List[List[float]], float]] = []  # (page_rec_idx, poly_list, det_score)

            for p_idx, det_res in enumerate(det_results):
                in_path = det_res.get("input_path", input_paths[min(p_idx, len(input_paths) - 1)])
                orig_name = path_to_name.get(in_path, Path(in_path).name)
                img_bgr = det_res.get("input_img")
                if img_bgr is None and os.path.exists(in_path) and Path(in_path).suffix.lower() != ".pdf":
                    img_bgr = cv2.imread(in_path)

                dt_polys = det_res.get("dt_polys", [])
                dt_scores = det_res.get("dt_scores", [])

                rec_idx = len(page_records)
                page_records.append(
                    {
                        "filename": orig_name,
                        "page_index": det_res.get("page_index") or (p_idx + 1),
                        "img_bgr": img_bgr,
                        "raw_items": [],
                    }
                )

                if img_bgr is not None and len(dt_polys) > 0:
                    for poly, d_score in zip(dt_polys, dt_scores):
                        poly_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)
                        crop_pil = crop_polygon_image(img_bgr, poly_list)
                        all_crops.append(crop_pil)
                        crop_map.append((rec_idx, poly_list, float(d_score)))

            # Single batched VietOCR inference across ALL crops from ALL uploaded images!
            if all_crops:
                vn_texts, vn_probs = viet_predictor.predict_batch(all_crops, return_prob=True)
                for (rec_idx, poly_list, d_score), txt, prob in zip(crop_map, vn_texts, vn_probs):
                    score_val = float(prob) if prob is not None else d_score
                    if score_val < score_thresh:
                        continue
                    page_records[rec_idx]["raw_items"].append(
                        {
                            "id": len(page_records[rec_idx]["raw_items"]) + 1,
                            "text": str(txt),
                            "confidence": round(score_val, 4),
                            "poly": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in poly_list],
                        }
                    )

            for p_idx, prec in enumerate(page_records):
                page_items, page_formatted_text = reconstruct_reading_layout(prec["raw_items"])
                header_prefix = f"=== [{prec['filename']}] ===" if len(page_records) > 1 else ""
                if header_prefix:
                    all_files_text.append(f"{header_prefix}\n{page_formatted_text}")
                else:
                    all_files_text.append(page_formatted_text)

                img_bgr = prec["img_bgr"]
                h, w = img_bgr.shape[:2] if img_bgr is not None else (0, 0)
                annotated_b64 = (
                    draw_annotated_image(img_bgr, page_items)
                    if (render_annotated and img_bgr is not None)
                    else None
                )

                pages_output.append(
                    {
                        "page": p_idx + 1,
                        "filename": prec["filename"],
                        "width": w,
                        "height": h,
                        "line_count": len(page_items),
                        "text": page_formatted_text,
                        "items": page_items,
                        "annotated_image": annotated_b64,
                    }
                )
            engine_desc = f"PP-OCRv6 Det ({resolved_dev.upper()}) + VietOCR {viet_mode.capitalize()}"

        else:
            # Standard PaddleOCR pipeline for international languages (EN, CH, JP, KR...)
            ocr_engine = get_ocr_engine(lang=lang, ocr_version=ocr_version, device_choice=device)
            input_paths = [tp for tp, _ in temp_files]
            path_to_name = {tp: fn for tp, fn in temp_files}
            results = ocr_engine.predict(input_paths, text_rec_score_thresh=score_thresh)

            for p_idx, res in enumerate(results):
                in_path = res.get("input_path", input_paths[min(p_idx, len(input_paths) - 1)])
                orig_name = path_to_name.get(in_path, Path(in_path).name)
                rec_texts = res.get("rec_texts", [])
                rec_scores = res.get("rec_scores", [])
                rec_polys = res.get("rec_polys", [])

                raw_items = []
                for idx, (txt, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                    poly_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)
                    raw_items.append(
                        {
                            "id": idx + 1,
                            "text": str(txt),
                            "confidence": round(float(score), 4),
                            "poly": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in poly_list],
                        }
                    )

                page_items, page_formatted_text = reconstruct_reading_layout(raw_items)
                header_prefix = f"=== [{orig_name}] ===" if len(results) > 1 else ""
                if header_prefix:
                    all_files_text.append(f"{header_prefix}\n{page_formatted_text}")
                else:
                    all_files_text.append(page_formatted_text)

                doc_prep = res.get("doc_preprocessor_res")
                img_bgr = doc_prep["output_img"] if (doc_prep and "output_img" in doc_prep) else None
                h, w = img_bgr.shape[:2] if img_bgr is not None else (0, 0)
                annotated_b64 = (
                    draw_annotated_image(img_bgr, page_items)
                    if (render_annotated and img_bgr is not None)
                    else None
                )

                pages_output.append(
                    {
                        "page": p_idx + 1,
                        "filename": orig_name,
                        "width": w,
                        "height": h,
                        "line_count": len(page_items),
                        "text": page_formatted_text,
                        "items": page_items,
                        "annotated_image": annotated_b64,
                    }
                )
            engine_desc = f"PaddleOCR {ocr_version} ({resolved_dev.upper()})"

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        total_lines = sum(p["line_count"] for p in pages_output)
        avg_conf = (
            round(
                sum(item["confidence"] for p in pages_output for item in p["items"]) / total_lines,
                4,
            )
            if total_lines > 0
            else 0.0
        )

        return JSONResponse(
            {
                "success": True,
                "file_count": len(temp_files),
                "page_count": len(pages_output),
                "device_used": resolved_dev.upper(),
                "engine_used": engine_desc,
                "elapsed_ms": elapsed_ms,
                "ms_per_page": round(elapsed_ms / max(1, len(pages_output)), 1),
                "total_lines": total_lines,
                "avg_confidence": avg_conf,
                "full_text": "\n\n".join(all_files_text),
                "pages": pages_output,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý OCR: {str(e)}")
    finally:
        for tp, _ in temp_files:
            if tp and os.path.exists(tp):
                try:
                    os.remove(tp)
                except OSError:
                    pass


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("  PaddleOCR Web Studio v3.0 (GPU/CPU + Batch Multi-Image) tại: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
