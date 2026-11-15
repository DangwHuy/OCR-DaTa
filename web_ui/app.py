import atexit
import base64
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Isolate PaddlePaddle process from PyTorch cuDNN DLLs (VietOCR runs on GPU in vietocr_worker.py)
sys.modules["torch"] = None

import cv2
import numpy as np
import paddle
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

from paddleocr import PaddleOCR, TextDetection

app = FastAPI(
    title="PaddleOCR Web Studio",
    description="Giao diện Web nhận diện văn bản (OCR) Tiếng Việt & Đa ngôn ngữ siêu tốc trên GPU/CPU",
    version="4.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
WORKER_SCRIPT = BASE_DIR / "vietocr_worker.py"

_ocr_lock = threading.Lock()
_det_instances: Dict[Tuple[str, str], TextDetection] = {}
_ocr_instances: Dict[tuple, PaddleOCR] = {}


class VietOCRWorkerClient:
    """Persistent subprocess client running PyTorch CUDA VietOCR without cuDNN DLL conflict."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            self._proc = subprocess.Popen(
                [sys.executable, str(WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )
        return self._proc

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            proc = self._ensure_started()
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            resp_line = proc.stdout.readline()
            if not resp_line:
                raise RuntimeError("VietOCR worker process terminated unexpectedly.")
            data = json.loads(resp_line)
            if not data.get("ok"):
                raise RuntimeError(data.get("error", "VietOCR worker error"))
            return data

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                self._proc = None


_viet_worker = VietOCRWorkerClient()
atexit.register(_viet_worker.close)


def resolve_paddle_device(device_choice: str) -> str:
    gpu_ok = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    if device_choice == "cpu":
        return "cpu"
    return "gpu:0" if gpu_ok else "cpu"


def get_det_engine(model_name: str = "PP-OCRv5_mobile_det", device_choice: str = "gpu") -> TextDetection:
    p_dev = resolve_paddle_device(device_choice)
    key = (model_name, p_dev)
    with _ocr_lock:
        if key not in _det_instances:
            print(f"[PaddleOCR Web] Đang khởi tạo {model_name} trên {p_dev.upper()}...")
            kwargs: Dict[str, Any] = {
                "model_name": model_name,
                "device": p_dev,
                "unclip_ratio": 1.7,
            }
            if p_dev == "cpu":
                kwargs["cpu_threads"] = os.cpu_count() or 8
                kwargs["enable_mkldnn"] = True
            _det_instances[key] = TextDetection(**kwargs)
        return _det_instances[key]


def get_ocr_engine(lang: str = "en", ocr_version: str = "PP-OCRv6", device_choice: str = "gpu") -> PaddleOCR:
    p_dev = resolve_paddle_device(device_choice)
    actual_lang = "vi" if lang in ("vi_fast", "vi_pro") else lang
    actual_ver = "PP-OCRv6" if ocr_version in ("VietOCR-Seq2Seq", "VietOCR-Transformer") else ocr_version
    key = (actual_lang, actual_ver, p_dev)
    with _ocr_lock:
        if key not in _ocr_instances:
            print(f"[PaddleOCR Web] Đang khởi tạo PaddleOCR ({actual_lang}, {actual_ver}) trên {p_dev.upper()}...")
            kwargs: Dict[str, Any] = {
                "lang": actual_lang,
                "ocr_version": actual_ver,
                "device": p_dev,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            if p_dev == "cpu":
                kwargs["cpu_threads"] = os.cpu_count() or 8
                kwargs["enable_mkldnn"] = True
            _ocr_instances[key] = PaddleOCR(**kwargs)
        return _ocr_instances[key]


def crop_polygon_b64(img_bgr: np.ndarray, poly: List[List[float]]) -> str:
    """Perspective-crop a 4-point text box into a padded BMP base64 string (zero-compression overhead)."""
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

        pad_y = max(4, int(crop_h * 0.18))
        pad_x = max(3, int(crop_w * 0.03))
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
    else:
        x_min = max(0, int(np.min(pts[:, 0])) - 4)
        y_min = max(0, int(np.min(pts[:, 1])) - 5)
        x_max = min(w_img, int(np.max(pts[:, 0])) + 4)
        y_max = min(h_img, int(np.max(pts[:, 1])) + 5)
        warped = img_bgr[y_min:y_max, x_min:x_max]
        if warped.size == 0:
            warped = np.full((32, 64, 3), 255, dtype=np.uint8)

    # Ensure small text lines are upscaled cleanly to at least 32px height for crisp diacritics
    wh, ww = warped.shape[:2]
    if wh > 0 and wh < 32:
        scale = 32.0 / wh
        warped = cv2.resize(warped, (max(8, int(round(ww * scale))), 32), interpolation=cv2.INTER_CUBIC)
        wh, ww = warped.shape[:2]

    # Quantize width to 64px buckets using BORDER_REPLICATE (no harsh white seam on scanned/gray backgrounds)
    if wh > 0:
        scaled_w = max(32, int(round(32.0 * ww / wh)))
        bucket_w = int(math.ceil(scaled_w / 64.0) * 64)
        target_orig_w = int(round(bucket_w * wh / 32.0))
        if target_orig_w > ww:
            warped = cv2.copyMakeBorder(
                warped, 0, 0, 0, target_orig_w - ww, borderType=cv2.BORDER_REPLICATE
            )

    pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    buf = io.BytesIO()
    pil_img.save(buf, format="BMP")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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

            ordered_items.append(
                {
                    "id": new_id,
                    "text": b["text"],
                    "confidence": b["confidence"],
                    "poly": b["poly"],
                }
            )
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
        (59, 130, 246),
        (16, 185, 129),
        (245, 158, 11),
        (236, 72, 153),
        (139, 92, 246),
        (6, 182, 212),
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
    combined.save(buf, format="JPEG", quality=88)
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
        ("Tốc độ:", "Chạy 100% trên GPU RTX 3050 Ti siêu tốc (dưới 0.5 giây/trang)"),
    ]

    y = 132
    for label, val in lines:
        draw.text((55, y), label, fill=(71, 85, 105), font=font_body)
        draw.text((220, y), val, fill=(15, 23, 42), font=font_body)
        y += 48

    draw.line([55, h - 65, w - 55, h - 65], fill=(226, 232, 240), width=2)
    draw.text((55, h - 55), "Hybrid Engine v4.0: Full GPU Pipeline (PP-OCRv5/v6 Det + VietOCR CUDA)", fill=(100, 116, 139), font=font_small)

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
        "cpu_threads": max(4, (os.cpu_count() or 8) - 2),
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
    temp_files: List[Tuple[str, str]] = []

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
        det_model = "PP-OCRv5_mobile_det" if viet_mode == "seq2seq" else "PP-OCRv6_medium_det"
        resolved_dev = resolve_paddle_device(device)

        pages_output = []
        all_files_text = []

        if use_vietocr:
            det_engine = get_det_engine(model_name=det_model, device_choice=device)
            input_paths = [tp for tp, _ in temp_files]
            path_to_name = {tp: fn for tp, fn in temp_files}

            det_results = list(det_engine.predict(input_paths))

            page_records = []
            all_crops_b64: List[str] = []
            crop_map: List[Tuple[int, List[List[float]], float]] = []

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
                        all_crops_b64.append(crop_polygon_b64(img_bgr, poly_list))
                        crop_map.append((rec_idx, poly_list, float(d_score)))

            if all_crops_b64:
                worker_resp = _viet_worker.request(
                    {
                        "cmd": "predict",
                        "mode": viet_mode,
                        "device": device,
                        "crops": all_crops_b64,
                    }
                )
                vn_texts = worker_resp["texts"]
                vn_probs = worker_resp["probs"]

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
            engine_desc = f"{det_model} ({resolved_dev.upper()}) + VietOCR {viet_mode.capitalize()} ({resolved_dev.upper()})"

        else:
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
    print("  PaddleOCR Web Studio v4.0 (Full GPU Pipeline) tại: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
