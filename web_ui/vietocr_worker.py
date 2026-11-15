import base64
import io
import json
import os
import sys
import warnings

# Redirect all warnings/logs to stderr so stdout stays strictly JSON-RPC
warnings.filterwarnings("ignore")
os.environ["PYTHONUTF8"] = "1"

import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor as VietOCRPredictor

torch.set_num_threads(os.cpu_count() or 8)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_predictors = {}


def get_predictor(mode: str, device_choice: str) -> VietOCRPredictor:
    dev = "cuda:0" if (device_choice != "cpu" and torch.cuda.is_available()) else "cpu"
    key = (mode, dev)
    if key not in _predictors:
        sys.stderr.write(f"[VietOCR Worker] Loading {mode} on {dev}...\n")
        sys.stderr.flush()
        with open(os.path.join(BASE_DIR, "base.yml"), encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        if mode == "seq2seq":
            with open(os.path.join(BASE_DIR, "vgg_seq2seq.yml"), encoding="utf-8") as f:
                cfg_dict.update(yaml.safe_load(f))
            weights_path = hf_hub_download(
                repo_id="lenhanm1/enterprise-printed-ocr-vietocr-tesseract",
                filename="vietocr/vgg-seq2seq-enterprise-print.pth",
            )
        else:
            with open(os.path.join(BASE_DIR, "vgg_transformer.yml"), encoding="utf-8") as f:
                cfg_dict.update(yaml.safe_load(f))
            weights_path = hf_hub_download(
                repo_id="doanhm/vgg_transformer",
                filename="vgg_transformer.pth",
            )

        cfg = Cfg(cfg_dict)
        cfg["cnn"]["pretrained"] = False
        cfg["device"] = dev
        cfg["predictor"]["beamsearch"] = False
        cfg["weights"] = weights_path
        cfg["dataset"]["image_min_width"] = 32
        cfg["dataset"]["image_max_width"] = 768

        predictor = VietOCRPredictor(cfg)
        predictor.model.eval()
        if dev == "cpu":
            # Dynamic INT8 quantization speeds up Linear/LSTM/GRU layers on CPU by ~25-35%
            predictor.model = torch.ao.quantization.quantize_dynamic(
                predictor.model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8,
            )
        _predictors[key] = predictor
        sys.stderr.write(f"[VietOCR Worker] Ready: {mode} on {dev}!\n")
        sys.stderr.flush()
    return _predictors[key]


def main():
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
            if req.get("cmd") == "ping":
                resp = {
                    "ok": True,
                    "cuda": torch.cuda.is_available(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                }
            elif req.get("cmd") == "predict":
                mode = req.get("mode", "seq2seq")
                device_choice = req.get("device", "gpu")
                crops_b64 = req.get("crops", [])

                predictor = get_predictor(mode, device_choice)
                pil_imgs = []
                for b64 in crops_b64:
                    img_bytes = base64.b64decode(b64)
                    pil_imgs.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

                if pil_imgs:
                    with torch.inference_mode():
                        texts, probs = predictor.predict_batch(pil_imgs, return_prob=True)
                    probs_list = [float(p) if p is not None else 0.9 for p in probs]
                else:
                    texts, probs_list = [], []

                resp = {"ok": True, "texts": texts, "probs": probs_list}
            else:
                resp = {"ok": False, "error": f"Unknown command: {req.get('cmd')}"}
        except Exception as e:
            resp = {"ok": False, "error": str(e)}

        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
