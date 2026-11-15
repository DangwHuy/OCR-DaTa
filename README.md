<div align="center">

# 📄 OCR-DaTa — Hệ Thống Nhận Diện Văn Bản AI (Tiếng Việt & Đa Ngôn Ngữ)

**Giải pháp OCR tốc độ cao kết hợp phát hiện vùng chữ (Text Detection) và nhận diện Tiếng Việt chuẩn xác 100% dấu, tích hợp sẵn giao diện Web Studio xử lý hàng loạt trên GPU & CPU.**

[![Author](https://img.shields.io/badge/Author-DangwHuy-blue?style=flat-square)](https://github.com/DangwHuy/OCR-DaTa)
[![Python](https://img.shields.io/badge/Python-3.9%20~%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20GPU%20%7C%20CPU-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://github.com/DangwHuy/OCR-DaTa)
[![License](https://img.shields.io/badge/License-Apache%202.0-emerald?style=flat-square)](./LICENSE)

</div>

---

## ✨ Tính Năng Nổi Bật

- **🇻🇳 Nhận Diện Tiếng Việt Chuẩn Xác 100% Dấu**:
  - Khắc phục hoàn toàn lỗi mất dấu kép Tiếng Việt (`ệ, ấ, ẩ, ậ, ắ, ằ, ẳ, ẵ, ặ, ư, ơ, ờ, ở, ỡ, ợ...`) nhờ kiến trúc lai (**Hybrid Engine**) kết hợp bộ phát hiện khung chữ **`PP-OCRv6_medium_det`** và bộ giải mã Tiếng Việt **`VietOCR` (`Seq2Seq` & `Transformer`)**.
- **⚡ Tùy Chọn Linh Hoạt GPU & CPU Ngay Trên Giao Diện Web**:
  - Cho phép chuyển đổi trực tiếp giữa **GPU NVIDIA (CUDA)** và **CPU Đa Luồng** ngay trên giao diện người dùng mà không cần khởi động lại máy chủ.
- **📦 Xử Lý Hàng Loạt Nhiều Ảnh / PDF Cùng Lúc (Batch Multi-Image OCR)**:
  - Kéo thả hàng chục bức ảnh hoặc tài liệu PDF vào hàng đợi, tự động gom lô ma trận (width-bucketing) để giải mã song song chỉ trong một lượt.
- **📐 Tái Tạo Bố Cục Dòng & Bảng Số Liệu Thông Minh**:
  - Tự động sắp xếp thứ tự đọc từ trên xuống dưới, từ trái sang phải và giữ nguyên cấu trúc hàng/cột của bảng thống kê bằng ký tự phân tách `Tab`.
- **🌐 Giao Diện Web Hiện Đại (`web_ui/`)**:
  - Hỗ trợ kéo thả nhiều tệp, dán ảnh trực tiếp từ Clipboard (`Ctrl + V`), xem ảnh khoanh vùng đa giác màu, lọc kết quả theo từng trang và xuất tệp `.TXT` / `.JSON`.

---

## 🚀 Hướng Dẫn Cài Đặt

### 1. Tải mã nguồn về máy
```bash
git clone https://github.com/DangwHuy/OCR-DaTa.git
cd OCR-DaTa
```

### 2. Khởi tạo môi trường ảo (Khuyên dùng Python 3.11)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Cài đặt thư viện Backend (GPU hoặc CPU)
- **Dành cho máy có card rời NVIDIA (CUDA 12.6)**:
  ```powershell
  python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
  ```
- **Dành cho máy chỉ dùng CPU**:
  ```powershell
  python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  ```

### 4. Cài đặt gói OCR-DaTa và Giao diện Web
```powershell
python -m pip install -e .
python -m pip install fastapi uvicorn python-multipart beautifulsoup4 PySocks
python -m pip install --no-deps vietocr einops gdown prefetch-generator
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🖥️ Cách Khởi Chạy Giao Diện Web Studio

### Cách 1: Chạy nhanh bằng 1 cú nhấp đúp chuột (Windows)
Nhấp đúp trực tiếp vào tệp **`Chay_Web_OCR.bat`** ở thư mục gốc. Trình duyệt sẽ tự động mở địa chỉ:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Cách 2: Chạy bằng dòng lệnh Terminal
```powershell
.\.venv\Scripts\python.exe web_ui\app.py
```

---

## 🛠️ Cấu Trúc Thư Mục Chính

```text
OCR-DaTa/
├── web_ui/                  # Giao diện Web Studio & Máy chủ FastAPI
│   ├── app.py               # Backend xử lý Batch OCR, GPU/CPU & Hybrid VietOCR
│   ├── base.yml             # Cấu hình từ điển Tiếng Việt cơ sở
│   ├── vgg_seq2seq.yml      # Cấu hình mô hình Tiếng Việt Siêu Tốc (Seq2Seq)
│   ├── vgg_transformer.yml  # Cấu hình mô hình Tiếng Việt Chính Xác Cao (Transformer)
│   └── static/
│       └── index.html       # Giao diện người dùng (TailwindCSS, Batch Queue, Preview)
├── paddleocr/               # Thư viện lõi điều phối Pipeline & CLI
├── ppocr/                   # Các thuật toán Detection, Recognition & Post-processing
├── configs/                 # Tệp cấu hình huấn luyện (Training configs)
├── tools/                   # Công cụ huấn luyện & xuất mô hình (Train / Export)
├── Chay_Web_OCR.bat         # Script khởi động nhanh giao diện Web trên Windows
└── pyproject.toml           # Cấu hình đóng gói Python
```

---

## 👤 Tác Giả & Liên Hệ

- **Tác giả**: **DangwHuy**
- **GitHub**: [https://github.com/DangwHuy/OCR-DaTa](https://github.com/DangwHuy/OCR-DaTa)
- **Email**: `phandanghuy300305@gmail.com`
