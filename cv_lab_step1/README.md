# Computer Vision Lab — Step 1

This step covers:
- Reading a color image (BGR)
- Converting to grayscale in two ways:
  - Mean of R,G,B
  - HSV model: V (value) channel
 - Binarization: fixed threshold, Otsu
 - Histogram ops: normalize, equalize, contrast stretch
 - Convolutions: Gaussian blur, Laplacian sharpen, Sobel edges
 - Geometry: cyclic (wrap) shift, rotation about center
  (Optional) Show histogram via Parameters panel button

Desktop UI: PySide6 (Qt for Python).

> Comments inside the code are in English by request.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python -m app.main
```

### Windows: подробные инструкции (плоская структура)

- Git Bash (рекомендуется, если используешь Git Bash):
```bash
cd /d/CompVision/cv_lab_step1
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.main
```

- PowerShell:
```powershell
cd D:\CompVision\cv_lab_step1
python -m venv .venv
 .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.main
```

Проверка окружения:
```bash
python -c "import sys; print(sys.executable)"
pip show PySide6
```

Then open any `.jpg/.png/.bmp` image via **File → Open Image…**.
You will see tabs:
- Original (BGR→RGB)
- Gray — mean(R,G,B)
- Gray — HSV (V channel)
- Binarize — Fixed thr=128
- Binarize — Otsu
 - Normalize
 - Equalize
 - Contrast stretch
 - Blur — Gaussian (ksize=5)
 - Sharpen — Laplacian
 - Edges — Sobel
 - Geometry — Cyclic shift
 - Geometry — Rotate center +15°

## Parameters and saving

- Right dock panel contains parameters:
  - Threshold (with invert)
  - Gaussian: kernel size (odd), sigma
  - Laplacian sharpen: alpha, kernel size (odd)
  - Sobel: kernel size (odd)
  - Shift: dx, dy
  - Rotate: angle (deg)
- Changes apply immediately.
- Use toolbar "Save" to export current tab result.
 - File → Export Results… — сохраняет все вкладки в выбранную папку + params.json.

## Histogram

- Use "Show Histogram" button in the Parameters panel to open a dialog.
- The dialog plots the gray distribution (current tab) and its equalized version.

## Export

- File → Export Results… — сохраняет изображения всех вкладок в выбранную папку и `params.json` с текущими параметрами.
