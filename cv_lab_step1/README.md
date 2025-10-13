# Computer Vision Lab — Steps 1 & 2

## Step 1: Basic Image Processing
- Reading a color image (BGR)
- Converting to grayscale in two ways:
  - Mean of R,G,B
  - HSV model: V (value) channel
- Binarization: fixed threshold, Otsu
- Histogram ops: normalize, equalize, contrast stretch
- Convolutions: Gaussian blur, Laplacian sharpen, Sobel edges
- Geometry: cyclic (wrap) shift, rotation about center

## Step 2: Feature Detection
- **Hough Line Detection**: Detect straight lines using Hough transform
- **Hough Circle Detection**: Detect circles using Hough transform
- **Local Statistical Features**: Compute mean, variance, skewness, kurtosis in local windows
- **Texture Segmentation**: Region growing segmentation based on texture similarity with seed point

Desktop UI: PySide6 (Qt for Python) with interactive parameters and real-time updates.

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

### Step 1 Tabs:
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

### Step 2 Tabs:
- **Hough Lines**: Detect straight lines (green lines overlaid on image)
- **Hough Circles**: Detect circles (green circles with red centers)
- **Local Statistics — Mean**: Local mean intensity map
- **Local Statistics — Variance**: Local variance map (texture measure)
- **Local Statistics — Skewness**: Local skewness map (distribution asymmetry)
- **Local Statistics — Kurtosis**: Local kurtosis map (distribution peakedness)
- **Texture Segmentation**: Colored regions based on texture similarity

## Parameters and saving

- Right dock panel contains parameters (context-sensitive):
  - **Step 1 parameters**:
    - Threshold (with invert)
    - Gaussian: kernel size (odd), sigma
    - Laplacian sharpen: alpha, kernel size (odd)
    - Sobel: kernel size (odd)
    - Shift: dx, dy
    - Rotate: angle (deg)
  - **Step 2 parameters**:
    - **Hough Lines**: threshold, min line length, max line gap
    - **Hough Circles**: dp, min distance, param1, param2
    - **Local Statistics**: window size
    - **Texture Segmentation**: threshold, window size, seed point (x, y)
- Changes apply immediately.
- Use toolbar "Save" to export current tab result.
- File → Export Results… — сохраняет все вкладки в выбранную папку + params.json.

## Histogram

- Use "Show Histogram" button in the Parameters panel to open a dialog.
- The dialog plots the gray distribution (current tab) and its equalized version.

## Testing Instructions

### How to test Step 2 features:

1. **Hough Lines**:
   - Open an image with clear straight lines (e.g., building, road, geometric shapes)
   - Go to "Hough Lines" tab
   - Adjust parameters: threshold (10-500), min line length (10-200), max line gap (5-50)
   - Lower threshold = more lines detected, higher = fewer but stronger lines

2. **Hough Circles**:
   - Open an image with circular objects (coins, balls, wheels)
   - Go to "Hough Circles" tab
   - Adjust parameters: dp (0.5-3.0), min distance (10-200), param1/param2 (10-200)
   - Higher param2 = fewer circles, lower = more circles

3. **Local Statistics**:
   - Go to any "Local Statistics" tab (Mean, Variance, Skewness, Kurtosis)
   - Adjust window size (5-51, odd numbers)
   - Larger windows = smoother results, smaller = more detail
   - Variance shows texture patterns, Skewness shows asymmetry, Kurtosis shows peakedness

4. **Texture Segmentation**:
   - Go to "Texture Segmentation" tab
   - Set seed point (x, y) by clicking on a region you want to segment
   - Adjust threshold (5-100) and window size (5-51)
   - Lower threshold = more similar regions, higher = more strict similarity
   - Different regions will be colored differently

### What to expect:
- **Lines**: Green lines overlaid on original image
- **Circles**: Green circles with red centers
- **Statistics**: Grayscale maps showing local feature values
- **Segmentation**: Color-coded regions based on texture similarity

## Export

- File → Export Results… — сохраняет изображения всех вкладок в выбранную папку и `params.json` с текущими параметрами.
