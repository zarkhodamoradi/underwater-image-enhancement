
# Underwater Image Enhancement

An end-to-end Classical Computer Vision pipeline built with **Python**, **OpenCV**, and **NumPy** to restore, de-haze, and enhance degraded underwater imagery.

Due to selective wavelength absorption (red light attenuates first, followed by green and blue) and particulate scattering ("marine snow"), raw underwater photos typically suffer from extreme blue/green color casts, low contrast, non-uniform illumination, and edge softness. This pipeline systematically addresses each distortion mode in sequential stages.

---

## Dataset

This project utilizes the **UIEB (Underwater Image Enhancement Benchmark)** dataset (`raw-890` subset):

* 🔗 **Official Benchmark Page:** [UIEB Dataset Project Page](https://li-chongyi.github.io/proj_benchmark.html)
* 🔗 **Kaggle Mirror (Direct Download):** [UIEB Dataset Refrence](https://www.kaggle.com/datasets/larjeck/uieb-dataset-reference) / [UIEB Raw 890 on Kaggle](https://www.kaggle.com/datasets/larjeck/uieb-dataset-raw)

To use the dataset with this notebook, download and extract the raw images into the `raw-890/` directory.

---

## Pipeline Architecture

```text
                      Raw Input Image
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 1. White Balancing (Gray-World Algorithm)     │ ──► Neutralizes dominant blue/green cast
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 2. Local Contrast Enhancement (CLAHE in LAB) │ ──► Equalizes L-channel without hue shift
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 3. Non-linear Gamma Correction               │ ──► Rebalances shadow and highlight dynamic range
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 4. Non-Local Means Denoising (NLM)           │ ──► Suppresses particulate and sensor noise
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │ 5. Spatial Kernel Sharpening                 │ ──► Restores fine textures and edge gradients
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
                    Enhanced Image
```

---

## Key Methodologies & Formulations

### 1. Gray-World White Balance

Assumes that the spatial average of scene reflectance across all color channels is achromatic (neutral gray). Channel scaling is computed as:

$$\mu_{\text{gray}} = \frac{\mu_R + \mu_G + \mu_B}{3}, \quad I'_c(x, y) = \text{clip}\left( I_c(x, y) \times \frac{\mu_{\text{gray}}}{\mu_c}, 0, 255 \right)$$

This restores suppressed red tones and eliminates dominant cyan/green tinting.

### 2. CLAHE on CIE L\*a\*b\*

To prevent chromatic distortion while expanding dynamic range:

- Converts the image from RGB to perceptual L<sup>\*</sup>a<sup>\*</sup>b<sup>\*</sup> space.
- Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) exclusively on the **L<sup>\*</sup> (Luminance)** channel with a clip limit to prevent over-amplifying noise in homogeneous regions.
- Keeps chromatic channels (a<sup>\*</sup>, b<sup>\*</sup>) unchanged.

### 3. Gamma Correction (Power-Law Transform)

Adjusts illumination response non-linearly:

$$O = 255 \times \left( \frac{I}{255} \right)^\gamma$$

Recovers details from underexposed shadows while maintaining highlight control.

### 4. Non-Local Means Denoising

Underwater particulates ("marine snow") and sensor noise are removed using OpenCV's `fastNlMeansDenoisingColored`, performing patch-based similarity checks to suppress noise without blurring structural edges.

### 5. High-Pass Spatial Sharpening

Compensates for turbidity and forward scattering blur via 2D spatial convolution using a Laplacian high-pass sharpening kernel:

$$
K = \begin{bmatrix} 
0 & -1 & 0 \\ 
-1 & 5 & -1 \\ 
0 & -1 & 0 
\end{bmatrix}
$$

---


## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/zarkhodamoradi/underwater-image-enhancement.git
cd underwater-image-enhancement
```

### 2. Install dependencies

```bash
pip install opencv-python numpy matplotlib jupyter
```

---

## Usage

1. Download the dataset and place the images into `raw-890/` (or update the file path in the notebook).
2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook Underwater-image-enhancement.ipynb
   ```

3. Run the notebook cells sequentially to execute the pipeline and view side-by-side visual comparisons.

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.

