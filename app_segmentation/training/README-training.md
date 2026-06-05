# Semantic Segmentation

Repository for the files of the semantic segmentation seminar, presented in the Deep Learning 2025.1 class.

## Steps for necessary libraries:
* Tested with ``conda``, Python 3.10 and CUDA 11.8
  
> ```bash
> conda create --name deepl python=3.10 --no-default-packages
> conda activate deepl
> ```

> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

## Hands-on
This notebook describes a practical application on semantic segmentation using a small subset of a dataset that contains pet images and their masks. This script is meant to assist in future implementations for semantic segmentation as a head start, including aspects of processing data, loading and configuring models, adjusting architecture for a new dataset, using new libraries to help in metric calculation, among other aspects.

---

### Task 1: Introduction and segmentation training

* **Dataset**: Oxford-IIIT Pet
* **Models**: UNet and FCN-ResNet50

### Task 2: Evaluation and Metrics

* Accuracy
* IoU
* Usage of torchmetrics

### Task 3: Visualization of results
* Predictions on test images
* Compare original image, ground truth mask and prediction
* Compare different models

### Android deployment

After running the notebook and generating `SimpleUNet_best.pth`, export the FP32 model used by the Android app:

```bash
pip install -r requirements-export.txt
python export_tflite.py
```

The default output is:

```text
../android_app/app/src/main/assets/model.tflite
```

The Java Android app expects the same preprocessing used in the notebook: RGB image resized to `128x128`, converted to tensor, and normalized with ImageNet mean/std. The exported model output can be either `[1, 3, H, W]` or `[1, H, W, 3]`; the app detects both layouts and applies `argmax` per pixel.
