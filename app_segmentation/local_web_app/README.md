# Local Web App

Local browser-based version for testing segmentation without Android Studio, ADB, a phone, or an emulator.

This application does not replace the Android APK. It uses the trained model (`training/SimpleUNet_best.pth`) and lets you visually validate:

- image upload;
- mask prediction;
- mask overlay on the original image;
- class percentages for `Pet`, `Background`, and `Border`.

## How to Run

Use the same Python environment used to run the notebook.

Install the dependencies:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r training\requirements.txt
pip install -r local_web_app\requirements.txt
```

Confirm that this file exists:

```text
training/SimpleUNet_best.pth
```

Start the server:

```powershell
cd app_segmentation
python local_web_app\server.py
```

Open in the browser:

```text
http://localhost:8000
```

Then:

1. Click the file picker.
2. Select a `.jpg`, `.jpeg`, `.png`, or `.webp` image.
3. Click the segmentation button.
4. Check the original image, the overlay image, and the mask.

## Dependencies

The environment needs:

```text
torch
torchvision
pillow
```

If needed, install:

```bash
pip install -r local_web_app/requirements.txt
```

## Weight File

By default, the server looks for:

```text
training/SimpleUNet_best.pth
```

To use another file:

```powershell
python local_web_app\server.py --weights path\to\SimpleUNet_best.pth
```
