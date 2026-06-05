# Android Java App

Java Android app for the semantic segmentation task.

Open this folder in Android Studio:

```text
android_app
```

Before running the app, replace `app/src/main/assets/model.tflite` with the FP32 model exported by `../training/export_tflite.py`.

Build prerequisite: use JDK 17 or JDK 21 in Android Studio/Gradle. JDK 26 may fail with the Gradle version configured in this project.

App flow:

1. Take a photo or load an image.
2. Tap the segmentation button.
3. View the original image in the `Original` section.
4. View the prediction section, with the colored mask overlaid on the image.

Main files:

- `app/src/main/java/com/andre/tflite/segmentation/MainActivity.java`
- `app/src/main/java/com/andre/tflite/segmentation/Segmenter.java`
- `app/src/main/java/com/andre/tflite/segmentation/SegmentationResult.java`
- `app/src/main/res/layout/activity_main.xml`

Detailed testing guide:

```text
../docs/mobile_app_test.md
```
