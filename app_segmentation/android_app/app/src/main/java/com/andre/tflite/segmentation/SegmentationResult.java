package com.andre.tflite.segmentation;

import android.graphics.Bitmap;

public class SegmentationResult {
    public final Bitmap maskBitmap;
    public final Bitmap overlayBitmap;
    public final int[] classPixelCounts;

    public SegmentationResult(Bitmap maskBitmap, Bitmap overlayBitmap, int[] classPixelCounts) {
        this.maskBitmap = maskBitmap;
        this.overlayBitmap = overlayBitmap;
        this.classPixelCounts = classPixelCounts;
    }
}
