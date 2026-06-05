package com.andre.tflite.segmentation;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class Segmenter implements AutoCloseable {
    private enum TensorLayout { NHWC, NCHW }
    private enum OutputLayout { NHWC, NCHW, LABEL_MAP }

    private final Interpreter interpreter;
    private NnApiDelegate nnapiDelegate;
    private GpuDelegate gpuDelegate;

    private final DataType inputType;
    private final float inputScale;
    private final int inputZeroPoint;
    private final TensorLayout inputLayout;
    private final int inputHeight;
    private final int inputWidth;

    private final DataType outputType;
    private final float outputScale;
    private final int outputZeroPoint;
    private final OutputLayout outputLayout;
    private final int outputHeight;
    private final int outputWidth;
    private final int numClasses;
    private final int outputElementCount;
    private final int outputByteCount;

    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std = {0.229f, 0.224f, 0.225f};
    private final int[] maskColors = {
            Color.rgb(244, 67, 54),
            Color.rgb(76, 175, 80),
            Color.rgb(33, 150, 243)
    };
    private final int[] overlayColors = {
            Color.argb(150, 244, 67, 54),
            Color.argb(0, 76, 175, 80),
            Color.argb(170, 33, 150, 243)
    };

    public Segmenter(AssetManager assetManager, String modelPath) throws IOException {
        this(assetManager, modelPath, 3);
    }

    public Segmenter(AssetManager assetManager, String modelPath, int fallbackNumClasses) throws IOException {
        ByteBuffer modelBuffer = loadModelFile(assetManager, modelPath);
        interpreter = buildInterpreterWithFallbacks(modelBuffer);

        int[] inputShape = interpreter.getInputTensor(0).shape();
        if (inputShape.length != 4) {
            throw new IllegalArgumentException("Entrada esperada com 4 dimensoes.");
        }

        inputType = interpreter.getInputTensor(0).dataType();
        inputScale = interpreter.getInputTensor(0).quantizationParams().getScale();
        inputZeroPoint = interpreter.getInputTensor(0).quantizationParams().getZeroPoint();

        if (inputShape[3] == 3) {
            inputLayout = TensorLayout.NHWC;
            inputHeight = inputShape[1];
            inputWidth = inputShape[2];
        } else if (inputShape[1] == 3) {
            inputLayout = TensorLayout.NCHW;
            inputHeight = inputShape[2];
            inputWidth = inputShape[3];
        } else {
            throw new IllegalArgumentException("Entrada RGB esperada em NHWC ou NCHW.");
        }

        int[] outputShape = interpreter.getOutputTensor(0).shape();
        outputType = interpreter.getOutputTensor(0).dataType();
        outputScale = interpreter.getOutputTensor(0).quantizationParams().getScale();
        outputZeroPoint = interpreter.getOutputTensor(0).quantizationParams().getZeroPoint();

        int elementCount = 1;
        for (int value : outputShape) {
            elementCount *= value;
        }
        outputElementCount = elementCount;
        outputByteCount = outputElementCount * bytesPerElement(outputType);

        if (outputShape.length == 4 && outputShape[3] >= 2 && outputShape[3] <= 64) {
            outputLayout = OutputLayout.NHWC;
            outputHeight = outputShape[1];
            outputWidth = outputShape[2];
            numClasses = outputShape[3];
        } else if (outputShape.length == 4 && outputShape[1] >= 2 && outputShape[1] <= 64) {
            outputLayout = OutputLayout.NCHW;
            outputHeight = outputShape[2];
            outputWidth = outputShape[3];
            numClasses = outputShape[1];
        } else if (outputShape.length == 3) {
            outputLayout = OutputLayout.LABEL_MAP;
            outputHeight = outputShape[1];
            outputWidth = outputShape[2];
            numClasses = fallbackNumClasses;
        } else {
            throw new IllegalArgumentException("Saida de segmentacao esperada como [1,H,W,C], [1,C,H,W] ou [1,H,W].");
        }
    }

    private Interpreter buildInterpreterWithFallbacks(ByteBuffer modelBuffer) {
        Interpreter nnapiInterpreter = tryNnApi(modelBuffer);
        if (nnapiInterpreter != null) {
            return nnapiInterpreter;
        }

        Interpreter gpuInterpreter = tryGpu(modelBuffer);
        if (gpuInterpreter != null) {
            return gpuInterpreter;
        }

        modelBuffer.rewind();
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(Math.min(Runtime.getRuntime().availableProcessors(), 4));
        return new Interpreter(modelBuffer, options);
    }

    private Interpreter tryNnApi(ByteBuffer modelBuffer) {
        NnApiDelegate delegate = null;
        try {
            delegate = new NnApiDelegate();
            modelBuffer.rewind();
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(Math.min(Runtime.getRuntime().availableProcessors(), 4));
            options.addDelegate(delegate);
            Interpreter candidate = new Interpreter(modelBuffer, options);
            nnapiDelegate = delegate;
            return candidate;
        } catch (Throwable ignored) {
            if (delegate != null) {
                delegate.close();
            }
            return null;
        }
    }

    private Interpreter tryGpu(ByteBuffer modelBuffer) {
        GpuDelegate delegate = null;
        try {
            delegate = new GpuDelegate();
            modelBuffer.rewind();
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(Math.min(Runtime.getRuntime().availableProcessors(), 4));
            options.addDelegate(delegate);
            Interpreter candidate = new Interpreter(modelBuffer, options);
            gpuDelegate = delegate;
            return candidate;
        } catch (Throwable ignored) {
            if (delegate != null) {
                delegate.close();
            }
            return null;
        }
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        try (InputStream inputStream = assetManager.open(modelPath);
             ByteArrayOutputStream outputStream = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[16 * 1024];
            int read;
            while ((read = inputStream.read(chunk)) != -1) {
                outputStream.write(chunk, 0, read);
            }

            byte[] modelBytes = outputStream.toByteArray();
            ByteBuffer buffer = ByteBuffer.allocateDirect(modelBytes.length).order(ByteOrder.nativeOrder());
            buffer.put(modelBytes);
            buffer.rewind();
            return buffer;
        }
    }

    public SegmentationResult segment(Bitmap bitmap) {
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true);
        ByteBuffer input = makeInputBuffer(scaled);
        ByteBuffer output = ByteBuffer.allocateDirect(outputByteCount).order(ByteOrder.nativeOrder());

        interpreter.run(input, output);

        float[] values = readOutput(output);
        int[] mask = decodeMask(values);
        Bitmap maskBitmap = createMaskBitmap(mask, false);
        Bitmap overlayMask = createMaskBitmap(mask, true);
        Bitmap overlayBitmap = overlayOnOriginal(bitmap, overlayMask);
        int[] counts = new int[numClasses];

        for (int classId : mask) {
            if (classId >= 0 && classId < counts.length) {
                counts[classId]++;
            }
        }

        return new SegmentationResult(maskBitmap, overlayBitmap, counts);
    }

    private ByteBuffer makeInputBuffer(Bitmap bitmap) {
        if (inputType == DataType.FLOAT32) {
            return bitmapToFloat32(bitmap);
        }
        if (inputType == DataType.UINT8) {
            return bitmapToQuantized(bitmap, false);
        }
        if (inputType == DataType.INT8) {
            return bitmapToQuantized(bitmap, true);
        }
        throw new IllegalArgumentException("Tipo de entrada nao suportado: " + inputType);
    }

    private ByteBuffer bitmapToFloat32(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * inputHeight * inputWidth * 3).order(ByteOrder.nativeOrder());
        int[] pixels = new int[inputHeight * inputWidth];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        if (inputLayout == TensorLayout.NHWC) {
            for (int pixel : pixels) {
                buffer.putFloat(normalizedChannel(pixel, 0));
                buffer.putFloat(normalizedChannel(pixel, 1));
                buffer.putFloat(normalizedChannel(pixel, 2));
            }
        } else {
            for (int channel = 0; channel < 3; channel++) {
                for (int pixel : pixels) {
                    buffer.putFloat(normalizedChannel(pixel, channel));
                }
            }
        }

        buffer.rewind();
        return buffer;
    }

    private ByteBuffer bitmapToQuantized(Bitmap bitmap, boolean signed) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * 3).order(ByteOrder.nativeOrder());
        int[] pixels = new int[inputHeight * inputWidth];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        if (inputLayout == TensorLayout.NHWC) {
            for (int pixel : pixels) {
                buffer.put(quantize(normalizedChannel(pixel, 0), signed));
                buffer.put(quantize(normalizedChannel(pixel, 1), signed));
                buffer.put(quantize(normalizedChannel(pixel, 2), signed));
            }
        } else {
            for (int channel = 0; channel < 3; channel++) {
                for (int pixel : pixels) {
                    buffer.put(quantize(normalizedChannel(pixel, channel), signed));
                }
            }
        }

        buffer.rewind();
        return buffer;
    }

    private byte quantize(float value, boolean signed) {
        int minQ = signed ? -128 : 0;
        int maxQ = signed ? 127 : 255;
        int quantized = Math.round(value / inputScale + inputZeroPoint);
        return (byte) Math.max(minQ, Math.min(maxQ, quantized));
    }

    private float normalizedChannel(int pixel, int channel) {
        int raw;
        if (channel == 0) {
            raw = (pixel >> 16) & 0xFF;
        } else if (channel == 1) {
            raw = (pixel >> 8) & 0xFF;
        } else {
            raw = pixel & 0xFF;
        }
        float value = raw / 255f;
        return (value - mean[channel]) / std[channel];
    }

    private float[] readOutput(ByteBuffer output) {
        output.rewind();

        if (outputType == DataType.FLOAT32) {
            float[] values = new float[outputElementCount];
            output.asFloatBuffer().get(values);
            return values;
        }

        byte[] bytes = new byte[outputElementCount];
        output.get(bytes);
        float[] values = new float[outputElementCount];

        if (outputType == DataType.UINT8) {
            for (int index = 0; index < outputElementCount; index++) {
                values[index] = (((int) bytes[index] & 0xFF) - outputZeroPoint) * outputScale;
            }
            return values;
        }

        if (outputType == DataType.INT8) {
            for (int index = 0; index < outputElementCount; index++) {
                values[index] = (bytes[index] - outputZeroPoint) * outputScale;
            }
            return values;
        }

        throw new IllegalArgumentException("Tipo de saida nao suportado: " + outputType);
    }

    private int[] decodeMask(float[] values) {
        int[] mask = new int[outputHeight * outputWidth];

        if (outputLayout == OutputLayout.NHWC) {
            for (int y = 0; y < outputHeight; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    int bestClass = 0;
                    float bestScore = Float.NEGATIVE_INFINITY;
                    for (int classId = 0; classId < numClasses; classId++) {
                        float score = values[((y * outputWidth + x) * numClasses) + classId];
                        if (score > bestScore) {
                            bestScore = score;
                            bestClass = classId;
                        }
                    }
                    mask[y * outputWidth + x] = bestClass;
                }
            }
        } else if (outputLayout == OutputLayout.NCHW) {
            for (int y = 0; y < outputHeight; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    int bestClass = 0;
                    float bestScore = Float.NEGATIVE_INFINITY;
                    for (int classId = 0; classId < numClasses; classId++) {
                        float score = values[(classId * outputHeight * outputWidth) + (y * outputWidth) + x];
                        if (score > bestScore) {
                            bestScore = score;
                            bestClass = classId;
                        }
                    }
                    mask[y * outputWidth + x] = bestClass;
                }
            }
        } else {
            for (int index = 0; index < mask.length; index++) {
                int classId = Math.round(values[index]);
                mask[index] = Math.max(0, Math.min(numClasses - 1, classId));
            }
        }

        return mask;
    }

    private Bitmap createMaskBitmap(int[] mask, boolean useOverlayAlpha) {
        int[] pixels = new int[outputHeight * outputWidth];
        for (int index = 0; index < pixels.length; index++) {
            pixels[index] = colorForClass(mask[index], useOverlayAlpha);
        }
        return Bitmap.createBitmap(pixels, outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
    }

    private int colorForClass(int classId, boolean useOverlayAlpha) {
        int[] palette = useOverlayAlpha ? overlayColors : maskColors;
        if (classId >= 0 && classId < palette.length) {
            return palette[classId];
        }

        int hue = (classId * 47) % 255;
        if (useOverlayAlpha) {
            return Color.argb(150, hue, 255 - hue, (hue * 3) % 255);
        }
        return Color.rgb(hue, 255 - hue, (hue * 3) % 255);
    }

    private Bitmap overlayOnOriginal(Bitmap original, Bitmap overlayMask) {
        Bitmap output = Bitmap.createBitmap(original.getWidth(), original.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        canvas.drawBitmap(original, 0f, 0f, null);

        Bitmap scaledMask = Bitmap.createScaledBitmap(overlayMask, original.getWidth(), original.getHeight(), false);
        Paint paint = new Paint();
        paint.setFilterBitmap(false);
        canvas.drawBitmap(scaledMask, 0f, 0f, paint);
        return output;
    }

    private int bytesPerElement(DataType type) {
        if (type == DataType.FLOAT32) {
            return 4;
        }
        if (type == DataType.UINT8 || type == DataType.INT8) {
            return 1;
        }
        throw new IllegalArgumentException("Tipo de tensor nao suportado: " + type);
    }

    @Override
    public void close() {
        interpreter.close();
        if (nnapiDelegate != null) {
            nnapiDelegate.close();
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
        }
    }
}
