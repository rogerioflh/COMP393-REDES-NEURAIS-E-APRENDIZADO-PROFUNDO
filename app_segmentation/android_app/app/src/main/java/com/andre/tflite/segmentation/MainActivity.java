package com.andre.tflite.segmentation;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Toast;

import com.andre.tflite.segmentation.databinding.ActivityMainBinding;

import java.io.IOException;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final int REQUEST_GALLERY = 10;
    private static final int REQUEST_CAMERA = 11;

    private ActivityMainBinding binding;
    private Bitmap selectedBitmap;
    private Segmenter segmenter;
    private final String modelFile = "model.tflite";
    private final String[] classes = {"Pet", "Fundo", "Borda"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openCamera();
            }
        });

        binding.btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openGallery();
            }
        });

        binding.btnPredict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                runSegmentation();
            }
        });
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, REQUEST_CAMERA);
        } else {
            Toast.makeText(this, "Nenhum app de camera encontrado.", Toast.LENGTH_SHORT).show();
        }
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "Selecionar imagem"), REQUEST_GALLERY);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode != RESULT_OK || data == null) {
            return;
        }

        if (requestCode == REQUEST_GALLERY) {
            loadGalleryImage(data.getData());
        } else if (requestCode == REQUEST_CAMERA) {
            Object image = data.getExtras() != null ? data.getExtras().get("data") : null;
            if (image instanceof Bitmap) {
                setSelectedImage((Bitmap) image);
            }
        }
    }

    private void loadGalleryImage(Uri uri) {
        if (uri == null) {
            return;
        }

        try {
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
            setSelectedImage(bitmap);
        } catch (IOException error) {
            binding.txtResult.setText("Falha ao carregar imagem: " + error.getMessage());
        }
    }

    private void setSelectedImage(Bitmap bitmap) {
        selectedBitmap = bitmap;
        binding.imageView.setImageBitmap(bitmap);
        binding.segmentedImageView.setImageDrawable(null);
        binding.txtResult.setText("Imagem carregada. Pronta para segmentar.");
    }

    private void runSegmentation() {
        if (selectedBitmap == null) {
            binding.txtResult.setText("Selecione uma imagem primeiro.");
            return;
        }

        binding.txtResult.setText("Executando segmentacao FP32...");

        try {
            SegmentationResult result = getSegmenter().segment(selectedBitmap);
            binding.segmentedImageView.setImageBitmap(result.overlayBitmap);
            binding.txtResult.setText(buildResultText(result.classPixelCounts));
        } catch (Throwable error) {
            binding.txtResult.setText("Falha na segmentacao: " + error.getMessage());
        }
    }

    private Segmenter getSegmenter() throws IOException {
        if (segmenter == null) {
            segmenter = new Segmenter(getAssets(), modelFile);
        }
        return segmenter;
    }

    private String buildResultText(int[] counts) {
        int total = 0;
        for (int count : counts) {
            total += count;
        }
        total = Math.max(total, 1);

        StringBuilder builder = new StringBuilder("Segmentacao concluida (modelo FP32)");
        for (int index = 0; index < classes.length; index++) {
            int count = index < counts.length ? counts[index] : 0;
            double percent = count * 100.0 / total;
            builder.append("\n")
                    .append(classes[index])
                    .append(": ")
                    .append(String.format(Locale.US, "%.1f", percent))
                    .append("%");
        }
        return builder.toString();
    }

    @Override
    protected void onDestroy() {
        if (segmenter != null) {
            segmenter.close();
        }
        super.onDestroy();
    }
}
