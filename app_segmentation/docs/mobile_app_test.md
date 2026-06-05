# Teste do Aplicativo Mobile

Este guia descreve como validar a parte mobile da atividade depois que o notebook gerar `SimpleUNet_best.pth`.

## 1. Gerar o Modelo TFLite

No terminal do ambiente em que o treino foi executado:

```bash
cd training
pip install -r requirements-export.txt
python export_tflite.py
```

O script deve criar:

```text
android_app/app/src/main/assets/model.tflite
```

## 2. Abrir o Projeto Android

Pre-requisitos:

- Android Studio instalado.
- JDK 17 ou JDK 21 configurado para o Gradle/Android Studio.
- Evite JDK 26 para este projeto, pois o Gradle/Kotlin usado aqui falha ao interpretar `java version "26.0.1"`.

Abra no Android Studio:

```text
app_segmentation/android_app
```

Espere o Gradle sincronizar. O codigo mobile principal esta em:

```text
android_app/app/src/main/java/com/andre/tflite/segmentation
```

Arquivos centrais:

- `MainActivity.java`: camera/galeria e botao `Segmentar`.
- `Segmenter.java`: carregamento TFLite, pre-processamento, inferencia e overlay.
- `SegmentationResult.java`: retorno da mascara, overlay e contagem por classe.
- `activity_main.xml`: tela com imagem original e imagem segmentada.

## 3. Realizar a Predicao no App

Execute o app em um smartphone ou emulador Android.

Fluxo esperado:

1. Toque em `Tirar foto` para usar a camera, ou `Carregar` para selecionar uma imagem.
2. Confirme que a imagem aparece na secao `Original`.
3. Toque em `Segmentar`.
4. Aguarde a mensagem `Segmentacao concluida`.
5. Confira a secao `Segmentacao`, que deve exibir a imagem original com a mascara colorida sobreposta.

## 4. Criterio Visual de Sucesso

O print final deve mostrar:

- uma imagem carregada ou capturada;
- a saida segmentada com overlay colorido;
- o texto com percentuais das classes `Pet`, `Fundo` e `Borda`.

Se aparecer `Falha na segmentacao`, verifique se `model.tflite` existe em `android_app/app/src/main/assets` e se ele foi gerado pelo `training/export_tflite.py`, nao pelo app antigo de classificacao.
