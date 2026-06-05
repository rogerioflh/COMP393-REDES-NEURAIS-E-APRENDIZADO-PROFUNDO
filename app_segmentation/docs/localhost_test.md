# Teste Localhost Sem ADB

Use este fluxo quando quiser validar a segmentacao em navegador, sem Android Studio, sem emulador e sem depuracao ADB.

## Passos

1. Abra um terminal na pasta principal:

```powershell
cd app_segmentation
```

2. Instale as dependencias, se ainda nao instalou:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r training\requirements.txt
pip install -r local_web_app\requirements.txt
```

3. Confirme que o peso treinado existe:

```text
training/SimpleUNet_best.pth
```

4. Rode o servidor local:

```powershell
python local_web_app\server.py
```

5. Abra:

```text
http://localhost:8000
```

6. Na pagina:

- clique em `Choose File`;
- selecione uma imagem `.jpg`, `.jpeg`, `.png` ou `.webp`;
- prefira imagem de cachorro ou gato;
- nao selecione `.pth`, `.tflite`, notebook, PDF ou codigo;
- clique em `Segmentar`.

7. Confira:

- imagem original;
- mascara isolada;
- imagem com mascara sobreposta;
- percentuais das classes `Pet`, `Fundo` e `Borda`.

## Observacao

Este teste valida o modelo e a experiencia visual de segmentacao em uma interface web local. Ele nao valida o APK Android nem o codigo Java do app mobile.
