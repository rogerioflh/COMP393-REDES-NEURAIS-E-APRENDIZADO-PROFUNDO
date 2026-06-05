# COMP393 - Redes Neurais e Aprendizado Profundo

Repositório de atividades práticas da disciplina de Redes Neurais e Aprendizado Profundo. O conteúdo está organizado em três blocos principais:

1. fundamentos de redes convolucionais, implementados passo a passo com NumPy;
2. classificação de imagens com CNNs, métricas além da acurácia, desbalanceamento, transferência de aprendizado e Grad-CAM;
3. segmentação semântica com PyTorch, exportação para TensorFlow Lite, teste local em navegador e aplicativo Android.

## Estrutura geral

```text
.
+-- CNN_basics/
|   +-- Cópia_de_Convolution_model_Step_by_Step_v2a.ipynb
|   +-- cnn_utils.py
|   +-- test_utils.py
|   +-- images/
+-- CNN_classification/
|   +-- README.md
|   +-- cnn_classification_assignment.ipynb
|   +-- final_report_template.md
|   +-- generate_cnn_assignment_notebook.py
+-- app_segmentation/
|   +-- README.md
|   +-- docs/
|   +-- training/
|   +-- local_web_app/
|   +-- android_app/
+-- .gitignore
```

## Base matemática usada nas atividades

As atividades trabalham com tensores de imagem e operações convolucionais. Para uma entrada

```text
X in R^(m x H x W x C)
```

onde `m` é o número de exemplos, `H` é a altura, `W` é a largura e `C` é o número de canais, uma convolução 2D calcula ativações a partir de janelas locais. Para um filtro `W_k` e viés `b_k`, a saída em uma posição espacial pode ser descrita por:

```text
Z[i, h, w, k] = sum(a_slice_prev * W_k) + b_k
```

Com filtro de tamanho `f`, padding `p` e stride `s`, as dimensões espaciais da saída são:

```text
H_out = floor((H_in + 2p - f) / s) + 1
W_out = floor((W_in + 2p - f) / s) + 1
```

Nas atividades de classificação, a predição final normalmente é:

```text
y_hat = argmax_c p(c | x)
```

Nas atividades de segmentação, a decisão é feita por pixel:

```text
class(y, x) = argmax_c logits(c, y, x)
```

Para avaliar segmentação, o notebook usa acurácia média e IoU/Jaccard. Para uma classe `c`:

```text
IoU_c = |Pred_c intersect True_c| / |Pred_c union True_c|
```

O IoU médio é calculado pela média dos valores por classe.

## `CNN_basics`

Pasta dedicada aos fundamentos operacionais de redes convolucionais. O artefato principal é o notebook:

```text
CNN_basics/Cópia_de_Convolution_model_Step_by_Step_v2a.ipynb
```

### Atividade proposta

Implementar manualmente as principais operações de uma CNN antes de usar frameworks de alto nível. A ideia é entender como as dimensões, filtros, padding, stride e pooling funcionam internamente.

### Implementações feitas

No notebook foram preenchidas as funções avaliadas:

- `zero_pad(X, pad)`: aplica padding zero nas dimensões de altura e largura.
- `conv_single_step(a_slice_prev, W, b)`: calcula a resposta escalar de um filtro em uma janela local.
- `conv_forward(A_prev, W, b, hparameters)`: implementa a propagação direta de uma camada convolucional usando loops explícitos sobre exemplos, posições espaciais e canais de saída.
- `pool_forward(A_prev, hparameters, mode)`: implementa pooling direto nos modos `max` e `average`.

### Arquivos de suporte

- `cnn_utils.py`: funções auxiliares para carregar dados, criar mini-batches, converter rótulos para one-hot e realizar predição.
- `test_utils.py`: funções de teste para validar tipo, formato e valores das implementações.
- `images/`: figuras usadas pelo notebook para explicar padding, convolução, pooling, arquitetura e exemplos visuais.

Essa pasta é uma atividade de base matemática e algorítmica: o foco não é treinar um modelo final, mas implementar as operações que compõem uma CNN.

## `CNN_classification`

Pasta dedicada à classificação de imagens com redes convolucionais e análise crítica de métricas.

### Arquivos principais

- `cnn_classification_assignment.ipynb`: notebook principal da atividade.
- `README.md`: instruções resumidas para executar o notebook no Google Colab.
- `final_report_template.md`: estrutura sugerida para um relatório final.
- `generate_cnn_assignment_notebook.py`: script reprodutível que gera o notebook e o template de relatório.

### Atividade proposta

Estudar por que acurácia isolada não é suficiente para avaliar modelos de classificação. A atividade compara cenários com dados balanceados, dados desbalanceados, transferência de aprendizado e inspeção visual por Grad-CAM.

### Cenário 1: CIFAR-10 com CNN treinada do zero

O notebook usa o dataset CIFAR-10, com 10 classes de imagens RGB 32x32.

Implementações feitas:

- separação explícita entre treino, validação e teste;
- CNN baseline com blocos `Conv2D`, `MaxPooling2D`, `Flatten`, camada densa e saída `softmax`;
- CNN melhorada com data augmentation, batch normalization, dropout, regularização L2, early stopping e redução de taxa de aprendizado;
- avaliação com acurácia, macro F1, matriz de confusão e exemplos de acertos/erros com alta confiança;
- comparação entre modelo baseline e modelo melhorado.

### Cenário 2: DermaMNIST com classes desbalanceadas

O notebook usa DermaMNIST para discutir classificação em contexto médico-style, onde a distribuição das classes é desigual.

Implementações feitas:

- carregamento do dataset pelo pacote `medmnist`;
- visualização da distribuição de classes;
- cálculo da acurácia trivial da classe majoritária;
- CNN para classificação multiclasse;
- avaliação por acurácia balanceada, macro precision, macro recall, macro F1 e recall por classe;
- treinamento baseline;
- treinamento com `class_weight`;
- treinamento com oversampling aleatório das classes minoritárias;
- comparação dos métodos por métricas macro e recall mínimo por classe.

Esse cenário mostra que um modelo pode ter boa acurácia global e ainda falhar nas classes raras.

### Cenário 3: Oxford-IIIT Pet com transferência de aprendizado

O notebook usa Oxford-IIIT Pet para classificação de raças de cães e gatos.

Implementações feitas:

- carregamento do dataset via `tensorflow_datasets`;
- pré-processamento e redimensionamento das imagens;
- modelo com `MobileNetV2` pré-treinada na ImageNet;
- treinamento inicial apenas da cabeça classificadora;
- fine-tuning das últimas camadas do backbone;
- comparação entre cabeça congelada e modelo ajustado;
- Grad-CAM para visualizar regiões da imagem associadas à decisão do modelo;
- análise de acertos confiantes, erros confiantes e exemplos de baixa confiança.

O objetivo desse bloco é mostrar que transferência de aprendizado não transfere "conhecimento semântico perfeito"; ela reaproveita filtros e representações visuais que podem ou não ser adequados ao novo domínio.

## `app_segmentation`

Pasta consolidada da atividade de segmentação semântica e implantação.

### Objetivo da atividade

Treinar um modelo de segmentação semântica no dataset Oxford-IIIT Pet, avaliar métricas básicas, visualizar predições e disponibilizar a inferência em:

- uma interface local em navegador;
- um aplicativo Android Java com TensorFlow Lite.

As classes usadas na segmentação são:

```text
0 = Pet
1 = Fundo
2 = Borda
```

As máscaras originais do Oxford-IIIT Pet são 1-indexadas. Durante o treinamento, elas são convertidas para 0-index:

```text
target = mask_original - 1
```

### `app_segmentation/training`

Contém a parte de treinamento, avaliação e exportação do modelo.

Arquivos principais:

- `segmentation-practice-2025-1.ipynb`: notebook de treinamento e avaliação.
- `export_tflite.py`: script de exportação da `SimpleUNet` treinada para TFLite FP32.
- `requirements.txt`: dependências do treinamento.
- `requirements-export.txt`: dependências específicas da exportação.
- `README-training.md`: instruções do fluxo de treinamento/exportação.

Implementações feitas no notebook:

- uso de PyTorch, TorchVision e TorchMetrics;
- carregamento do Oxford-IIIT Pet com `target_types="segmentation"`;
- redimensionamento das imagens e máscaras para `128x128`;
- normalização das imagens com médias e desvios do ImageNet:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

- definição de uma `SimpleUNet` com blocos `DoubleConv`, encoder, bottleneck, decoder e skip connections;
- treinamento da U-Net com `CrossEntropyLoss` e otimizador Adam;
- treinamento opcional de uma `FCN-ResNet50` pré-treinada, adaptada para 3 classes;
- salvamento dos melhores pesos por menor perda de validação;
- cálculo manual de IoU médio;
- cálculo de IoU/Jaccard e acurácia macro com TorchMetrics;
- visualização de imagem original, máscara real, predição da U-Net, predição da FCN e comparação lado a lado.

Artefatos gerados localmente:

- `SimpleUNet_best.pth`;
- `FCN_best.pth`;
- `training/data/`.

Esses artefatos são produtos do treino e estão ignorados pelo Git para evitar versionamento de dataset e pesos grandes.

### `app_segmentation/local_web_app`

Implementa uma interface web local para testar a segmentação sem depender de Android Studio, ADB, emulador ou celular.

Arquivos principais:

- `server.py`: servidor HTTP local e pipeline de inferência.
- `requirements.txt`: dependências da interface local.
- `README.md`: instruções de execução.

Implementações feitas:

- definição da mesma arquitetura `SimpleUNet` usada no treinamento;
- carregamento de `training/SimpleUNet_best.pth`;
- upload de imagens `.jpg`, `.jpeg`, `.png` ou `.webp`;
- redimensionamento para o tamanho esperado pelo modelo;
- normalização com os mesmos `mean` e `std` do treinamento;
- inferência com PyTorch;
- conversão dos logits em máscara por `argmax`;
- geração da máscara RGB;
- sobreposição da máscara na imagem original;
- cálculo percentual de pixels em cada classe;
- renderização de uma página HTML com imagem original, overlay, máscara isolada e percentuais das classes.

Execução:

```powershell
cd app_segmentation
python local_web_app\server.py
```

Depois, abrir:

```text
http://localhost:8000
```

### `app_segmentation/android_app`

Projeto Android Java que executa o modelo exportado para TensorFlow Lite.

Arquivos e diretórios principais:

- `build.gradle`, `settings.gradle`, `gradle.properties`: configuração Gradle do projeto.
- `gradlew`, `gradlew.bat`, `gradle/wrapper/`: wrapper do Gradle.
- `app/build.gradle`: configuração do módulo Android.
- `app/src/main/AndroidManifest.xml`: permissões e configuração do app.
- `app/src/main/res/layout/activity_main.xml`: tela principal.
- `app/src/main/java/com/andre/tflite/segmentation/MainActivity.java`: fluxo de câmera, galeria e botão de segmentação.
- `app/src/main/java/com/andre/tflite/segmentation/Segmenter.java`: carregamento TFLite, pré-processamento, inferência, decodificação e overlay.
- `app/src/main/java/com/andre/tflite/segmentation/SegmentationResult.java`: estrutura de retorno da segmentação.
- `app/src/main/assets/README.md`: instrução para posicionar `model.tflite`.

Configuração Android:

```text
namespace: com.andre.tflite.segmentation
minSdk: 24
targetSdk: 35
compileSdk: 36
TensorFlow Lite: 2.14.0
```

Implementações feitas:

- seleção de imagem pela galeria;
- captura de imagem pela câmera;
- carregamento do modelo `model.tflite` nos assets;
- tentativa de execução com NNAPI, depois GPU, depois CPU como fallback;
- suporte a entrada `NHWC` ou `NCHW`;
- suporte a saída `NHWC`, `NCHW` ou mapa de rótulos;
- suporte a tensores `float32` e quantizados;
- normalização RGB compatível com o treinamento;
- decodificação da saída em máscara por pixel;
- geração de overlay colorido sobre a imagem original;
- exibição dos percentuais das classes `Pet`, `Fundo` e `Borda`.

Antes de executar o app mobile, gere o modelo:

```powershell
cd app_segmentation\training
pip install -r requirements-export.txt
python export_tflite.py
```

O arquivo esperado pelo app é:

```text
app_segmentation/android_app/app/src/main/assets/model.tflite
```

Requisito recomendado:

```text
JDK 17 ou JDK 21
```

O próprio README interno alerta que JDK 26 pode causar falha com a versão de Gradle configurada.

### `app_segmentation/docs`

Documentação de teste e evidências visuais.

Arquivos principais:

- `localhost_test.md`: passo a passo para testar a segmentação no navegador.
- `mobile_app_test.md`: passo a passo para testar o app Android.
- `images/select-folder-section.png`: imagem da etapa de seleção de arquivo.
- `images/localhost-segmentation-output.png`: exemplo de saída da segmentação local.

Essa pasta documenta o critério visual de sucesso: imagem original, máscara segmentada, overlay colorido e percentuais das classes.

## Como executar cada atividade

### Fundamentos de CNN

Abra o notebook:

```text
CNN_basics/Cópia_de_Convolution_model_Step_by_Step_v2a.ipynb
```

Execute as células em ordem para validar `zero_pad`, `conv_single_step`, `conv_forward` e `pool_forward`.

### Classificação com CNN

Recomendado executar no Google Colab com GPU:

```text
CNN_classification/cnn_classification_assignment.ipynb
```

O notebook baixa os datasets necessários e executa os cenários CIFAR-10, DermaMNIST e Oxford-IIIT Pet.

### Segmentação local

Depois de treinar a `SimpleUNet` e obter `training/SimpleUNet_best.pth`:

```powershell
cd app_segmentation
pip install -r training\requirements.txt
pip install -r local_web_app\requirements.txt
python local_web_app\server.py
```

Acesse:

```text
http://localhost:8000
```

### Segmentação no Android

Gere o modelo TFLite:

```powershell
cd app_segmentation\training
pip install -r requirements-export.txt
python export_tflite.py
```

Depois abra:

```text
app_segmentation/android_app
```

no Android Studio e execute o aplicativo em um dispositivo ou emulador.

## Controle de arquivos grandes e gerados

O `.gitignore` do repositório ignora:

- caches Python e Jupyter;
- ambientes virtuais;
- diretórios `.gradle/`;
- diretórios `build/`;
- APKs/AABs gerados;
- dataset local da segmentação;
- checkpoints `.pth` do treinamento.

Isso evita versionar caches, builds, datasets baixados e pesos grandes. O código, notebooks, documentação, imagens explicativas e configuração Android ficam versionados.

## Resumo das entregas

- `CNN_basics`: implementação matemática das operações básicas de CNN.
- `CNN_classification`: laboratório completo de classificação com avaliação crítica de métricas.
- `app_segmentation/training`: treinamento e avaliação de segmentadores semânticos.
- `app_segmentation/local_web_app`: interface local para demonstrar segmentação sem Android.
- `app_segmentation/android_app`: aplicativo Android Java com inferência TensorFlow Lite.
- `app_segmentation/docs`: guias de teste e imagens de evidência.
