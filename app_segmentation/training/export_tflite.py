import argparse
from pathlib import Path

import torch
import torch.nn as nn

try:
    import litert_torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "litert_torch nao esta instalado. Em Colab/Linux, execute: "
        "pip install -r requirements-export.txt"
    ) from exc


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=(64, 128)):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def resolve_path(base_dir, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main():
    base_dir = Path(__file__).resolve().parent
    default_output = "../android_app/app/src/main/assets/model.tflite"

    parser = argparse.ArgumentParser(description="Exporta a SimpleUNet treinada para TFLite FP32.")
    parser.add_argument("--weights", default="SimpleUNet_best.pth", help="Arquivo .pth salvo pelo notebook.")
    parser.add_argument("--output", default=default_output, help="Destino do arquivo .tflite.")
    parser.add_argument("--image-size", type=int, default=128, help="Tamanho quadrado de entrada usado no treino.")
    parser.add_argument("--num-classes", type=int, default=3, help="Numero de classes da segmentacao.")
    args = parser.parse_args()

    weights_path = resolve_path(base_dir, args.weights)
    output_path = resolve_path(base_dir, args.output)

    if not weights_path.exists():
        raise SystemExit(f"Pesos nao encontrados: {weights_path}")

    model = SimpleUNet(out_channels=args.num_classes)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    sample_inputs = (torch.randn(1, 3, args.image_size, args.image_size),)
    edge_model = litert_torch.convert(model, sample_inputs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output_path))
    print(f"Modelo TFLite FP32 exportado para: {output_path}")


if __name__ == "__main__":
    main()
