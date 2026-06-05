import base64
import html
import io
import sys
from cgi import FieldStorage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    import torch
    import torch.nn as nn
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Dependencias ausentes. Execute no ambiente Python do notebook:\n"
        "pip install torch torchvision pillow\n"
        "Depois rode novamente: python local_web_app\\server.py"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT_DIR / "training" / "SimpleUNet_best.pth"
IMAGE_SIZE = 128
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Pet", "Fundo", "Borda"]
MASK_COLORS = [
    (244, 67, 54, 150),
    (76, 175, 80, 0),
    (33, 150, 243, 170),
]
DISPLAY_MASK_COLORS = [
    (244, 67, 54, 255),
    (76, 175, 80, 255),
    (33, 150, 243, 255),
]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES, features=(64, 128)):
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


MODEL = None


def load_model(weights_path):
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesos nao encontrados: {weights_path}")

    model = SimpleUNet(out_channels=NUM_CLASSES)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_model(weights_path):
    global MODEL
    if MODEL is None:
        MODEL = load_model(weights_path)
    return MODEL


def image_to_tensor(image):
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).convert("RGB")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(resized.tobytes()))
    data = data.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).float() / 255.0
    normalized = (data - MEAN) / STD
    return normalized.unsqueeze(0).to(DEVICE)


def mask_to_rgba(mask, colors):
    pixels = bytearray()
    flat = mask.reshape(-1).tolist()
    for class_id in flat:
        pixels.extend(colors[int(class_id) % len(colors)])
    return Image.frombytes("RGBA", (IMAGE_SIZE, IMAGE_SIZE), bytes(pixels))


def run_prediction(image, weights_path):
    model = get_model(weights_path)
    tensor = image_to_tensor(image)

    with torch.no_grad():
        logits = model(tensor)
        mask = torch.argmax(logits, dim=1)[0].cpu()

    counts = torch.bincount(mask.reshape(-1), minlength=NUM_CLASSES).tolist()
    total = max(sum(counts), 1)

    overlay_mask = mask_to_rgba(mask, MASK_COLORS).resize(image.size, Image.NEAREST)
    display_mask = mask_to_rgba(mask, DISPLAY_MASK_COLORS).resize(image.size, Image.NEAREST)
    original_rgba = image.convert("RGBA")
    overlay = Image.alpha_composite(original_rgba, overlay_mask)

    percentages = [
        (CLASS_NAMES[index], counts[index] * 100.0 / total)
        for index in range(NUM_CLASSES)
    ]
    return overlay.convert("RGB"), display_mask.convert("RGB"), percentages


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_page(message="", original=None, overlay=None, mask=None, percentages=None):
    result_html = ""
    if original is not None and overlay is not None and mask is not None and percentages is not None:
        rows = "\n".join(
            f"<li><strong>{html.escape(name)}</strong>: {percent:.1f}%</li>"
            for name, percent in percentages
        )
        result_html = f"""
        <section class="results">
          <div>
            <h2>Original</h2>
            <img src="{image_to_data_url(original)}" alt="Imagem original">
          </div>
          <div>
            <h2>Segmentacao</h2>
            <img src="{image_to_data_url(overlay)}" alt="Imagem com mascara sobreposta">
          </div>
          <div>
            <h2>Mascara</h2>
            <img src="{image_to_data_url(mask)}" alt="Mascara segmentada">
          </div>
          <div class="metrics">
            <h2>Classes</h2>
            <ul>{rows}</ul>
          </div>
        </section>
        """

    escaped_message = html.escape(message)
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Segmentacao Semantica Local</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f4f6f8;
      color: #1c242c;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      font-size: 26px;
      margin: 0 0 8px;
    }}
    p {{
      line-height: 1.5;
    }}
    form {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      background: #ffffff;
      padding: 16px;
      border: 1px solid #d9e0e7;
      border-radius: 6px;
    }}
    button {{
      border: 0;
      background: #1167b1;
      color: #ffffff;
      padding: 10px 14px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }}
    .message {{
      margin: 16px 0;
      color: #8a4b00;
      font-weight: 600;
    }}
    .results {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}
    .results > div {{
      background: #ffffff;
      border: 1px solid #d9e0e7;
      border-radius: 6px;
      padding: 12px;
    }}
    .results h2 {{
      font-size: 16px;
      margin: 0 0 10px;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      background: #e8edf2;
    }}
    ul {{
      padding-left: 18px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Segmentacao Semantica Local</h1>
    <p>Interface local para testar o modelo treinado sem Android Studio, ADB, celular ou emulador.</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Segmentar</button>
    </form>
    <div class="message">{escaped_message}</div>
    {result_html}
  </main>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    weights_path = DEFAULT_WEIGHTS

    def do_GET(self):
        self.respond_html(render_page(message=f"Modelo: {self.weights_path}"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self.send_error(404)
            return

        try:
            form = FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            field = form["image"]
            image_bytes = field.file.read()
            original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            overlay, mask, percentages = run_prediction(original, self.weights_path)
            page = render_page(
                message="Segmentacao concluida.",
                original=original,
                overlay=overlay,
                mask=mask,
                percentages=percentages,
            )
            self.respond_html(page)
        except Exception as exc:
            self.respond_html(render_page(message=f"Falha na segmentacao: {exc}"), status=500)

    def respond_html(self, body, status=200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Servidor local para testar segmentacao sem ADB.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    args = parser.parse_args()

    Handler.weights_path = Path(args.weights).resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Servidor em http://{args.host}:{args.port}")
    print(f"Pesos: {Handler.weights_path}")
    print("Pressione Ctrl+C para encerrar.")
    server.serve_forever()


if __name__ == "__main__":
    main()
