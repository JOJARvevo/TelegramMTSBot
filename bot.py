import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple

import certifi
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from facenet_pytorch import InceptionResnetV1


os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 160
MODEL_PATH = Path(__file__).with_name("best.pth")
DEFAULT_TOKEN = "SECRET"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def load_facenet(backbone_device: torch.device) -> nn.Module:
    # Pretrained FaceNet model used to extract embeddings during training
    model = InceptionResnetV1(pretrained="vggface2").eval()
    return model.to(backbone_device)


def load_classifier(checkpoint_path: Path, classifier_device: torch.device) -> EmbeddingClassifier:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
    classifier = EmbeddingClassifier().to(classifier_device)
    state_dict = torch.load(checkpoint_path, map_location=classifier_device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


FACENET = load_facenet(device)
CLASSIFIER = load_classifier(MODEL_PATH, device)


def predict_gender(image: Image.Image) -> Tuple[str, float]:
    tensor = inference_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        flipped = torch.flip(tensor, dims=[-1])  # mirror augmentation, like during embedding extraction
        emb_orig = FACENET(tensor)
        emb_flip = FACENET(flipped)
        embedding = (emb_orig + emb_flip) / 2
        logits = CLASSIFIER(embedding).view(-1)
        probability = torch.sigmoid(logits)[0].item()
    label = "Мужчина" if probability >= 0.5 else "Женщина"
    return label, probability


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Отправь мне фотографию лица, и я постараюсь определить пол человека на изображении. Если на фотографии большой фон, я не смогу дать точную оценку"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Просто отправь фотографию лица. Я обработаю изображение и скажу, мужчина или женщина. Если на фотографии большой фон, я не смогу дать точную оценку"
    )


async def _reply_with_prediction(update: Update, data: bytes) -> None:
    await update.message.chat.send_action(action=ChatAction.TYPING)
    try:
        prediction, probability = predict_gender(Image.open(BytesIO(data)))
        if prediction == 'Женщина':
            await update.message.reply_text(f"{prediction} ({(1 - probability) * 100:.1f}% уверенность)")
        else:
            await update.message.reply_text(f"{prediction} ({probability * 100:.1f}% уверенность)")
    except Exception:
        logger.exception("Failed to process incoming image")
        await update.message.reply_text("Не удалось обработать фотографию. Попробуйте другое изображение.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return
    telegram_file = await update.message.photo[-1].get_file()
    data = await telegram_file.download_as_bytearray()
    await _reply_with_prediction(update, data)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return
    mime_type = update.message.document.mime_type or ""
    if "image" not in mime_type:
        await update.message.reply_text("Пожалуйста, отправьте изображение с лицом.")
        return
    telegram_file = await update.message.document.get_file()
    data = await telegram_file.download_as_bytearray()
    await _reply_with_prediction(update, data)


def build_application(token: str) -> Application:
    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    return application


def main() -> None:
    token = os.getenv("SECRET", DEFAULT_TOKEN)
    if token == DEFAULT_TOKEN:
        logger.warning("Установите TELEGRAM_BOT_TOKEN для боевого режима. Не расстраивайте Артёма Фукса.")
    application = build_application(token)
    logger.info("Запуск бота на Facenet-эмбеддингах...")
    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
