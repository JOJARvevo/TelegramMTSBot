# Telegram Gender Bot

Telegram bot that classifies the gender of the person on a photo with a pre-trained EfficientNet-B0 model.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export TELEGRAM_BOT_TOKEN=<your_token>
python bot.py
```

`bot.py` uses the EfficientNet-B0 classifier from `efficientnet_gender.pth`.

To run the version that extracts Facenet embeddings (`best.pth`), execute:

```bash
export TELEGRAM_BOT_TOKEN=<your_token>
python bot2.py
```

On the first launch `bot2.py` downloads the pretrained Facenet weights to the local torch cache (SSL trust is configured through `certifi`). After the bot starts, send it a face photo as a regular photo or as an image document — it will reply with either “Man” or “Woman” plus the confidence.
