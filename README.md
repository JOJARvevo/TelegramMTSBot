# Telegram Gender Bot

Telegram bot that classifies the gender of the person on a photo with a pre-trained FaceNet + MLP model.

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

`bot.py` uses the MLP classifier from `best.pth`.

To run the version that extracts Facenet embeddings (`best.pth`), execute:

```bash
export TELEGRAM_BOT_TOKEN=<your_token>
python bot2.py
```