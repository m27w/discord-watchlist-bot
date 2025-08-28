FROM python:3.12-slim

WORKDIR /app

# Fonts for matplotlib labels (optional but nice)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run continuously
CMD ["python", "discord_watchlist_bot.py"]
