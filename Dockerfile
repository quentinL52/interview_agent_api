# Dockerfile
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copier les fichiers de requirements
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copier tout le code
COPY --chown=user . /app

# Exposer le port 7860 (requis par HF Spaces)
EXPOSE 7860

# Lancer l'application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]