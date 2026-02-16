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

# Rendre le script de démarrage exécutable
RUN chmod +x /app/start.sh

# Exposer le port 7860 (requis par HF Spaces)
EXPOSE 7860

# Lancer l'application via le script de démarrage
CMD ["./start.sh"]