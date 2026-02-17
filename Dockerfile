FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Définir explicitement le dossier de données NLTK pour éviter les surprises
ENV NLTK_DATA="/home/user/nltk_data"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Créer le dossier et télécharger les corpus spécifiques
RUN mkdir -p /home/user/nltk_data && \
    python -m textblob.download_corpora && \
    python -m nltk.downloader punkt_tab

COPY --chown=user . /app
RUN chmod +x /app/start.sh

EXPOSE 7860
CMD ["./start.sh"]