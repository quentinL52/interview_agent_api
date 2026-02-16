#!/bin/bash

# Start Celery worker in the background
echo "Starting Celery worker..."
celery -A src.celery_app worker --loglevel=info &

# Start FastAPI server
# Using exec to make uvicorn the main process (PID 1)
echo "Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port 7860
