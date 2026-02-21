import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.services.graph_service import GraphInterviewProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Interview Simulation API",
    description="API for interview simulations.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=True,
)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class HealthCheck(BaseModel):
    status: str = "ok"

@app.get("/", response_model=HealthCheck, tags=["Status"])
async def health_check():
    return HealthCheck()

@app.post("/simulate-interview/")
@limiter.limit("5/minute")
async def simulate_interview(request: Request):
    """
    This endpoint receives the interview data, instantiates the graph processor
    and starts the conversation.
    """
    try:
        payload = await request.json()

        if not all(k in payload for k in ["user_id", "job_offer_id", "cv_document", "job_offer"]):
            raise HTTPException(status_code=400, detail="Missing data in payload (user_id, job_offer_id, cv_document, job_offer).")

        logger.info(f"Starting simulation for user: {payload['user_id']}")

        processor = GraphInterviewProcessor(payload)
        result = processor.invoke(payload.get("messages", []), cheat_metrics=payload.get("cheat_metrics"))

        return JSONResponse(content=result)

    except ValueError as ve:
        logger.error(f"Data validation error: {ve}", exc_info=True)
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        logger.error(f"Internal error in simulate-interview endpoint: {e}", exc_info=True)
        return JSONResponse(
            content={"error": "An internal error occurred on the assistant's server."},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
