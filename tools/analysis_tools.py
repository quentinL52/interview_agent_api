import logging
from langchain_core.tools import tool
from src.services.analysis_service import AnalysisService
import json
import os
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import httpx

logger = logging.getLogger(__name__)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

class InterviewAnalysisArgs(BaseModel):
    """Arguments for the trigger_interview_analysis tool."""
    user_id: str = Field(..., description="The unique identifier for the user.")
    job_offer_id: str = Field(..., description="The unique identifier for the job offer.")
    job_description: str = Field(..., description="The full JSON string of the job offer description.")
    conversation_history: List[Dict[str, Any]] = Field(..., description="The complete conversation history between the user and the agent.")
    cv_content: str = Field(..., description="The content of the candidate's CV (JSON string or text).")
    cheat_metrics: Dict[str, Any] = Field(None, description="Metrics related to copy-paste behavior.")

@tool("trigger_interview_analysis", args_schema=InterviewAnalysisArgs)
def trigger_interview_analysis(user_id: str, job_offer_id: str, job_description: str, conversation_history: List[Dict[str, Any]], cv_content: str, cheat_metrics: Dict[str, Any] = None):
    """
    Call this tool to end the interview and launch the final analysis.
    Arguments: user_id, job_offer_id, job_description, conversation_history, cv_content.
    """
    try:
        logger.info(f"Tool 'trigger_interview_analysis' called for user_id: {user_id}")
        
        analysis_service = AnalysisService() 
        
        feedback_data = analysis_service.run_analysis(
            conversation_history=conversation_history,
            job_description=job_description,
            cv_content=cv_content,
            cheat_metrics=cheat_metrics
        )
        
        feedback_payload = {
            "user_id": user_id,
            "interview_id": job_offer_id, 
            "feedback_content": feedback_data,
            "feedback_date": datetime.utcnow().isoformat()
        }
        
        try:
            response = httpx.post(f"{BACKEND_API_URL}/api/v1/feedback/", json=feedback_payload, timeout=30.0)
            response.raise_for_status()
            logger.info("Feedback saved to Backend API successfully.")
        except Exception as api_err:
            logger.error(f"Failed to save feedback to API: {api_err}")
        
        return "Analysis triggered and completed successfully."

    except Exception as e:
        logger.error(f"Error in analysis tool: {e}", exc_info=True)
        return "An error occurred while launching the analysis."
