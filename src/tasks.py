from src.celery_app import celery_app
from src.tools.analysis_tools import trigger_interview_analysis
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def run_analysis_task(self, user_id, job_offer_id, job_description, conversation_history, cv_content, cheat_metrics=None, simulation_report=None):
    """
    Background task to run the feedback analysis.
    """
    logger.info(f"Starting background analysis for job {job_offer_id}")
    try:
        result = trigger_interview_analysis.invoke({
            "user_id": user_id,
            "job_offer_id": job_offer_id,
            "job_description": job_description,
            "conversation_history": conversation_history,
            "cv_content": cv_content,
            "cheat_metrics": cheat_metrics or {},
            "simulation_report": simulation_report
        })
        logger.info("Background analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Background analysis failed (attempt {self.request.retries + 1}): {e}")
        raise self.retry(exc=e)
