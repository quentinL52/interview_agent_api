import os
import sys
import json
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.graph_service import GraphInterviewProcessor

# Force re-configuration of logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='simulation_test.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Mock data
USER_ID = "test_user"
JOB_OFFER_ID = "test_job"
CV_DATA = {
    "candidat": {
        "info_personnelle": {"nom": "Doe", "prenom": "John"},
        "reconversion": {"is_reconversion": False},
        "etudiant": {
            "is_etudiant": True, 
            "niveau_etudes": "Master 2",
            "specialite": "IA",
            "latest_education_end_date": "2025"
        },
        "expériences": [{"poste": "Dev", "entreprise": "TestCorp", "durée": "2 ans"}],
        "compétences": {
            "hard_skills": ["Python", "Docker"],
            "soft_skills": ["Curiosité"],
            "skills_with_context": [
                {"skill": "Python", "context": "Projet académique"},
                {"skill": "Docker", "context": "Stage"}
            ]
        },
        "projets": {
            "professional": [
                {
                    "title": "Projet A",
                    "technologies": ["Python", "Flask"],
                    "outcomes": ["API déployée", "User base +10%"]
                }
            ],
            "personal": []
        },
        "centres_interet": ["Football", "Voyage"]
    }
}
JOB_OFFER = {
    "poste": "Backend Developer",
    "entreprise": "AIRH",
    "mission": "Develop API",
    "profil_recherche": "Passionné",
    "competences": "Python, FastAPI"
}

PAYLOAD = {
    "user_id": USER_ID,
    "job_offer_id": JOB_OFFER_ID,
    "cv_document": CV_DATA,
    "job_offer": JOB_OFFER,
    "messages": []
}

def run_simulation_start():
    logger.info("Initializing Processor for START SCENARIO...")
    processor = GraphInterviewProcessor(PAYLOAD)
    
    # 0 messages -> Should trigger IceBreaker first message
    output = processor.invoke([])
    logger.info(f"\n=== START OF INTERVIEW ===")
    logger.info(f"Agent Response: {output['response']}")
    logger.info(f"Status: {output['status']}")

def run_simulation_end():
    logger.info("Initializing Processor for END SCENARIO...")
    processor = GraphInterviewProcessor(PAYLOAD)
    
    # Simulate a history with 10 user messages (Completed flow)
    # The Orchestrator counts only HUMAN messages.
    conversation = []
    for i in range(10):
        conversation.append({"role": "user", "content": f"Response {i+1}"})
        conversation.append({"role": "assistant", "content": f"Question {i+2}"})
    
    logger.info(f"\n=== TRIGGERING END OF INTERVIEW (10 User Messages) ===")
    output = processor.invoke(conversation)
    logger.info(f"Final Agent Response: {output['response'][:100]}...")
    logger.info(f"Final Status: {output['status']}")

if __name__ == "__main__":
    try:
        run_simulation_start()
        # run_simulation_end()
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
