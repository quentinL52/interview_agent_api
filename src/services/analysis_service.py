import logging
import json
from typing import Dict, List, Any, Optional
from src.services.feedback_crew import FeedbackCrew
from src.services.integrity_service import IntegrityService
from src.services.search_service import SearchService
from src.config import crew_openai
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def _flatten_dict_values(d) -> list:
    """Recursively flatten all values from a nested dict/list structure into a list of strings."""
    text = []
    if not isinstance(d, dict):
        return [str(d)]
    for k, v in d.items():
        if isinstance(v, dict):
            text.extend(_flatten_dict_values(v))
        elif isinstance(v, list):
            for i in v:
                if isinstance(i, dict):
                    text.extend(_flatten_dict_values(i))
                else:
                    text.append(str(i))
        else:
            text.append(str(v))
    return text


class AnalysisService:
    def __init__(self):
        self.integrity_service = IntegrityService()
        self.search_service = SearchService()
        self.llm = crew_openai()

    def run_analysis(self, conversation_history: List[Dict[str, Any]], job_description: str, cv_content: str, cheat_metrics: Dict[str, Any] = None, simulation_report: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs the feedback analysis using CrewAI with enhanced pre-processing (Multi-Agent RAG).
        If simulation_report is provided, it is used as the result, bypassing CrewAI.
        """
        logger.info("Starting Interview Feedback Analysis...")
        
        if simulation_report:
            logger.info("Using pre-computed Simulation Report. Bypassing CrewAI.")
            try:
                # Run Integrity Analysis (Cheating detection)
                transcript_text = " ".join([m.get('content', '') for m in conversation_history if m.get('role') == 'user'])
                # Extract CV text (flatten values for stylometry)
                cv_text = ". ".join(_flatten_dict_values(cv_content)) if isinstance(cv_content, dict) else str(cv_content)

                integrity_report = self.integrity_service.analyze_integrity(
                    cv_text=cv_text,
                    interview_text=transcript_text,
                    existing_metrics=cheat_metrics
                )
                
                result = simulation_report.copy()
                result["integrity_report"] = integrity_report
                result["cheat_metrics"] = cheat_metrics
                
                return result
                
            except Exception as e:
                logger.error(f"Error enriching simulation report: {e}")
                return simulation_report

        logger.info("No Simulation Report provided. Fallback to CrewAI.")
        
        try:
            # Parse inputs
            if isinstance(job_description, str):
                try:
                    job_offer_data = json.loads(job_description)
                except json.JSONDecodeError:
                    job_offer_data = {"description": job_description}
            else:
                job_offer_data = job_description

            # Prepare text for analysis
            transcript_text = " ".join([m.get('content', '') for m in conversation_history if m.get('role') == 'user'])
            
            # Extract CV text (flatten values for stylometry)
            cv_text = ". ".join(_flatten_dict_values(cv_content)) if isinstance(cv_content, dict) else str(cv_content)

            # Run RAG Gap Analysis (Search Agent Grounding)
            job_mission = job_offer_data.get('mission', '') or job_offer_data.get('description', '')
            gap_analysis = self.search_service.analyze_gap(cv_text, job_mission)
            
            # Run Integrity Analysis
            integrity_report = self.integrity_service.analyze_integrity(
                cv_text=cv_text,
                interview_text=transcript_text,
                existing_metrics=cheat_metrics
            )
            
            # Detect Job Seniority (Robust LLM Method)
            seniority = self._analyze_job_context_with_llm(job_offer_data)
            
            # Merge Metrics
            enhanced_metrics = cheat_metrics or {}
            enhanced_metrics.update({
                "integrity_report": integrity_report,
                "semantic_score": gap_analysis.get('semantic_score', 0.0),
                "required_seniority": seniority
            })
            
            logger.info(f"Enhanced Metrics: {enhanced_metrics}")
            logger.info(f"Gap Analysis: {gap_analysis}")

            crew = FeedbackCrew(
                job_offer=job_offer_data,
                cv_content=cv_content,
                conversation_history=conversation_history,
                cheat_metrics=enhanced_metrics,
                gap_analysis=gap_analysis
            )
            
            result = crew.run()
            logger.info("Feedback Analysis completed successfully.")
            return result

        except Exception as e:
            logger.error(f"Error during feedback analysis: {e}", exc_info=True)
            return {"error": str(e)}

    def _analyze_job_context_with_llm(self, job_data: Dict[str, Any]) -> str:
        """
        Uses LLM to detect seniority context, avoiding false positives (e.g. 'Reporting to Head of Data').
        """
        try:
            description = str(job_data)
            prompt = f"""
            ANALYSE LE CONTEXTE DE CETTE OFFRE D'EMPLOI :
            {description}
            
            TÂCHE : Détermine le niveau de séniorité requis pour le CANDIDAT.
            
            RÈGLES CRITIQUES :
            1. "STAGE", "ALTERNANCE", "APPRENTISSAGE" = TOUJOURS "JUNIOR".
            2. Ignore les mentions de la hiérarchie (ex: "Sous la direction du Senior Manager" -> Le poste n'est PAS Senior).
            3. "Débutant", "Junior", "0-2 ans", "Sortie d'école" = "JUNIOR".
            4. "Lead", "Expert", "Manager", "+5 ans", "Architecte" = "SENIOR".
            5. Sinon -> "MID".

            Réponds UNIQUEMENT par un seul mot : JUNIOR, SENIOR ou MID.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip().upper()
            
            if result not in ["JUNIOR", "SENIOR", "MID"]:
                return "MID" # Fallback
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM seniority detection: {e}")
            return "MID" # Safe fallback