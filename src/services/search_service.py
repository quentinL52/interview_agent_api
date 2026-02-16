import logging
import re
from typing import Dict, List, Any, Tuple
from src.services.semantic_service import SemanticService

logger = logging.getLogger(__name__)

class SearchService:
    """
    Agent de Recherche (RAG & Grounding).
    Responsable de l'analyse des écarts (Gap Analysis) et de la détection des profils (Reconversion).
    """

    JOB_TYPES = {
        "DATA_ENGINEER": ["engineer", "ingénieur data", "mlops", "architecte", "platform"],
        "DATA_SCIENTIST": ["scientist", "science", "nlp", "computer vision", "chercher", "research"],
        "DATA_ANALYST": ["analyst", "analytics", "bi", "business intelligence", "dashboard"],
    }
    
    VERB_MAPPINGS = {
        "DATA_ENGINEER": [
            "optimiser", "déployer", "industrialiser", "automatiser", "architecturer",
            "monitorer", "scaler", "refactorer", "migrer", "contraindre"
        ],
        "DATA_SCIENTIST": [
            "entraîner", "finetuner", "expérimenter", "évaluer", "modéliser",
            "optimiser", "analyser", "comparer", "implémenter"
        ],
        "DATA_ANALYST": [
            "visualiser", "présenter", "identifier", "extraire", "recommander",
            "analyser", "synthétiser", "automatiser", "reporter"
        ]
    }
    
    # Fallback to general verbs if no type detected
    DEFAULT_VERBS = VERB_MAPPINGS["DATA_ENGINEER"] + VERB_MAPPINGS["DATA_SCIENTIST"]

    def __init__(self):
        self.semantic_service = SemanticService()

    def analyze_gap(self, cv_text: str, job_description: str) -> Dict[str, Any]:
        """
        Effectue une analyse des écarts entre le CV et l'offre.
        Retourne un dictionnaire contenant les gaps, les verbes d'action, et le statut de reconversion.
        """
        logger.info("Starting Gap Analysis...")
        
        # 0. Job Type Detection
        job_type = self._detect_job_type(job_description)
        logger.info(f"Detected Job Type: {job_type}")
        
        # 1. Action Verbs Extraction (Dynamic based on Job Type)
        target_verbs = self.VERB_MAPPINGS.get(job_type, self.DEFAULT_VERBS)
        found_verbs = self._extract_action_verbs(cv_text, target_verbs)
        
        # Score normalized by a reasonable expectation (e.g. finding 3 distinct verbs is good)
        production_score = min(1.0, len(found_verbs) / 4.0)
        
        # 2. Semantic Grounding
        semantic_score = self.semantic_service.compute_similarity(cv_text, job_description)
        
        # 3. Reconversion Reporting
        is_reconversion, reconversion_reason = self._detect_reconversion(cv_text, job_description)
        
        return {
            "job_type": job_type,
            "semantic_score": semantic_score,
            "production_verbs_found": found_verbs,
            "production_mindset_score": production_score,
            "is_reconversion": is_reconversion,
            "reconversion_reason": reconversion_reason,
            "hidden_skill_gaps": "Analyse à compléter par LLM"
        }

    def _detect_job_type(self, job_desc: str) -> str:
        """Détermine le type de poste (Engineer, Scientist, Analyst) d'après la description."""
        text_lower = job_desc.lower()
        
        scores = {k: 0 for k in self.JOB_TYPES.keys()}
        
        for j_type, keywords in self.JOB_TYPES.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[j_type] += 1
        
        # Return key with max score, default to GENERAL if no matches or ties (logic simplified)
        best_match = max(scores, key=scores.get)
        if scores[best_match] == 0:
            return "GENERAL_TECH"
            
        return best_match

    def _extract_action_verbs(self, text: str, target_verbs: List[str]) -> List[str]:
        """Extrait les verbes d'action clés présents dans le texte."""
        text_lower = text.lower()
        found = []
        for verb in target_verbs:
            # Simple word boundary check
            if re.search(r'\b' + re.escape(verb) + r'\w*', text_lower):
                found.append(verb)
        return list(set(found))

    def _detect_reconversion(self, cv_text: str, job_desc: str) -> Tuple[bool, str]:
        """
        Détecte si le candidat est en reconversion.
        Logique simple: Mots clés 'formation', 'bootcamp', 'reconversion' + manque d'xp longue durée dans le domaine cible.
        """
        cv_lower = cv_text.lower()
        
        reconversion_keywords = ["reconversion", "bootcamp", "formation intensive", "rncp", "transition professionnelle"]
        for kw in reconversion_keywords:
            if kw in cv_lower:
                return True, f"Mot-clé détecté : '{kw}'"
        
        # Note: A more robust check would involve parsing dates and titles, 
        # but this simple heuristic allows flagging potential profiles for the Agents to confirm.
        return False, "Parcours classique apparent"
