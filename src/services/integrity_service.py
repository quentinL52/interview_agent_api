import logging
from src.services.nlp_service import NLPService
from typing import Dict, Any

logger = logging.getLogger(__name__)

class IntegrityService:
    def __init__(self):
        self.nlp = NLPService()

    def analyze_integrity(self, cv_text: str, interview_text: str, existing_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Combines stylometric analysis and AI detection to produce an integrity report.
        """
        logger.info("Starting Integrity Analysis...")
        
        # 1. AI Detection on Interview Transcript
        interview_metrics = self.nlp.compute_all_metrics(interview_text)
        
        # 2. Stylometric Consistency (CV vs Interview)
        # We only care about consistency if we have enough text
        stylometric_flag = False
        gap_details = ""
        
        if cv_text and len(cv_text) > 100:
            cv_metrics = self.nlp.compute_all_metrics(cv_text)
            
            # Compare Readability (Flesch)
            readability_gap = abs(interview_metrics["readability"] - cv_metrics["readability"])
            if readability_gap > 30: # Huge gap in complexity
                stylometric_flag = True
                gap_details += f"Readability Gap ({readability_gap}); "
                
            # Compare Vocabulary Richness
            ttr_gap = abs(interview_metrics["lexical_diversity"] - cv_metrics["lexical_diversity"])
            if ttr_gap > 0.2:
                stylometric_flag = True
                gap_details += f"Vocab Gap ({ttr_gap}); "

        # 3. Rules for Red Flags
        ai_suspicion_score = 0
        reasons = []

        # Low Perplexity = AI
        if interview_metrics["perplexity"] < 25: 
            ai_suspicion_score += 40
            reasons.append("Perplexity very low (Robotic)")
        
        # Low Burstiness = AI
        if interview_metrics["burstiness"] < 0.2:
            ai_suspicion_score += 30
            reasons.append("Low Burstiness (Monotone)")
            
        # Stylometric Mismatch
        if stylometric_flag:
            ai_suspicion_score += 20
            reasons.append(f"Style Mismatch with CV: {gap_details}")

        final_score = min(100, ai_suspicion_score)
        
        return {
            "ai_score": final_score,
            "stylometry_mismatch": stylometric_flag,
            "metrics": interview_metrics,
            "reasons": reasons,
            "raw_gap": gap_details
        }
