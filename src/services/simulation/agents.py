import json
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from src.services.simulation.schemas import (
    IceBreakerOutput, TechnicalOutput, BehavioralOutput, SituationOutput, 
    TechnicalSkillGap, ProjectTechUnderstanding, BehavioralCompetency, 
    SimulationReport
)
from src.services.simulation.scoring import (
    calculate_technical_gap_score, 
    calculate_project_tech_understanding_score, 
    calculate_behavioral_score, 
    calculate_situation_score
)

logger = logging.getLogger(__name__)

class InterviewAgentExtractor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _get_history_text(self, messages: List[BaseMessage]) -> str:
        return "\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

    def extract_icebreaker(self, messages: List[BaseMessage], cv_data: Dict[str, Any]) -> IceBreakerOutput:
        logger.info("Extracting Ice Breaker data...")
        history = self._get_history_text(messages)
        
        prompt = f"""
        Tu es un expert en analyse d'entretien. Analyse l'échange suivant (phase d'Ice Breaker) et extrais les informations structurées.
        
        CONTEXTE CANDIDAT:
        {json.dumps(cv_data.get('info_personnelle', {}), ensure_ascii=False)}
        {json.dumps(cv_data.get('reconversion', {}), ensure_ascii=False)}

        HISTORIQUE ECHANGE:
        {history}
        
        Tâche: Extraire le type de profil, l'expérience, la cohérence, la motivation, le contexte et les points à explorer.
        """
        
        extractor = self.llm.with_structured_output(IceBreakerOutput)
        return extractor.invoke([SystemMessage(content=prompt)])

    def extract_technical(self, messages: List[BaseMessage], job_offer: Dict[str, Any]) -> TechnicalOutput:
        logger.info("Extracting Technical data...")
        history = self._get_history_text(messages)
        
        prompt = f"""
        Tu es un expert technique. Analyse l'échange suivant (phase Technique) et extrais les compétences validées, les lacunes et la compréhension des technos.
        
        OFFRE:
        {json.dumps(job_offer, ensure_ascii=False)}

        HISTORIQUE ECHANGE:
        {history}
        
        Tâche: Remplir la grille d'évaluation technique. Pour les indicateurs binaires, sois strict : true seulement si le candidat l'a explicitement démontré.
        """
        
        extractor = self.llm.with_structured_output(TechnicalOutput)
        data = extractor.invoke([SystemMessage(content=prompt)])
        
        # Calculate scores - normalize all to 0-5 scale
        scores = []
        for gap in data.lacunes_explorees:
            gap.niveau_detecte = calculate_technical_gap_score(gap.indicateurs)
            normalized = (gap.niveau_detecte / 4.0) * 5.0  # 0-4 -> 0-5
            scores.append(normalized)

        for tech in data.comprehension_technos_projets:
            tech.score = calculate_project_tech_understanding_score(tech.indicateurs)
            scores.append(float(tech.score))  # already 1-5

        for val in data.competences_validees:
            scores.append(float(val.score))  # already 1-5

        if scores:
            data.score_technique_global = round(sum(scores) / len(scores), 1)
        else:
            data.score_technique_global = 0.0
            
        return data

    def extract_behavioral(self, messages: List[BaseMessage]) -> BehavioralOutput:
        logger.info("Extracting Behavioral data...")
        history = self._get_history_text(messages)
        
        prompt = f"""
        Tu es un expert RH. Analyse l'échange suivant (phase Comportementale) et extrais l'évaluation des compétences.
        
        HISTORIQUE ECHANGE:
        {history}
        
        Tâche: Evaluer chaque compétence comportementale abordée via la méthode STAR.
        """
        
        extractor = self.llm.with_structured_output(BehavioralOutput)
        data = extractor.invoke([SystemMessage(content=prompt)])
        
        # Calculate scores
        scores = []
        for comp in data.competences_evaluees:
            comp.score = calculate_behavioral_score(comp.competence, comp.indicateurs)
            scores.append(comp.score)

        for sjt in data.sjt_results:
            if sjt.score_choix is not None and sjt.justification_score is not None:
                sjt.score_sjt = round((sjt.score_choix * 0.6) + (sjt.justification_score * 0.4), 1)
            scores.append(sjt.score_sjt)
             
        if scores:
            data.score_comportemental_global = round(sum(scores) / len(scores), 1)
        else:
             data.score_comportemental_global = 0.0
             
        return data

    def extract_situation(self, messages: List[BaseMessage]) -> SituationOutput:
        logger.info("Extracting Situation data...")
        history = self._get_history_text(messages)
        
        prompt = f"""
        Tu es un expert technique. Analyse l'échange suivant (phase Mise en Situation) et évalue la performance du candidat.
        
        HISTORIQUE ECHANGE:
        {history}
        
        Tâche: Remplir la grille d'évaluation de la mise en situation.
        """
        
        extractor = self.llm.with_structured_output(SituationOutput)
        data = extractor.invoke([SystemMessage(content=prompt)])
        
        # Calculate score
        data.score_mise_en_situation = calculate_situation_score(data.indicateurs)
        
        return data

    def extract_simulation_report(self, 
                                  messages: List[BaseMessage], 
                                  icebreaker: IceBreakerOutput,
                                  technical: TechnicalOutput,
                                  behavioral: BehavioralOutput,
                                  situation: SituationOutput) -> SimulationReport:
        logger.info("Generating Final Simulation Report...")
        
        # We don't necessarily need the whole history if we have structured data, 
        # but the LLM might need it for "Synthese". 
        # Let's provide a summary of structured data to save tokens.
        
        context_data = {
            "icebreaker": icebreaker.dict() if icebreaker else {},
            "technical": technical.dict() if technical else {},
            "behavioral": behavioral.dict() if behavioral else {},
            "situation": situation.dict() if situation else {}
        }
        
        prompt = f"""
        Tu es un Expert Recruteur Senior. Rédige le rapport final de l'entretien basé sur les données extraites.
        
        DONNÉES STRUCTURÉES (SCORES & INDICATEURS):
        {json.dumps(context_data, ensure_ascii=False)}
        
        Tâche:
        1. Calcule le score global (Moyenne pondérée : Technique 40%, Comportemental 30%, Situation 20%, Icebreaker/Soft 10% - ou use ton jugement expert).
        2. Rédige une synthèse du candidat (2-3 phrases).
        3. Liste les points forts et faibles.
        4. Donne une recommandation claire (GO/NO GO).
        5. Rédige un feedback pour le candidat (bienveillant et constructif).
        """
        
        extractor = self.llm.with_structured_output(SimulationReport)
        report = extractor.invoke([SystemMessage(content=prompt)])
        
        # Inject the source objects back into the report (optional, as they are part of the model but null in extraction input)
        # Actually LLM might return them null or empty. We should re-attach the real objects.
        report.icebreaker = icebreaker
        report.technical = technical
        report.behavioral = behavioral
        report.situation = situation
        
        return report
