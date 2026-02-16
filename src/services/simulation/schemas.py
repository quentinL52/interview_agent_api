from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Ice Breaker Agent ---

class IceBreakerOutput(BaseModel):
    profil_type: str = Field(..., description="Type de profil (reconversion, étudiant, junior, expérimenté)")
    annees_experience_domaine: str = Field(..., description="Années d'expérience dans le domaine cible (0-2, 2-5, 5+)")
    coherence_parcours: str = Field(..., description="Cohérence parcours -> poste visé (forte, moyenne, faible)")
    motivation_detectee: bool = Field(..., description="Motivation exprimée")
    contexte_specifique: str = Field(..., description="Contexte spécifique du candidat")
    points_a_explorer: List[str] = Field(default_factory=list, description="Points à explorer dans la suite")

# --- Technical Agent ---

class TechnicalSkillGap(BaseModel):
    skill: str
    niveau_detecte: int = Field(..., description="Niveau détecté (0-4)")
    indicateurs: Dict[str, bool] = Field(..., description="Indicateurs binaires (concept_sous_jacent, experience_liee, outil_adjacent, cas_usage, strategie_montee)")
    transferabilite: str = Field(..., description="Transférabilité (faible, moyenne, forte)")
    questions_posees: List[str] = Field(default_factory=list)

class ProjectTechUnderstanding(BaseModel):
    skill: str
    source_projet: str
    score: int = Field(..., description="Score (1-5)")
    indicateurs: Dict[str, bool] = Field(..., description="Indicateurs binaires (justifie_choix, fonctionnement_interne, identifie_limites, propose_alternatives, quantifie_resultats, resolution_probleme)")
    questions_posees: List[str] = Field(default_factory=list)

class ValidatedSkill(BaseModel):
    skill: str
    score: int
    source: str

class TechnicalOutput(BaseModel):
    competences_validees: List[ValidatedSkill] = Field(default_factory=list)
    lacunes_explorees: List[TechnicalSkillGap] = Field(default_factory=list)
    comprehension_technos_projets: List[ProjectTechUnderstanding] = Field(default_factory=list)
    score_technique_global: float = Field(..., description="Score technique global calculé")
    points_a_explorer_comportemental: List[str] = Field(default_factory=list)

# --- Behavioral Agent ---

class BehavioralCompetency(BaseModel):
    competence: str
    score: int = Field(..., description="Score (1-5)")
    indicateurs: Dict[str, bool] = Field(..., description="Indicateurs binaires spécifiques à la compétence")
    questions_posees: List[str] = Field(default_factory=list)

class SJTResult(BaseModel):
    scenario_id: str
    choix: str
    score_choix: float
    justification_score: float
    score_sjt: float

class BehavioralOutput(BaseModel):
    competences_evaluees: List[BehavioralCompetency] = Field(default_factory=list)
    sjt_results: List[SJTResult] = Field(default_factory=list)
    score_comportemental_global: float = Field(..., description="Score comportemental global calculé")
    signaux_forts: List[str] = Field(default_factory=list)
    signaux_faibles: List[str] = Field(default_factory=list)
    points_a_integrer_mise_en_situation: List[str] = Field(default_factory=list)

# --- Situation Agent ---

class SituationOutput(BaseModel):
    scenario_utilise: str
    score_mise_en_situation: int = Field(..., description="Score sur 5")
    indicateurs: Dict[str, bool] = Field(..., description="Indicateurs (comprehension_probleme, demarche_structuree, pertinence_technique, gestion_contraintes, communication_solution, identification_risques, proposition_alternatives)")
    questions_posees: List[str] = Field(default_factory=list)
    observations: str

# --- Full Report ---

class SimulationReport(BaseModel):
    icebreaker: Optional[IceBreakerOutput] = None
    technical: Optional[TechnicalOutput] = None
    behavioral: Optional[BehavioralOutput] = None
    situation: Optional[SituationOutput] = None
    
    score_global: float = Field(..., description="Score global de l'entretien sur 5")
    synthese_candidat: str = Field(..., description="Synthèse textuelle du profil")
    points_forts: List[str] = Field(default_factory=list, description="Top 3 points forts")
    points_faibles: List[str] = Field(default_factory=list, description="Top 3 points faibles")
    recommandation: str = Field(..., description="GO / NO GO / A CREUSER")
    feedback_candidat: str = Field(..., description="Feedback constructif adressé au candidat")
