"""Pydantic schemas for interview feedback output."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidatFeedback(BaseModel):
    """Feedback section for the candidate."""
    points_forts: List[str] = Field(..., description="Liste des points forts démontrés")
    points_a_ameliorer: List[str] = Field(..., description="Liste des axes d'amélioration identifiés")
    conseils_apprentissage: List[str] = Field(..., description="Ressources ou actions concrètes pour progresser")
    score_global: int = Field(..., description="Note globale sur 100")
    feedback_constructif: str = Field(..., description="Message bienveillant et constructif adressé au candidat")


class MetricsBreakdown(BaseModel):
    """Detailed breakdown of scores."""
    communication: float = Field(..., description="Score 0-10: Orthographe, grammaire, clarté")
    technique: float = Field(..., description="Score 0-10: Compétence technique (SOAR)")
    comportemental: float = Field(..., description="Score 0-10: Soft skills (STAR)")
    fidelite_cv: float = Field(..., description="Score 0-10: Cohérence avec le CV")


class FraudDetectionMetrics(BaseModel):
    """Detailed fraud detection metrics (0-100)."""
    vocab_score: int = Field(..., description="Usage de mots 'GPT' (delve, tapestry...)")
    structure_score: int = Field(..., description="Patterns structurels (listes, longueur...)")
    paste_score: int = Field(..., description="Basé sur l'événement copier-coller")
    pattern_score: int = Field(..., description="Répétitions, questions en retour, proximité CV")

class FraudDetection(BaseModel):
    """Complete fraud analysis."""
    score_global: int = Field(..., description="Probabilité globale d'usage IA (0-100)")
    red_flag: bool = Field(False, description="Vrai si score > 70")
    detected_keywords: List[str] = Field(..., description="Mots suspects identifiés")
    resume: str = Field(..., description="Eplication courte")
    details: FraudDetectionMetrics

class DashboardCompetences(BaseModel):
    """Score de similarité basé sur la triade."""
    technique: float = Field(..., description="Adéquation Hard Skills (0-100)")
    cognitive: float = Field(..., description="Raisonnement et structure (0-100)")
    comportementale: float = Field(..., description="Soft Skills & Culture Fit (0-100)")
    average_score: float = Field(..., description="Moyenne pondérée")

class DecisionStrategique(BaseModel):
    """Pilier décisionnel du rapport."""
    recommendation: str = Field(..., description="RECRUTER, APPROFONDIR ou REJETER")
    action_plan: str = Field(..., description="Recommandation d'action immédiate (ex: Tester sur SQL)")
    so_what: str = Field(..., description="Impact business direct du candidat (Le 'Pourquoi' stratégique)")

class RolsPcdAnalysis(BaseModel):
    """Analyse structurée ROLS et PCD."""
    rols_summary: str = Field(..., description="Résumé ROLS (Résumé, Objectifs, Localisation, Stratégie)")
    pcd_analysis: str = Field(..., description="Analyse PCD (Produits, Clients, Distribution)")

class EntrepriseInsights(BaseModel):
    """Insights section for the recruiter/company."""
    correspondance_profil_offre: str = Field(..., description="Analyse de l'adéquation CV/Poste")
    
    # Dashboard
    dashboard: DashboardCompetences = Field(..., description="Tableau de bord des compétences")
    
    # Qualitative Analysis
    qualitative_analysis: RolsPcdAnalysis = Field(..., description="Analyse structurée ROLS/PCD")
    
    # Strategic Decision
    decision: DecisionStrategique = Field(..., description="Aide à la décision")

    # Fraud & Cheat Detection
    fraud_detection: FraudDetection = Field(..., description="Analyse complète anti-triche")
    
    # Legacy Metrics (kept for backward compatibility if needed, or mapped from dashboard)
    metrics: MetricsBreakdown = Field(..., description="Détail des notes par catégorie")
    
    # Global verification
    red_flag_detected: bool = Field(False, description="True si un comportement inacceptable est détecté")
    red_flag_reason: Optional[str] = Field(None, description="Raison du Veto si applicable")


class FeedbackOutput(BaseModel):
    """Complete feedback output with candidate and company sections."""
    candidat: CandidatFeedback
    entreprise: EntrepriseInsights
