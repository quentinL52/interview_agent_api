
import pytest
from src.services.search_service import SearchService

class TestSearchService:
    def setup_method(self):
        self.service = SearchService()

    def test_detect_job_type(self):
        assert self.service._detect_job_type("Cherche Senior Data Scientist NLP") == "DATA_SCIENTIST"
        assert self.service._detect_job_type("Offre Data Analyst PowerBI") == "DATA_ANALYST"
        assert self.service._detect_job_type("Besoin d'un ML Engineer pour la prod") == "DATA_ENGINEER"
        assert self.service._detect_job_type("Développeur Python Backend") == "GENERAL_TECH"

    def test_extract_action_verbs_found(self):
        text = "J'ai pu optimiser les performances et déployer le modèle en production."
        verbs = self.service._extract_action_verbs(text, self.service.VERB_MAPPINGS["DATA_ENGINEER"])
        assert "optimiser" in verbs
        assert "déployer" in verbs
        assert len(verbs) >= 2

    def test_extract_action_verbs_analyst(self):
        text = "J'ai réalisé des visualisations et présenté des insights."
        verbs = self.service._extract_action_verbs(text, self.service.VERB_MAPPINGS["DATA_ANALYST"])
        assert "visualiser" in verbs or "présenter" in verbs

    def test_extract_action_verbs_none(self):
        text = "J'ai fait de la gestion de projet et de la rédaction."
        verbs = self.service._extract_action_verbs(text, self.service.VERB_MAPPINGS["DATA_ENGINEER"])
        assert len(verbs) == 0

    def test_detect_reconversion_true(self):
        cv_text = "Après un bootcamp intensif en Data Science, je cherche..."
        job_desc = "Cherche Data Scientist junior."
        is_reconversion, reason = self.service._detect_reconversion(cv_text, job_desc)
        assert is_reconversion is True
        assert "bootcamp" in reason

    def test_detect_reconversion_false(self):
        cv_text = "Ingénieur diplomé avec 5 ans d'expérience."
        job_desc = "Cherche Senior Dev."
        is_reconversion, reason = self.service._detect_reconversion(cv_text, job_desc)
        assert is_reconversion is False
        assert "Parcours classique" in reason

    def test_analyze_gap_structure(self):
        cv_text = "Développeur Python avec expérience en optimiser et monitorer."
        job_desc = "Cherche expert Python pour optimiser le backend."
        
        result = self.service.analyze_gap(cv_text, job_desc)
        
        assert "semantic_score" in result
        assert "production_mindset_score" in result
        assert "is_reconversion" in result
        assert "production_verbs_found" in result
        assert "optimiser" in result["production_verbs_found"]
