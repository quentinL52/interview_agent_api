from typing import Dict

def calculate_technical_gap_score(indicators: Dict[str, bool]) -> int:
    """
    Calcule le niveau (0-4) pour une lacune technique basée sur les indicateurs binaires.
    
    Poids:
    - concept_sous_jacent: 2
    - experience_liee: 1
    - outil_adjacent: 1
    - cas_usage: 1
    - strategie_montee: 1
    
    Mapping:
    - 0 pts -> Niveau 0
    - 1-2 pts -> Niveau 1
    - 3 pts -> Niveau 2
    - 4 pts -> Niveau 3
    - 5-6 pts -> Niveau 4
    """
    weights = {
        "concept_sous_jacent": 2,
        "experience_liee": 1,
        "outil_adjacent": 1,
        "cas_usage": 1,
        "strategie_montee": 1
    }
    
    score = sum(weights.get(k, 0) for k, v in indicators.items() if v)

    if score == 0: return 0
    if score <= 2: return 1
    if score == 3: return 2
    if score == 4: return 3
    return 4

def calculate_project_tech_understanding_score(indicators: Dict[str, bool]) -> int:
    """
    Calcule le score (1-5) pour la compréhension d'une technologie projet.
    
    Poids:
    - justifie_choix: 1
    - fonctionnement_interne: 3
    - identifie_limites: 2
    - propose_alternatives: 2
    - quantifie_resultats: 1
    - resolution_probleme: 2
    
    Total max: 11
    
    Mapping:
    - 0-2 -> 1/5 (usage superficiel)
    - 3-4 -> 2/5 (usage fonctionnel)
    - 5-6 -> 3/5 (compréhension partielle)
    - 7-9 -> 4/5 (bonne compréhension)
    - 10-11 -> 5/5 (maîtrise démontrée)
    """
    weights = {
        "justifie_choix": 1,
        "fonctionnement_interne": 3,
        "identifie_limites": 2,
        "propose_alternatives": 2,
        "quantifie_resultats": 1,
        "resolution_probleme": 2
    }
    
    score = sum(weights.get(k, 0) for k, v in indicators.items() if v)

    if score <= 2: return 1
    if score <= 4: return 2
    if score <= 6: return 3
    if score <= 9: return 4
    return 5

def calculate_behavioral_score(competence: str, indicators: Dict[str, bool]) -> int:
    """
    Calcule le score (1-5) pour une compétence comportementale.
    Utilise une pondération spécifique par compétence si définie, sinon une générique.
    Mapping proportionnel au max possible.
    """
    # Exemples de poids par défaut si non spécifiés dans le code appelant
    # Ici on suppose que les indicateurs passés correspondent à ceux attendus pour la compétence
    # On va utiliser une heuristique simple: 1 point par indicateur si pas de poids spécifique,
    # ou on code en dur les poids des exemples connus.
    
    # Poids par défaut pour les exemples connus
    weights_map = {
        "Adaptabilité": {
            "situation_changement": 2,
            "actions_concretes": 2,
            "apprentissage_ajustement": 2,
            "resultat_quantifie": 1,
            "limites_reconnues": 1,
            "transfert_competence": 2
        },
        "Apprentissage autonome": {
            "demarche_auto_formation": 2, # Mapping approximatif des clés JSON
            "ressources_specifiques": 1,
            "progression_mesurable": 2,
            "application_concrete": 2,
            "identifie_reste_a_apprendre": 1
        }
    }

    # Si la compétence est connue, on utilise ses poids, sinon 1.
    competence_weights = weights_map.get(competence, {})
    
    total_score = 0
    max_score = 0
    
    for key, val in indicators.items():
        weight = competence_weights.get(key, 1) # Default weight 1 if unknown key
        if val:
            total_score += weight
        max_score += weight
        
    if max_score == 0:
        return 3  # Score neutre si aucun indicateur defini

    ratio = total_score / max_score
    
    # Mapping linéaire approximatif vers 1-5
    if ratio <= 0.2: return 1
    if ratio <= 0.4: return 2
    if ratio <= 0.6: return 3
    if ratio <= 0.8: return 4
    return 5

def calculate_situation_score(indicators: Dict[str, bool]) -> int:
    """
    Calcule le score (1-5) pour la mise en situation.
    
    Poids:
    - comprehension_probleme: 2
    - demarche_structuree: 3
    - pertinence_technique: 3
    - gestion_contraintes: 2
    - communication_solution: 1
    - identification_risques: 2
    - proposition_alternatives: 1
    
    Total max: 14
    
    Mapping:
    - 0-3 -> 1/5
    - 4-6 -> 2/5
    - 7-9 -> 3/5
    - 10-12 -> 4/5
    - 13-14 -> 5/5
    """
    weights = {
        "comprehension_probleme": 2,
        "demarche_structuree": 3,
        "pertinence_technique": 3,
        "gestion_contraintes": 2,
        "communication_solution": 1,
        "identification_risques": 2,
        "proposition_alternatives": 1
    }
    
    score = sum(weights.get(k, 0) for k, v in indicators.items() if v)
    
    if score <= 3: return 1
    if score <= 6: return 2
    if score <= 9: return 3
    if score <= 12: return 4
    return 5
