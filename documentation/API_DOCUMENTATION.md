# Interview Simulation API Documentation

Cette documentation détaille l'utilisation de l'API de simulation d'entretiens, motorisée par LangGraph et FastAPI.

## Introduction

L'**Interview Simulation API** permet de mener des simulations d'entretiens d'embauche réalistes. Elle utilise un système d'agents spécialisés (Icebreaker, Auditeur, Enquêteur, Stratège, Projecteur) pour évaluer un candidat par rapport à une offre d'emploi spécifique.

**Base URL:** `http://localhost:7860` (ou URL de déploiement)  
**Version:** `1.0.0`

## Configuration & Authentification

L'API utilise des variables d'environnement pour la configuration :
- `OPENAI_API_KEY`: Requis pour les agents LLM.
- `LANGTRACE_API_KEY`: Pour l'observabilité (Langtrace).
- `CORS_ORIGINS`: Liste des origines autorisées (défaut: `*`).

## Endpoints

### 1. Health Check
Vérifie que l'API est opérationnelle.

- **URL:** `/`
- **Method:** `GET`
- **Success Response (200 OK):**
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Simulate Interview
Déclenche ou poursuit une simulation d'entretien.

- **URL:** `/simulate-interview/`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "user_id": "string",          // ID unique de l'utilisateur
    "job_offer_id": "string",     // ID de l'offre d'emploi
    "cv_document": {              // Données extraites du CV
      "candidat": { ... }         // Voir section "Data Models"
    },
    "job_offer": {                // Détails de l'offre
      "poste": "string",
      "entreprise": "string",
      "mission": "string",
      ...
    },
    "messages": [                 // Historique de la conversation (optionnel)
      {
        "role": "user",
        "content": "Bonjour"
      },
      ...
    ],
    "cheat_metrics": { ... }      // Métriques anti-triche (optionnel)
  }
  ```

- **Success Response (200 OK):**
  ```json
  {
    "response": "Texte généré par l'agent IA",
    "status": "interviewing" | "interview_finished"
  }
  ```

- **Error Responses:**
  - `400 Bad Request`: Payload incomplet ou données invalides.
  - `500 Internal Server Error`: Erreur lors de l'exécution du graph LangGraph.

## Data Models (Schemas)

### Feedback Output (Output final de l'analyse)
Une fois l'entretien terminé, un rapport complet est généré (via Celery) suivant cette structure :

- **CandidatFeedback**: Points forts, axes d'amélioration, conseils, score global.
- **EntrepriseInsights**:
    - **Dashboard**: Technique, Cognitive, Comportementale (0-100).
    - **Decision**: RECRUTER, APPROFONDIR ou REJETER.
    - **Fraud Detection**: Score global d'usage d'IA, mots-clés détectés, alertes (red flags).

## Exemples d'Usage

### Exemple cURL
```bash
curl -X POST http://localhost:7860/simulate-interview/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "job_offer_id": "job_456",
    "cv_document": { "candidat": { "first_name": "Jean", "expériences": [] } },
    "job_offer": { "poste": "Développeur Python", "entreprise": "TechCorp" },
    "messages": []
  }'
```

### Exemple Python (Requests)
```python
import requests

url = "http://localhost:7860/simulate-interview/"
payload = {
    "user_id": "user_123",
    "job_offer_id": "job_456",
    "cv_document": { "candidat": { "first_name": "Jean", "expériences": [] } },
    "job_offer": { "poste": "Développeur Python", "entreprise": "TechCorp" },
    "messages": [{"role": "user", "content": "Je suis prêt."}]
}

response = requests.post(url, json=payload)
print(response.json())
```

## Error Handling

Toutes les erreurs renvoient un format JSON standardisé :
```json
{
  "error": "Description de l'erreur"
}
```

