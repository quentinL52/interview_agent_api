Un agent conversationnel intelligent sous forme d'API REST dédié à la simulation et à l'évaluation d'entretiens. Ce service s'appuie sur l'IA pour générer des questions dynamiques, analyser en temps réel les réponses des candidats et fournir une évaluation objective de leurs compétences.

Pensé pour s'intégrer de manière fluide au sein de systèmes multi-agents ou d'architectures orientées services, ce module est l'outil idéal pour automatiser les pré-qualifications et valoriser la diversité des parcours professionnels.

## 🚀 Fonctionnalités Principales

* **Simulation Interactive** : Génération de scénarios d'entretien adaptatifs et évolutifs en fonction du poste visé et du déroulé de la conversation.
* **Évaluation Sémantique** : Analyse contextuelle des réponses pour capter avec précision les compétences techniques et les soft skills du candidat.
* **Architecture Modulaire** : Conception "API-first" facilitant l'orchestration par d'autres frameworks ou le dialogue avec d'autres agents autonomes.
* **Conteneurisation Complète** : Déploiement standardisé, isolée et reproductible via Docker, idéal pour l'auto-hébergement et l'intégration sur VPS.

## 🛠️ Stack Technique

* **Langage** : Python 3.x
* **Déploiement** : Docker (Dockerfile inclus)
* **Format d'échange** : JSON

## ⚙️ Prérequis et Configuration

Le service nécessite la configuration de variables d'environnement (ex: clés d'API pour les LLM) pour fonctionner correctement.

1. **Cloner le dépôt :**
```bash
   git clone [https://github.com/quentinL52/interview_agent_api.git](https://github.com/quentinL52/interview_agent_api.git)
   cd interview_agent_api
