import os
import logging
import json
from pathlib import Path
from typing import TypedDict, Annotated, Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langtrace_python_sdk import langtrace

from src.services.simulation.agents import InterviewAgentExtractor
from src.services.simulation.schemas import (
    IceBreakerOutput, TechnicalOutput, BehavioralOutput, SituationOutput, SimulationReport
)

langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Number of questions per agent
QUESTIONS_PER_AGENT = {
    "icebreaker": 2,
    "auditeur": 3,
    "enqueteur": 2,
    "stratege": 2,
    "projecteur": 1  
}

AGENT_ORDER = ["icebreaker", "auditeur", "enqueteur", "stratege", "projecteur"]


EXIT_MESSAGE = """L'entretien est maintenant terminé.

Merci pour cet échange. Je finalise l'analyse de votre candidature, votre rapport sera disponible dans quelques instants."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    job_offer_id: str
    cv_data: Dict[str, Any]
    job_data: Dict[str, Any]
    section: str
    turn_count: int
    context: Dict[str, Any]
    cheat_metrics: Dict[str, Any]
    
    # Structured Data
    icebreaker_data: Optional[IceBreakerOutput]
    technical_data: Optional[TechnicalOutput]
    behavioral_data: Optional[BehavioralOutput]
    situation_data: Optional[SituationOutput]
    simulation_report: Optional[SimulationReport]


class GraphInterviewProcessor:
    """Strict interview simulation with per-agent question limits."""
    
    def __init__(self, payload: Dict[str, Any]):
        logging.info("Initialisation de RONI GraphInterviewProcessor...")
        
        self.user_id = payload["user_id"]
        self.job_offer_id = payload["job_offer_id"]
        self.job_offer = payload["job_offer"]
        self.cv_data = payload.get("cv_document", {}).get('candidat', {})

        if not self.cv_data:
            raise ValueError("Données du candidat non trouvées.")

        self.prompts = self._load_all_prompts()
        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.7)
        self.extractor = InterviewAgentExtractor(self.llm)
        
        self.graph = self._build_graph()
        logging.info("RONI Graph initialisé.")

    def _load_all_prompts(self) -> Dict[str, str]:
        prompts = {}
        prompt_files = {
            "icebreaker": PROMPTS_DIR / "agent_icebreaker.txt",
            "auditeur": PROMPTS_DIR / "agent_auditeur.txt",
            "enqueteur": PROMPTS_DIR / "agent_enqueteur.txt",
            "stratege": PROMPTS_DIR / "agent_stratege.txt",
            "projecteur": PROMPTS_DIR / "agent_projecteur.txt"
        }
        for key, path in prompt_files.items():
            try:
                prompts[key] = path.read_text(encoding='utf-8-sig')
            except FileNotFoundError:
                logger.error(f"Missing prompt: {path}")
                prompts[key] = f"You are {key}."
            except UnicodeDecodeError:
                logger.warning(f"Encoding issue with {path}, trying latin-1")
                prompts[key] = path.read_text(encoding='latin-1')
        return prompts

    def _count_user_messages(self, messages: List[BaseMessage]) -> int:
        """Count user messages from conversation history."""
        return sum(1 for m in messages if isinstance(m, HumanMessage))

    def _determine_section_and_status(self, user_msg_count: int) -> tuple[str, bool]:
        """
        Determine current section based on user message count.
        Returns: (section_name, should_end)
        """
        cumulative = 0
        
        for agent in AGENT_ORDER:
            agent_questions = QUESTIONS_PER_AGENT[agent]
            
            # Check if user is still within this agent's range
            if user_msg_count < cumulative + agent_questions:
                return agent, False
            
            cumulative += agent_questions
        
        # Past all agents -> END (projecteur completed)
        return "end", True

    # --- Context builders ---
    
    def _get_icebreaker_context(self, state: AgentState) -> str:
        candidat = state["cv_data"].get("info_personnelle", {})
        job = state["job_data"]
        
        # Extract Hobbies/Interests if available
        hobbies = ", ".join(state["cv_data"].get("centres_interet", []))
        if not hobbies:
            hobbies = "Non spécifiés"
            
        # Extract Reconversion Context
        reconversion_data = state["cv_data"].get("reconversion", {})
        is_reco = reconversion_data.get("is_reconversion", False)
        reco_context = "OUI" if is_reco else "NON"
        
        # Extract Student Context
        etudiant_data = state["cv_data"].get("etudiant", {})
        is_etudiant = etudiant_data.get("is_etudiant", False)
        etudiant_context = f"OUI ({etudiant_data.get('niveau_etudes', '')})" if is_etudiant else "NON"
        
        return f"""
        === CONTEXTE CANDIDAT ===
        NOM: {candidat.get('nom', 'Candidat')} {candidat.get('first_name', '')}
        POSTE VISÉ: {job.get('poste', 'Non spécifié')}
        ENTREPRISE: {job.get('entreprise', 'Non spécifié')}
        
        === DOSSIER PERSONNEL ===
        CENTRES D'INTÉRÊT: {hobbies}
        EN RECONVERSION ?: {reco_context}
        ÉTUDIANT ?: {etudiant_context}
        """

    def _get_auditeur_context(self, state: AgentState) -> str:
        cv = state["cv_data"]
        job = state["job_data"]
        experiences = json.dumps(cv.get("expériences", []), ensure_ascii=False)
        
        # Format Projects with details
        projets_list = cv.get("projets", {}).get("professional", []) + cv.get("projets", {}).get("personal", [])
        projets_formatted = []
        for p in projets_list:
            techs = ", ".join(p.get("technologies", []))
            outcomes = ", ".join(p.get("outcomes", []))
            projets_formatted.append(f"- {p.get('title')}: {techs} (Résultats: {outcomes})")
        projets_str = "\n".join(projets_formatted)
        
        # Format Skills with Context
        skills_ctx = cv.get("compétences", {}).get("skills_with_context", [])
        skills_formatted = [f"{s.get('skill')} ({s.get('context')})" for s in skills_ctx]
        skills_str = ", ".join(skills_formatted)
        
        # Fallback to hard_skills if no context
        if not skills_str:
            skills_str = ", ".join(cv.get("compétences", {}).get("hard_skills", []))
        
        # Add Icebreaker Data
        ib_data = state.get("icebreaker_data")
        ib_context = ""
        if ib_data:
            ib_context = f"""
            === PROFIL DÉTECTÉ (ICEBREAKER) ===
            TYPE: {ib_data.profil_type}
            EXPÉRIENCE DOMAINE: {ib_data.annees_experience_domaine}
            MOTIVATION: {"OUI" if ib_data.motivation_detectee else "NON"}
            CONTEXTE: {ib_data.contexte_specifique}
            """
            
        return f"""
        === CONTEXTE TECHNIQUE ===
        MISSION DU POSTE: {job.get('mission', '')}
        EXPÉRIENCES CANDIDAT: {experiences}
        PROJETS SIGNIFICATIFS:
        {projets_str}
        
        COMPÉTENCES ET CONTEXTE: 
        {skills_str}
        {ib_context}
        """

    def _get_enqueteur_context(self, state: AgentState) -> str:
        cv = state["cv_data"]
        soft_skills = ", ".join(cv.get("compétences", {}).get("soft_skills", []))
        reconversion = cv.get("reconversion", {})
        is_reco = reconversion.get("is_reconversion", False)
        reco_txt = "OUI" if is_reco else "NON"
        
        # Add Technical Data (Gaps)
        tech_data = state.get("technical_data")
        tech_context = ""
        if tech_data:
            lacunes = [f"- {l.skill} (Niveau {l.niveau_detecte})" for l in tech_data.lacunes_explorees]
            tech_context = f"""
            === BILAN TECHNIQUE ===
            SCORE GLOBAL: {tech_data.score_technique_global}/5
            LACUNES IDENTIFIÉES:
            {chr(10).join(lacunes)}
            """

        return f"""
        === CONTEXTE COMPORTEMENTAL ===
        SOFT SKILLS: {soft_skills}
        {reco_txt}
        {tech_context}
        """

    def _get_stratege_context(self, state: AgentState) -> str:
        job = state["job_data"]
        
        # Add Behavioral Data
        beh_data = state.get("behavioral_data")
        beh_context = ""
        if beh_data:
            beh_context = f"""
            === BILAN COMPORTEMENTAL ===
            SCORE GLOBAL: {beh_data.score_comportemental_global}/5
            POINTS À INTÉGRER: {", ".join(beh_data.points_a_integrer_mise_en_situation)}
            """
            
        return f"""
        === CONTEXTE SJT (MISE EN SITUATION) ===
        MISSION: {job.get('mission', '')}
        CULTURE/VALEURS: {job.get('profil_recherche', '')}
        {beh_context}
        """

    def _get_projecteur_context(self, state: AgentState) -> str:
        job = state["job_data"]
        return f"""
        === CONTEXTE PROJECTION ===
        ENTREPRISE: {job.get('entreprise', '')}
        DESCRIPTION POSTE: {job.get('description_poste', '')}
        
        NOTE: C'est la dernière question de l'entretien.
        """

    # --- Orchestrator (strict per-agent logic) ---
    
    def _orchestrator_node(self, state: AgentState):
        messages = state["messages"]
        user_msg_count = self._count_user_messages(messages)
        
        section, should_end = self._determine_section_and_status(user_msg_count)
        
        logger.info(f"Orchestrator: {user_msg_count} user messages -> section={section}, end={should_end}")
        
        context_updates = state.get("context", {}).copy()
        next_dest = "end_interview" if should_end else section
        context_updates["next_dest"] = next_dest
        
        # Trigger Extraction based on transitions
        extract_target = None
        if section == "auditeur" and not state.get("icebreaker_data"):
            extract_target = "icebreaker"
        elif section == "enqueteur" and not state.get("technical_data"):
             extract_target = "technical"
        elif section == "stratege" and not state.get("behavioral_data"):
             extract_target = "behavioral"
        elif section == "projecteur" and not state.get("situation_data"):
             extract_target = "situation"
             
        # Check for Final Report extraction
        if should_end:
            # Check if we missed situation data (if flow ended abruptly?) - assuming linear flow for now.
            # Only trigger if we haven't tried yet (flag in context)
            if not state.get("situation_data") and not context_updates.get("situation_attempted"):
                 extract_target = "situation"
            elif not state.get("simulation_report") and not context_updates.get("report_attempted"):
                 extract_target = "report"
        
        if extract_target:
            context_updates["extract_target"] = extract_target
            logger.info(f"Triggering extraction for: {extract_target}")
        
        return {"section": section, "context": context_updates}

    def _extraction_node(self, state: AgentState):
        target = state["context"].get("extract_target")
        logger.info(f"Running extraction node for: {target}")
        updates = {}
        ctx = state["context"].copy()
        
        try:
            if target == "icebreaker":
                updates["icebreaker_data"] = self.extractor.extract_icebreaker(state["messages"], state["cv_data"])
            elif target == "technical":
                updates["technical_data"] = self.extractor.extract_technical(state["messages"], state["job_data"])
            elif target == "behavioral":
                updates["behavioral_data"] = self.extractor.extract_behavioral(state["messages"])
            elif target == "situation":
                updates["situation_data"] = self.extractor.extract_situation(state["messages"])
            elif target == "report":
                updates["simulation_report"] = self.extractor.extract_simulation_report(
                    state["messages"],
                    state.get("icebreaker_data"),
                    state.get("technical_data"),
                    state.get("behavioral_data"),
                    state.get("situation_data")
                )
        except Exception as e:
            logger.error(f"Extraction failed for {target}: {e}", exc_info=True)
        
        # Mark attempt to avoid infinite loops
        if target == "situation":
            ctx["situation_attempted"] = True
        elif target == "report":
            ctx["report_attempted"] = True
        
        # Clear extract_target safely
        ctx.pop("extract_target", None)
            
        return {**updates, "context": ctx}

    # --- Agent runner ---
    
    def _run_agent(self, state: AgentState, agent_key: str, context_str: str):
        prompt_template = self.prompts[agent_key]
        
        # Extract first name
        cv_info = state["cv_data"].get("info_personnelle", {})
        prenom = cv_info.get("prenom", "")
        
        # Get question limit
        nb_questions = QUESTIONS_PER_AGENT.get(agent_key, 2)

        try:
            instructions = prompt_template.format(
                user_id=state["user_id"],
                prenom=prenom,
                nb_questions=nb_questions,
                job_description=json.dumps(state["job_data"], ensure_ascii=False),
                poste=state["job_data"].get("poste", "Poste non spécifié"),
                entreprise=state["job_data"].get("entreprise", "Entreprise confidentielle")
            )
        except KeyError:
            instructions = prompt_template
        
        system_msg = f"{instructions}\n\n{context_str}"
        messages = [SystemMessage(content=system_msg)] + list(state["messages"])
        response = self.llm.invoke(messages)
        
        return {"messages": [response]}

    # --- Agent nodes ---
    
    def _icebreaker_node(self, s): return self._run_agent(s, "icebreaker", self._get_icebreaker_context(s))
    def _auditeur_node(self, s): return self._run_agent(s, "auditeur", self._get_auditeur_context(s))
    def _enqueteur_node(self, s): return self._run_agent(s, "enqueteur", self._get_enqueteur_context(s))
    def _stratege_node(self, s): return self._run_agent(s, "stratege", self._get_stratege_context(s))
    def _projecteur_node(self, s): return self._run_agent(s, "projecteur", self._get_projecteur_context(s))

    # --- Final analysis ---
    
    def _final_analysis_node(self, state: AgentState):
        """Trigger background analysis and return exit message."""
        from src.tasks import run_analysis_task
        
        logger.info("=== FINAL ANALYSIS TRIGGERED ===")
        
        try:
            hist = [{"role": ("user" if isinstance(m, HumanMessage) else "assistant"), "content": m.content} for m in state["messages"]]
            
            simulation_report_dict = None
            if state.get("simulation_report"):
                simulation_report_dict = state["simulation_report"].dict()
            
            run_analysis_task.delay(
                user_id=state['user_id'],
                job_offer_id=state['job_offer_id'],
                job_description=json.dumps(state['job_data'], ensure_ascii=False),
                conversation_history=hist,
                cv_content=json.dumps(state['cv_data'], ensure_ascii=False),
                cheat_metrics=state.get('cheat_metrics', {}),
                simulation_report=simulation_report_dict
            )
            logger.info("Background analysis task enqueued via Celery")
        except Exception as e:
            logger.error(f"Failed to enqueue analysis task: {e}")
        
        # Generate contextual exit message
        last_user_msg = state["messages"][-1].content if state["messages"] and isinstance(state["messages"][-1], HumanMessage) else ""
        
        if last_user_msg:
            closing_prompt = (
                f"Tu es un recruteur professionnel (RONI). L'entretien est terminé.\n"
                f"Le candidat vient de dire : \"{last_user_msg}\"\n"
                f"Réponds-y brièvement et aimablement (une phrase max). Si c'est une question, donnes-y une réponse simple.\n"
                f"Ne dis PAS au revoir tout de suite, contente-toi de répondre au dernier point soulevé."
            )
            try:
                ai_response = self.llm.invoke([SystemMessage(content=closing_prompt)])
                # Force append the exit message to guarantee trigger detection
                final_content = f"{ai_response.content}\n\n{EXIT_MESSAGE}"
            except Exception as e:
                logger.error(f"Error generating closing message: {e}")
                final_content = EXIT_MESSAGE
        else:
            final_content = EXIT_MESSAGE

        return {"messages": [AIMessage(content=final_content)], "context": {"next_dest": "end_interview"}}

    # --- Routing ---
    
    def _route_next_step(self, state: AgentState) -> str:
        # Check if extraction is needed
        if state.get("context", {}).get("extract_target"):
            return "extraction_node"
            
        dest = state.get("context", {}).get("next_dest", "icebreaker")
        if dest == "end_interview":
            # Ensure we have the report before finishing. Loop back to orchestrator.
            # Only loop back if we haven't attempted to extract the report yet.
            if not state.get("simulation_report") and not state.get("context", {}).get("report_attempted"):
                 return "orchestrator"
            return "final_analysis"
        return f"{dest}_agent"

    # --- Graph builder ---
    
    def _build_graph(self) -> any:
        graph = StateGraph(AgentState)
        
        graph.add_node("orchestrator", self._orchestrator_node)
        graph.add_node("extraction_node", self._extraction_node)  # Add extraction node
        graph.add_node("icebreaker_agent", self._icebreaker_node)
        graph.add_node("auditeur_agent", self._auditeur_node)
        graph.add_node("enqueteur_agent", self._enqueteur_node)
        graph.add_node("stratege_agent", self._stratege_node)
        graph.add_node("projecteur_agent", self._projecteur_node)
        graph.add_node("final_analysis", self._final_analysis_node)
        
        graph.set_entry_point("orchestrator")
        
        routing_map = {
            "extraction_node": "extraction_node",
            "icebreaker_agent": "icebreaker_agent",
            "auditeur_agent": "auditeur_agent",
            "enqueteur_agent": "enqueteur_agent",
            "stratege_agent": "stratege_agent",
            "projecteur_agent": "projecteur_agent",
            "final_analysis": "final_analysis",
            "orchestrator": "orchestrator"  # Added loopback
        }
        
        graph.add_conditional_edges("orchestrator", self._route_next_step, routing_map)
        graph.add_conditional_edges("extraction_node", self._route_next_step, routing_map)
        
        for node in ["icebreaker_agent", "auditeur_agent", "enqueteur_agent", "stratege_agent", "projecteur_agent", "final_analysis"]:
            graph.add_edge(node, END)
            
        return graph.compile()

    # --- Public API ---
    
    def invoke(self, messages: List[Dict[str, Any]], cheat_metrics: Dict[str, Any] = None):
        langchain_messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in messages]
        
        if not langchain_messages:
            langchain_messages.append(HumanMessage(content="Bonjour"))
        
        initial_state = {
            "user_id": self.user_id,
            "job_offer_id": self.job_offer_id,
            "messages": langchain_messages,
            "cv_data": self.cv_data,
            "job_data": self.job_offer,
            "section": "icebreaker",
            "turn_count": 0,
            "context": {},
            "cheat_metrics": cheat_metrics or {}
        }
        
        final_state = self.graph.invoke(initial_state)
        
        if not final_state or not final_state['messages']:
            return {"response": "Erreur système.", "status": "finished"}
            
        last_msg = final_state['messages'][-1]
        is_finished = final_state.get("context", {}).get("next_dest") == "end_interview"
        status = "interview_finished" if is_finished else "interviewing"
        
        return {"response": last_msg.content, "status": status}