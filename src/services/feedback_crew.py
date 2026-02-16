"""Feedback Crew for interview analysis using CrewAI."""

import json
import logging
from typing import Dict, Any, List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task, crew

from src.config import crew_openai
from src.schemas.feedback_schemas import FeedbackOutput

logger = logging.getLogger(__name__)


@CrewBase
class FeedbackCrew:
    """Crew for analyzing interview conversations and generating structured feedback."""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    def __init__(self, job_offer: Dict[str, Any], cv_content: str, conversation_history: List[Dict[str, Any]], cheat_metrics: Dict[str, Any] = None, gap_analysis: Dict[str, Any] = None):
        self.job_offer = json.dumps(job_offer, ensure_ascii=False)
        self.cv_content = cv_content
        self.conversation_history = json.dumps(conversation_history, ensure_ascii=False)
        self.cheat_metrics = json.dumps(cheat_metrics or {}, ensure_ascii=False)
        self.gap_analysis = json.dumps(gap_analysis or {}, ensure_ascii=False)
        self._llm = crew_openai()
    
    # --- Agents ---
    
    @agent
    def consistency_analyst(self) -> Agent:
        return Agent(config=self.agents_config["consistency_analyst"], llm=self._llm)
    
    @agent
    def search_analyst(self) -> Agent:
        return Agent(config=self.agents_config["search_analyst"], llm=self._llm)

    @agent
    def tech_expert(self) -> Agent:
        return Agent(config=self.agents_config["tech_expert"], llm=self._llm)
    
    @agent
    def business_evaluator(self) -> Agent:
        return Agent(config=self.agents_config["business_evaluator"], llm=self._llm)
    
    @agent
    def final_reporter(self) -> Agent:
        return Agent(config=self.agents_config["final_reporter"], llm=self._llm)
    
    # --- Tasks ---
    
    @task
    def consistency_task(self) -> Task:
        return Task(config=self.tasks_config["consistency_task"])
    
    @task
    def search_task(self) -> Task:
        return Task(config=self.tasks_config["search_task"])

    @task
    def tech_task(self) -> Task:
        return Task(config=self.tasks_config["tech_task"])
    
    @task
    def business_task(self) -> Task:
        return Task(config=self.tasks_config["business_task"])
    
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],
            output_pydantic=FeedbackOutput
        )
    
    # --- Crew ---
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
    
    # --- Public API ---
    
    def run(self) -> Dict[str, Any]:
        """Execute the feedback crew and return structured output."""
        logger.info("Starting Feedback Crew analysis...")
        
        # Safe extraction of job_type from gap_analysis string or dict
        job_type = "GENERAL_TECH"
        try:
            ga_data = json.loads(self.gap_analysis) if isinstance(self.gap_analysis, str) else self.gap_analysis
            job_type = ga_data.get("job_type", "GENERAL_TECH")
        except:
            pass

        inputs = {
            "job_offer": self.job_offer,
            "cv_content": self.cv_content,
            "conversation_history": self.conversation_history,
            "cheat_metrics": self.cheat_metrics,
            "gap_analysis": self.gap_analysis,
            "job_type": job_type
        }
        
        result = self.crew().kickoff(inputs=inputs)
        
        # Handle structured output
        if result.pydantic:
            return result.pydantic.model_dump()
        elif result.json_dict:
            return result.json_dict
        else:
            logger.warning("No structured output, parsing raw result")
            try:
                raw_str = str(result.raw)
                start = raw_str.find('{')
                end = raw_str.rfind('}') + 1
                if start != -1 and end > start:
                    return json.loads(raw_str[start:end])
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
            
            return {"error": "Could not parse output", "raw": str(result.raw)}
