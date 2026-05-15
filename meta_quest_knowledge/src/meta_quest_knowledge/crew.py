from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# Knowledge sources
pdf_source = PDFKnowledgeSource(
    file_paths=["Veeramalla_Observability_Platform_Documentation.pdf"]
)

@CrewBase
class MetaQuestKnowledge():
    """MetaQuestKnowledge crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def meta_quest_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['meta_quest_expert'],
            verbose=True
        )

    @task
    def answer_question_task(self) -> Task:
        return Task(
            config=self.tasks_config['answer_question_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MetaQuestKnowledge crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[pdf_source],
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "tinyllama",
                    "base_url": "http://localhost:11434"
                }
            }
        )
