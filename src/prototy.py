from crewai import Agent, Crew, Task, Process, LLM 
from crewai_tools import SerperDevTool
import os
import yaml

from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["SERPER_API_KEY"] = config['SERPER_API_KEY']

llm = LLM(model='gemini/gemini-2.5-pro-exp-03-25', api_key=config['GOOGLE_API_KEY'], verbose=True)
# llm= LLM(model="ollama/deepseek-r1:7b", base_url="http://localhost:11434")

researcher = Agent(
    llm=llm,
    tools=[SerperDevTool()],
    verbose=True,
    role = "Researcher",
    goal="Find the best AI tool for a given task",
    # function_calling_llm=llm,
    backstory= "You have a background in Machine Learning and Data Science",
)

research_task = Task(
    description="Find the best AI tool for a given task",
    expected_output="A summary of the top 3 trending AI News headlines",
    agent=researcher
)

crew = Crew(
    tasks=[research_task],
    agents=[researcher],
    verbose=True,
    process= Process.sequential
)

result = crew.kickoff()
print(result)