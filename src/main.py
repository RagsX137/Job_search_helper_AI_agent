from tools import *
from crewai import Agent, Crew, Task, Process, LLM 
from crewai_tools import SerperDevTool
import os
from dotenv import dotenv_values

config = dotenv_values(".env")
crew = Crew(
    tasks=[
        review_task,
        find_keywords,
        # keywords_matcher
        ],
    agents=[
        jd_analyzer, 
        keyword_finder,
        # keyword_analyzer
    ],
    verbose=True,
    process= Process.sequential
)

result = crew.kickoff()
# print(result)