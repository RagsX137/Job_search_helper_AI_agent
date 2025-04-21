from crewai import Agent, Crew, Task, Process, LLM 
from crewai_tools import SerperDevTool
import os
from dotenv import dotenv_values
import pandas as pd

resume_keywords_path = "./data/resume_keywords.csv"
jd_path = "./data/job_description.txt"
resume_path = "./data/resume.txt"
config = dotenv_values(".env")
os.environ["SERPER_API_KEY"] = config['SERPER_API_KEY']

llm = LLM(model='gemini/gemini-2.5-pro-exp-03-25', api_key=config['GOOGLE_API_KEY'], verbose=True)
# llm= LLM(model="ollama/deepseek-r1:7b", base_url="http://localhost:11434")

def parse_resume_keywords(path: str) -> list:
    df = pd.read_csv(path)
    # Remove leading spaces from column names and values
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Convert DataFrame to a list of keywords
    keywords_list = df.columns.tolist()
    keywords_list = [str(keyword).strip() for keyword in keywords_list if pd.notna(keyword)]

    return keywords_list

def read_file(filepath: str | None) -> str | Exception | None:
    """
    Reads content from a file.
    Returns content as string, an Exception object, or None if path is None or file does not exist.
    """
    if filepath is None:
        return None

    if not os.path.exists(filepath):
        return FileNotFoundError(f"Input file not found at {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return IOError(f"Error reading file {filepath}: {e}")




############### ANALYZE JD ################

jd_analyzer = Agent(
    llm=llm,
    verbose=True,
    role = "Hiring Manager",
    goal="Analyze the job description to hire the right candidate",
    # function_calling_llm=llm,
    backstory= "You are an expert hiring manager with background in Machine Learning and Data Science",
)

review_task = Task(
    description= f"Analyze the given job description at {read_file(jd_path)}. Extract and summarize daily responsibilities and required qualifications in a concise way",
    expected_output="Summary of daily responsibilities and required qualifications",
    agent=jd_analyzer,
    output_file="jd_summary.txt",
)


############### FIND KEYWORDS ################

keyword_finder = Agent(
    llm=llm,
    verbose=True,
    role = "Keyword Finder",
    goal="You must find they keywords (skills/technologies/tools) that an applicant should add to their resume to get past the Applicant Tracking System (ATS)",
    backstory= "You are an expert software engineer responsible for desingining applicant tracking systems (ATS) for roles in AI, Machine Learning and Data Science",
)

find_keywords = Task(
    description=f"Extract the keywords and keyword phrases in a {read_file(jd_path)} that need to be added in a resume to get past the Applicant Tracking System (ATS). Do not use any information outside of this source",
    expected_output="List of keywords and keyword phrases to be added in a resume",
    output_file="keywords_found.txt",
    agent=keyword_finder,
)

############### COMPARE RESUME KEYWORDS TO JOB DESCRIPTION KEYWORDS AND HIGHLIGHT MISSING KEYWORDS ################

keyword_analyzer = Agent(
    llm=llm,
    verbose=True,
    role = "Keyword Matcher",
    goal="Find keywords missing from the resume ",
    backstory= "You are an expert in manipulating the Applicant Tracking System (ATS). You have extensive experience in analyzing resume keywords and finding which ones match to keywords in the job description ",
)

keywords_matcher = Task(
    description=f"Review the list of keywords found in {read_file(jd_path)} and see which ones need to be added to {parse_resume_keywords(resume_keywords_path)}. Do not use any information outside of this source",
    expected_output="List of keywords and keyword phrases missing from the resume",
    output_file="keywords_matches.txt",
    agent=keyword_finder,
)