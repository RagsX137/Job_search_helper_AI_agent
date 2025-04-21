# Job Search Helper AI Agent 🤖

An intelligent AI-powered assistant that helps optimize your job search process using CrewAI framework. This tool analyzes job descriptions, extracts key information, and matches keywords to improve your application success rate.

## Features

- **Automated Job Description Analysis**: Intelligently breaks down job postings to identify critical requirements and qualifications
- **Smart Keyword Extraction**: Identifies and extracts relevant keywords and skills from job descriptions
- **Keyword Matching**: Compares extracted keywords against your profile/resume to find alignment
- **Sequential Processing**: Organized workflow that processes tasks in a logical order for best results

## Benefits

- 🎯 **Improved Application Targeting**: Better understand job requirements and match your qualifications
- ⏱️ **Time Saving**: Automates the tedious process of manually analyzing job descriptions
- 📈 **Higher Success Rate**: Helps optimize applications by focusing on matching keywords and requirements
- 🔍 **Deep Analysis**: Uses advanced AI to provide insights you might miss during manual review

## Technical Stack

- Built with Python using CrewAI framework
- Utilizes specialized AI agents for different tasks:
  - JD Analyzer
  - Keyword Finder
  - Keyword Analyzer
- Integrates with SerperDev for enhanced search capabilities
- Environment variable configuration for secure API management

## Getting Started

1. Clone the repository
2. Create a `.env` file with required API keys
3. Install dependencies with pip install -r requirements
4. Copy the job description from the job posting and paste in job_description.txt
5. Copy your resume and paste in resume.txt
6. Run `main.py` to start the analysis

## Usage

The system works in three main stages:
1. Job description review and analysis
2. Keyword identification and extraction
3. Keyword matching and optimization

## Future Enhancements

- Resume optimization suggestions
- Cover letter generation
- Interview preparation assistance
- Job market trend analysis

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for improvements.

## License

