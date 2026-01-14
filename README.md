# Jira Insights Tool

A CLI tool that analyzes active Jira sprint tickets and generates contextual questions for scrum leads. Built to help program managers scale across multiple teams without manually reviewing every ticket.

## The Problem

Scrum leads managing multiple teams can't read every ticket and comment. But they still need to ask the right questions in standups, sprint reviews, and 1:1s. This tool bridges that gap—surfacing what matters so you can lead informed conversations.

## Features

- **Sprint Analysis** - Pulls active tickets and comments from Jira API
- **AI-Generated Questions** - Uses OpenAI to generate contextual discussion prompts based on ticket content
- **Multi-Team Scale** - Review multiple sprints quickly without manual ticket-by-ticket reading
- **Markdown Output** - Clean output for easy copy/paste into meeting notes

## Tech Stack

- **Language:** Python
- **APIs:** Jira REST API, OpenAI API
- **Output:** Markdown to stdout

## Example Output
```
## Sprint 42 - Team Alpha

### PROJ-1234: Payment gateway timeout handling
- What's the fallback behavior if the retry logic exhausts all attempts?
- Has QA been able to reproduce the timeout conditions consistently?

### PROJ-1235: User dashboard performance
- Are we tracking metrics to validate the 40% improvement target?
- Any dependencies on the caching work in PROJ-1180?
```

## Why This Matters

Traditional sprint reviews either require reading every ticket (doesn't scale) or going in blind (misses important context). This tool gives scrum leads a middle path: AI-generated starting points that surface blockers, risks, and open questions without the manual overhead.

Built for program managers who lead with questions, not just status updates.

## Setup
```bash
# Clone repo
git clone https://github.com/hardenbr-personal/Jira-Insights-Tool.git
cd Jira-Insights-Tool

# Set environment variables (see example.env)
cp example.env .env
# Add your JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, OPENAI_API_KEY

# Run
python main.py
```

## License

MIT