from dotenv import load_dotenv
import os
import requests
print("Working directory:", os.getcwd())
from openai import OpenAI

# Load environment variables
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# Create OpenAI client using API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JIRA_SITE = "https://issues.apache.org/jira"
PROJECT_KEY = "SPARK"

# Step 1: Get latest unreleased fixVersion
version_url = f"{JIRA_SITE}/rest/api/2/project/{PROJECT_KEY}/versions"
version_response = requests.get(version_url)
versions = version_response.json()
unreleased = [v for v in versions if not v.get("released", True)]
# Find the first unreleased version that has open issues
for v in sorted(unreleased, key=lambda v: v["name"], reverse=True):
    version_name = v["name"]
    jql = f'project = {PROJECT_KEY} AND fixVersion = "{version_name}" AND statusCategory != Done'
    check_url = f"{JIRA_SITE}/rest/api/2/search"
    response = requests.get(check_url, params={"jql": jql, "maxResults": 1})
    if response.status_code == 200 and response.json().get("total", 0) > 0:
        fix_version = version_name
        break
else:
    print("⚠️ No unreleased versions with open issues found.")
    exit()

print(f"📦 Latest active unreleased version: {fix_version}")

print(f"📦 Latest unreleased version: {fix_version}")

# Step 2: Pull all open issues for that version
search_url = f"{JIRA_SITE}/rest/api/2/search"
jql = (
    f'project = {PROJECT_KEY} AND fixVersion = "{fix_version}" '
    f'AND statusCategory != Done'
)
params = {
    "jql": jql,
    "maxResults": 100,
    "fields": "key,summary,description,status,issuetype"
}
search_response = requests.get(search_url, params=params)
issues = search_response.json().get("issues", [])
print(f"✅ Pulled {len(issues)} open issues")

# Step 3: Format for GPT
issue_blurbs = []
for issue in issues:
    f = issue["fields"]
    summary = f["summary"]
    desc = f.get("description", "") or ""
    key = issue["key"]
    type_ = f["issuetype"]["name"]
    status = f["status"]["name"]
    text = f"[{type_}] {key}: {summary} ({status})\n{desc.strip()}\n"
    issue_blurbs.append(text)

input_text = "\n".join(issue_blurbs)

# Step 4: Send to GPT-4 for summary
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert Jira analyst."},
        {"role": "user", "content": f"Here are the open issues in the latest unreleased Apache Spark version, provide a summary of insights and next steps:\n\n{input_text}"}
    ],
    temperature=0.4,
    max_tokens=700
)

# Output the summary
print("\n📝 GPT Summary:\n")
print(response.choices[0].message.content)
