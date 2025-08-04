from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from datetime import datetime, timedelta
from openai import OpenAI

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# Create OpenAI client using API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JIRA_SITE = "https://issues.apache.org/jira"
PROJECT_KEY = "SPARK"
search_url = f"{JIRA_SITE}/rest/api/2/search"

# 🧐 Define version helpers
def get_latest_unreleased_version(versions):
    unreleased = [v for v in versions if not v.get("released", False)]
    if not unreleased:
        print("⚠️ No unreleased versions found.")
        exit()
    return sorted(unreleased, key=lambda v: v["name"], reverse=True)[0]["name"]

def get_latest_released_version(versions):
    released = [v for v in versions if v.get("released", False) and "releaseDate" in v]
    if not released:
        print("⚠️ No released versions found.")
        exit()
    return sorted(released, key=lambda v: v["releaseDate"], reverse=True)

# 🟢 Prompt user for analysis mode
print("Choose analysis mode:")
print("[1] Current in-progress summary")
print("[2] Retrospective (recently closed issues)")
mode = input("Enter 1 or 2: ").strip()

# Step 1: Fetch all fixVersions
version_url = f"{JIRA_SITE}/rest/api/2/project/{PROJECT_KEY}/versions"
version_response = requests.get(version_url)
versions = version_response.json()

# Step 2: Determine fixVersion and JQL based on mode
if mode == "1":
    fix_version = get_latest_unreleased_version(versions)
    print(f"📦 Using unreleased fixVersion: {fix_version}")
    jql = (
        f'project = {PROJECT_KEY} AND fixVersion = "{fix_version}" '
        f'AND statusCategory != Done'
    )
    gpt_prompt = "You are an expert Jira analyst. Summarize the current open work and suggest priorities."

elif mode == "2":
    print("🔍 Looking for recent closed issues in latest released versions...")

    fix_version = None
    recent_issues = []

    for v in get_latest_released_version(versions):
        candidate_version = v["name"]
        print(f"🔎 Checking fixVersion: {candidate_version}")
        
        temp_jql = (
            f'project = {PROJECT_KEY} AND fixVersion = "{candidate_version}" '
            f'AND statusCategory = Done AND issuetype IN (Bug, Task, Improvement)'
        )

        params = {
            "jql": temp_jql,
            "maxResults": 100,
            "fields": "key,summary,description,status,issuetype,resolutiondate"
        }

        response = requests.get(search_url, params=params)
        issues = response.json().get("issues", [])

        if not issues:
            continue

        # Try to pull only those resolved within the last 14 days of the most recent ticket
        issues_sorted = sorted(
            issues,
            key=lambda x: x["fields"].get("resolutiondate", ""),
            reverse=True
        )

        most_recent = issues_sorted[0]["fields"].get("resolutiondate")
        if not most_recent:
            continue

        recent_cutoff = datetime.fromisoformat(most_recent[:-5]) - timedelta(days=14)

        for issue in issues_sorted:
            resolved = issue["fields"].get("resolutiondate")
            if resolved:
                resolved_date = datetime.fromisoformat(resolved[:-5])
                if resolved_date >= recent_cutoff:
                    recent_issues.append(issue)

        if recent_issues:
            fix_version = candidate_version
            print(f"📦 Pulling retrospective from fixVersion: {fix_version}")
            break

    if not fix_version or not recent_issues:
        print("❌ No recently resolved issues found in any recent versions.")
        exit()

    gpt_prompt = (
        "You are facilitating a sprint retrospective. Summarize what was accomplished, recurring themes, "
        "and suggest discussion points for the team."
    )
else:
    print("❌ Invalid mode. Exiting.")
    exit()

# Step 3: Prepare issues for GPT
print(f"✅ Pulled {len(recent_issues) if mode == '2' else 'N/A'} issues")
issues = recent_issues if mode == "2" else requests.get(search_url, params={
    "jql": jql,
    "maxResults": 100,
    "fields": "key,summary,description,status,issuetype"
}).json().get("issues", [])

if not issues:
    print("⚠️ No matching issues found. Exiting without GPT summary.")
    exit()

# Step 4: Format for GPT
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

# Step 5: Send to GPT
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": gpt_prompt},
        {"role": "user", "content": f"Here are the Jira issues:\n\n{input_text}"}
    ],
    temperature=0.4,
    max_tokens=700
)

# Output summary
print("\n📜 GPT Summary:\n")
print(response.choices[0].message.content)
