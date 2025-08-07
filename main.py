from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from datetime import datetime, timedelta
from openai import OpenAI
import re

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# Create OpenAI client using API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JIRA_SITE = "https://issues.apache.org/jira"
PROJECT_KEY = "SPARK"
search_url = f"{JIRA_SITE}/rest/api/2/search"

# 😮 Define version helpers
def get_latest_unreleased_version(versions):
    spark_core_versions = [
        v for v in versions
        if not v.get("released", False)
        and re.match(r"^\d+\.\d+(\.\d+)?$", v["name"])
    ]
    if not spark_core_versions:
        print("⚠️ No matching unreleased Spark-core versions found.")
        exit()
    return sorted(spark_core_versions, key=lambda v: v["name"], reverse=True)[0]["name"]

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
    print(f"🔍 Using fixVersion: {fix_version}")
    print(f"📦 Using unreleased fixVersion: {fix_version}")
    jql = (
        f'project = {PROJECT_KEY} AND fixVersion = "{fix_version}" '
        f'AND statusCategory != Done'
    )
    gpt_prompt = "You are an expert Jira analyst. For each issue, generate a 1-line summary of its purpose or content."

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
max_issues = 50
issues = recent_issues if mode == "2" else requests.get(search_url, params={
    "jql": jql,
    "maxResults": max_issues,
    "fields": "key,summary,description,status,issuetype,assignee"
}).json().get("issues", [])[:max_issues]

print(f"✅ Pulled {len(issues)} issues")

if not issues:
    print("⚠️ No matching issues found. Exiting without GPT summary.")
    exit()

# Step 4: Batch GPT summaries
prompt_chunks = []
for issue in issues:
    key = issue["key"]
    summary = issue["fields"]["summary"]
    description = (issue["fields"].get("description") or "").strip()
    prompt_chunks.append(f"{key}: {summary}\n{description}")

batched_prompt = "\n\n".join(prompt_chunks)

batched_summary_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": (
            "You are an expert Jira analyst. For each issue below, generate a one-line summary.\n"
            "Format: KEY: One-line summary"
        )},
        {"role": "user", "content": batched_prompt}
    ],
    temperature=0.3,
    max_tokens=2000
)

summary_map = {}
for line in batched_summary_response.choices[0].message.content.strip().split("\n"):
    if ": " in line:
        key, summary_line = line.strip().split(": ", 1)
        summary_map[key.strip()] = summary_line.strip()

summarized = [(issue, summary_map.get(issue["key"], "")) for issue in issues]

# Step 5: Structure the display format
assigned_blurbs = []
unassigned_blurbs = []

for issue, summary_line in summarized:
    f = issue["fields"]
    key = issue["key"]
    summary = f["summary"]
    assignee = f.get("assignee")

    if assignee and assignee.get("displayName"):
        text = f"{key}: {summary}\n{summary_line}\n👤 Assignee: {assignee['displayName']}"
        assigned_blurbs.append(text)
    else:
        text = f"{key}: {summary}\n{summary_line}"
        unassigned_blurbs.append(text)

input_text = "\n\n".join([
    "📌 Assigned:\n\n" + "\n\n".join(assigned_blurbs) if assigned_blurbs else "",
    "⚪ Unassigned:\n\n" + "\n\n".join(unassigned_blurbs) if unassigned_blurbs else ""
]).strip()

print("\n📌 Current Open Work Summary:\n")
print(input_text)

# Step 6: Optional thematic clustering and follow-ups
if mode == "1":
    deep_prompt = (
        "Group these Jira issues into thematic clusters based on their descriptions and summaries.\n"
        "For each theme, include:\n"
        "1. A few representative issues with links (keep the ticket name/number and link on the same line).\n"
        "2. A list of follow-up questions the team should consider for this theme (e.g. blockers, ownership, coordination)."
    )

    deep_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert Agile analyst."},
            {"role": "user", "content": f"Here are the Jira issues:\n\n{input_text}\n\n{deep_prompt}"}
        ],
        temperature=0.5,
        max_tokens=900
    )

    print("\n📊 Thematic Clusters & Follow-Up Questions:\n")
    print(deep_response.choices[0].message.content)
