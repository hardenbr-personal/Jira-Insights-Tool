from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from datetime import datetime, timedelta
from openai import OpenAI
import re
import json

# -------------------- Setup --------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JIRA_SITE = "https://issues.apache.org/jira"
PROJECT_KEY = "SPARK"
SEARCH_URL = f"{JIRA_SITE}/rest/api/2/search"
VERSIONS_URL = f"{JIRA_SITE}/rest/api/2/project/{PROJECT_KEY}/versions"

# Models
SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
ESTIMATION_MODEL = os.getenv("OPENAI_ESTIMATION_MODEL", "gpt-4o-mini")  # New for estimates
DEEP_PRIMARY = os.getenv("OPENAI_DEEP_PRIMARY", "gpt-4o")      
DEEP_FALLBACK = os.getenv("OPENAI_DEEP_FALLBACK", "gpt-4o-mini")

INSIGHTS_PRIMARY = os.getenv("OPENAI_INSIGHTS_PRIMARY", "gpt-4o")
INSIGHTS_FALLBACK = os.getenv("OPENAI_INSIGHTS_FALLBACK", "gpt-4o-mini")

# Retrospective window and caps
RETRO_DAYS = int(os.getenv("RETRO_DAYS", "14"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))


# -------------------- Helpers --------------------
def log_usage(label, resp):
    try:
        u = getattr(resp, "usage", None)
        if not u:
            return
        pt = getattr(u, "prompt_tokens", getattr(u, "input_tokens", None))
        ct = getattr(u, "completion_tokens", getattr(u, "output_tokens", None))
        tt = getattr(u, "total_tokens", None)
        print(f"[tokens] {label}: prompt={pt} completion={ct} total={tt}")
    except Exception:
        pass


def chat(model_id, messages, max_out=900):
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_completion_tokens=max_out
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    finish = getattr(choice, "finish_reason", "unknown")
    log_usage(model_id, resp)
    return text, finish


def fetch_issues(jql: str, fields: str, max_results: int = MAX_RESULTS):
    params = {"jql": jql, "maxResults": max_results, "fields": fields}
    r = requests.get(SEARCH_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("issues", [])


def get_latest_unreleased_version(versions):
    spark_core_versions = [
        v for v in versions
        if not v.get("released", False)
        and re.match(r"^\d+\.\d+(\.\d+)?$", v["name"])
    ]
    if not spark_core_versions:
        print("⚠️ No matching unreleased Spark-core versions found.")
        raise SystemExit(0)
    return sorted(spark_core_versions, key=lambda v: v["name"], reverse=True)[0]["name"]


def get_latest_released_version(versions):
    released = [v for v in versions if v.get("released", False) and "releaseDate" in v]
    if not released:
        print("⚠️ No released versions found.")
        raise SystemExit(0)
    return sorted(released, key=lambda v: v["releaseDate"], reverse=True)


def extract_story_points(issue):
    """Extract story points from customfield or other JIRA fields"""
    fields = issue.get("fields", {})
    
    # Common story point field names in JIRA
    story_point_fields = [
        "customfield_10002",  # Common Atlassian default
        "customfield_10004", 
        "customfield_10016",
        "storyPoints"
    ]
    
    for field in story_point_fields:
        if field in fields and fields[field] is not None:
            try:
                return float(fields[field])
            except (ValueError, TypeError):
                continue
    
    return None


def fallback_estimate(issue_type, story_points):
    """Provide fallback estimates based on issue type and story points"""
    # If story points available, rough conversion
    if story_points:
        # Rough heuristic: 1 story point ≈ 0.5-1 day for most teams
        return min(story_points * 0.75, 8)  # Cap at 8 days
    
    # Basic estimates by issue type
    type_estimates = {
        "Bug": 1.5,
        "Task": 2,
        "Story": 3,
        "Epic": 8,
        "Improvement": 2.5,
        "Sub-task": 1,
        "Technical task": 2
    }
    
    return type_estimates.get(issue_type, 2)  # Default to 2 days


def generate_individual_estimate(issue):
    """Generate estimate for a single issue when batch processing fails"""
    try:
        key = issue["key"]
        summary = issue["fields"]["summary"]
        description = (issue["fields"].get("description") or "").strip()
        issue_type = issue["fields"]["issuetype"]["name"]
        story_points = extract_story_points(issue)
        
        # Build context for individual estimate
        context = f"Issue: {summary}\nType: {issue_type}"
        if description:
            desc_preview = description[:300] + "..." if len(description) > 300 else description
            context += f"\nDescription: {desc_preview}"
        if story_points:
            context += f"\nExisting Story Points: {story_points}"
        
        system_message = (
            "You are an expert software development estimator. Provide a realistic day estimate "
            "for this issue. Consider complexity, type of work, and typical development time. "
            "Respond with just a number (e.g., 0.5, 1, 2, 3, 5, etc.)"
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": context}
        ]
        
        response_text, _ = chat(ESTIMATION_MODEL, messages, max_out=50)
        
        # Extract number from response
        day_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
        if day_match:
            return float(day_match.group(1))
        else:
            # Fallback based on issue type and story points
            return fallback_estimate(issue_type, story_points)
            
    except Exception as e:
        print(f"⚠️ Error generating individual estimate for {key}: {e}")
        return fallback_estimate(issue.get("fields", {}).get("issuetype", {}).get("name"), None)


def estimate_days_for_issues(issues):
    """Generate day estimates for multiple issues in batch"""
    if not issues:
        return {}
    
    # Build batch prompt for estimation
    estimation_prompts = []
    for issue in issues:
        key = issue["key"]
        summary = issue["fields"]["summary"]
        description = (issue["fields"].get("description") or "").strip()
        issue_type = issue["fields"]["issuetype"]["name"]
        story_points = extract_story_points(issue)
        
        prompt = f"{key} ({issue_type}): {summary}"
        if description:
            # Truncate long descriptions to save tokens
            desc_preview = description[:200] + "..." if len(description) > 200 else description
            prompt += f"\nDescription: {desc_preview}"
        if story_points:
            prompt += f"\nStory Points: {story_points}"
        
        estimation_prompts.append(prompt)
    
    batched_prompt = "\n\n".join(estimation_prompts)
    
    system_message = (
        "You are an expert software development estimator. For each issue below, provide a realistic "
        "day estimate for completion by an experienced developer. Consider:\n"
        "- Issue complexity and scope\n"
        "- Type of work (Bug, Feature, Task, etc.)\n"
        "- Any story points if provided\n"
        "- Typical development, testing, and review time\n\n"
        "Respond with one line per issue in format: ISSUE_KEY: X days\n"
        "Use whole or half days (e.g., 0.5, 1, 1.5, 2, 3, 5, 8, etc.)"
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": batched_prompt}
    ]
    
    try:
        response_text, _ = chat(ESTIMATION_MODEL, messages, max_out=1000)
        
        # Parse estimates from response
        estimates = {}
        for line in response_text.splitlines():
            if ": " in line and " day" in line:
                try:
                    parts = line.strip().split(": ", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        # Extract number from "X days" format
                        day_text = parts[1].strip()
                        day_match = re.search(r'(\d+(?:\.\d+)?)', day_text)
                        if day_match:
                            estimates[key] = float(day_match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return estimates
        
    except Exception as e:
        print(f"⚠️ Error generating batch estimates: {e}")
        return {}


# -------------------- Mode selection --------------------
print("Choose analysis mode:")
print("[1] Current in-progress summary with estimates")
print("[2] Retrospective (recently closed issues)")
mode = input("Enter 1 or 2: ").strip()

# Fetch versions once (only used by mode 1)
try:
    versions = requests.get(VERSIONS_URL, timeout=60).json()
except Exception as e:
    print(f"⚠️ Error fetching versions: {e}")
    versions = []


# ==================== MODE 1: In‑progress with Estimates ====================
if mode == "1":
    fix_version = get_latest_unreleased_version(versions)
    print(f"🔍 Using fixVersion: {fix_version}")
    jql = (
        f'project = {PROJECT_KEY} AND fixVersion = "{fix_version}" '
        f'AND statusCategory != Done'
    )

    # Step 1: fetch issues with additional fields for estimation
    issues = fetch_issues(
        jql=jql,
        fields="key,summary,description,status,issuetype,assignee,customfield_10002,customfield_10004,customfield_10016",
        max_results=MAX_RESULTS
    )
    print(f"✅ Pulled {len(issues)} issues")
    if not issues:
        print("⚠️ No matching issues found. Exiting without GPT summary.")
        exit(0)

    # Step 2: Generate day estimates for all issues
    print("🤖 Generating day estimates...")
    day_estimates = estimate_days_for_issues(issues)

    # Step 3: batch one‑line summaries (cheap)
    prompt_chunks = []
    for issue in issues:
        key = issue["key"]
        summary = issue["fields"]["summary"]
        description = (issue["fields"].get("description") or "").strip()
        prompt_chunks.append(f"{key}: {summary}\n{description}")
    batched_prompt = "\n\n".join(prompt_chunks)

    summary_messages = [
        {"role": "system", "content": "You are an expert Jira analyst. For each issue below, generate a one-line summary. Format: ISSUE_KEY: One-line summary."},
        {"role": "user", "content": batched_prompt}
    ]
    resp, _ = chat(SUMMARY_MODEL, summary_messages, max_out=2000)

    summary_map = {}
    for line in (resp or "").splitlines():
        if ": " in line:
            k, s = line.strip().split(": ", 1)
            summary_map[k.strip()] = s.strip()

    summarized = [(issue, summary_map.get(issue["key"], "")) for issue in issues]

    # Step 4: format Assigned vs Unassigned with estimates
    assigned_blurbs, unassigned_blurbs = [], []
    total_days = 0
    
    for issue, summary_line in summarized:
        f = issue["fields"]
        key = issue["key"]
        summary = f["summary"]
        assignee = f.get("assignee")
        
        # Get day estimate
        estimated_days = day_estimates.get(key, 0)
        
        # If no estimate from batch API, generate individual estimate
        if estimated_days == 0:
            individual_estimate = generate_individual_estimate(issue)
            estimated_days = individual_estimate
        
        total_days += estimated_days
        
        # Format estimate display
        if estimated_days > 0:
            # Check if this was from batch API or individual generation
            if key in day_estimates:
                estimate_text = f"📅 Estimate: {estimated_days} day{'s' if estimated_days != 1 else ''}"
            else:
                estimate_text = f"📅 Estimate: {estimated_days} day{'s' if estimated_days != 1 else ''} (AI generated)"
        else:
            estimate_text = "📅 Estimate: Unable to generate"
        
        # Check for existing story points
        story_points = extract_story_points(issue)
        story_point_text = f"🔢 Story Points: {story_points}" if story_points else ""
        
        base_text = f"{key}: {summary}\n{summary_line}\n{estimate_text}"
        if story_point_text:
            base_text += f"\n{story_point_text}"
        
        if assignee and assignee.get("displayName"):
            text = f"{base_text}\n👤 Assignee: {assignee['displayName']}"
            assigned_blurbs.append(text)
        else:
            unassigned_blurbs.append(base_text)

    input_text = "\n\n".join([
        "📌 Assigned:\n\n" + "\n\n".join(assigned_blurbs) if assigned_blurbs else "",
        "⚪ Unassigned:\n\n" + "\n\n".join(unassigned_blurbs) if unassigned_blurbs else ""
    ]).strip()

    print("\n📌 Current Open Work Summary:\n")
    print(input_text)
    print(f"\n⏱️ Total Estimated Days: {total_days}")
    print(f"📊 Average per ticket: {total_days/len(issues):.1f} days")

    # Step 5: two‑pass clustering (keeping existing logic)
    cluster_lines = [f"{issue['key']}: {summary_line}" for issue, summary_line in summarized if summary_line]
    cluster_input = "\n".join(cluster_lines) or input_text

    def chat_simple(model_id, sys, usr, max_out=1200):
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            max_completion_tokens=max_out
        )
        ch = resp.choices[0]
        return (ch.message.content or "").strip(), getattr(ch, "finish_reason", "unknown")

    # Pass 1: clusters as compact JSON
    pass1_system = "You are an expert Agile analyst. Output only valid JSON. No commentary."
    pass1_user = (
        "Group these Jira issues into 3 to 6 thematic clusters.\n"
        "Return minimal JSON with this schema:\n"
        '{ "themes": [ { "title": str, "issues": [ { "key": str, "url": str } ] } ] }\n'
        f"Use URL format: {JIRA_SITE}/browse/KEY for each KEY.\n"
        "Rules:\n"
        "• Use 3 to 5 representative issues per theme to keep output compact.\n"
        "• Do not include follow-up questions in this pass.\n\n"
        f"Issues:\n{cluster_input}"
    )

    text1, finish1 = chat_simple(DEEP_PRIMARY, pass1_system, pass1_user, max_out=1200)
    if not text1:
        print(f"[info] Primary empty on pass1. finish_reason={finish1}. Trying fallback {DEEP_FALLBACK}...")
        text1, finish1 = chat_simple(DEEP_FALLBACK, pass1_system, pass1_user, max_out=1200)

    themes = []
    if text1:
        try:
            data = json.loads(text1)
            themes = data.get("themes", [])
        except Exception as e:
            print(f"[warn] Could not parse JSON from pass1: {e}")

    # Pass 2: follow‑up questions
    if not themes:
        print("\n📊 Thematic Clusters & Follow-Up Questions:\n")
        print("⚠️ No clusters returned after primary and fallback.")
    else:
        pass2_system = "You are an expert Agile analyst. Output clear Markdown."
        pass2_user = (
            "Given these themes and representative issues (as JSON), produce for each theme:\n"
            "• 3 to 5 follow‑up questions for the team.\n"
            "Keep it concise.\n\n"
            f"JSON:\n{json.dumps({'themes': themes}, ensure_ascii=False)}"
        )

        text2, finish2 = chat_simple(DEEP_PRIMARY, pass2_system, pass2_user, max_out=1000)
        if not text2:
            print(f"[info] Primary empty on pass2. finish_reason={finish2}. Trying fallback {DEEP_FALLBACK}...")
            text2, finish2 = chat_simple(DEEP_FALLBACK, pass2_system, pass2_user, max_out=1000)

        print("\n📊 Thematic Clusters:\n")
        for t in themes:
            # Calculate total days for this theme
            theme_days = sum(day_estimates.get(issue.get('key', ''), 0) 
                           for issue in t.get("issues", []))
            theme_title = t.get('title','(untitled)')
            
            print(f"• {theme_title} ({theme_days} days estimated)")
            for it in t.get("issues", []):
                issue_key = it.get('key')
                issue_days = day_estimates.get(issue_key, 0)
                days_text = f" ({issue_days} days)" if issue_days > 0 else ""
                print(f"  - {issue_key}{days_text} - {it.get('url')}")
        
        print("\n🧭 Follow‑Up Questions:\n")
        print(text2 if text2 else "⚠️ No follow‑up questions returned.")

# ==================== MODE 2: Retrospective (unchanged) ====================
elif mode == "2":
    # Step 1: fetch recently resolved issues in a time window
    jql = (
        f'project = {PROJECT_KEY} '
        f'AND statusCategory = Done '
        f'AND issuetype IN (Bug, Task, Improvement) '
        f'AND resolutiondate >= -{RETRO_DAYS}d '
        f'ORDER BY resolutiondate DESC'
    )
    fields = "key,summary,status,issuetype,resolutiondate"
    issues = fetch_issues(jql, fields, max_results=int(os.getenv("RETRO_MAX_RESULTS", "100")))
    print(f"✅ Pulled {len(issues)} recently resolved issues (last {RETRO_DAYS} days).")
    if not issues:
        print("⚠️ No issues found. Nothing to analyze.")
        exit(0)

    # Step 2: build compact JSON directly from Jira to save tokens
    compact = []
    for issue in issues:
        f = issue["fields"]
        compact.append({
            "key": issue["key"],
            "type": f["issuetype"]["name"],
            "status": f["status"]["name"],
            "summary": f["summary"],
            "resolved": f.get("resolutiondate"),
            "url": f"{JIRA_SITE}/browse/{issue['key']}"
        })

    # Step 3: Pass 1 themes as JSON
    pass1_messages = [
        {"role": "system", "content": "You are an expert Agile analyst. Output only valid JSON. No commentary."},
        {"role": "user", "content":
            "Group these recently resolved items into 3 to 5 concise themes. "
            "Return JSON: {\"themes\":[{\"title\":str,\"issues\":[{\"key\":str,\"url\":str}],\"rationale\":str}]}. "
            "Keep strings short and include only 3 to 5 representative issues per theme.\n\n"
            f"ITEMS_JSON:\n{json.dumps({'items': compact}, ensure_ascii=False)}"
        }
    ]
    t1, fin1 = chat(INSIGHTS_PRIMARY, pass1_messages, max_out=900)
    if not t1:
        print(f"[info] Primary empty (finish_reason={fin1}). Falling back to {INSIGHTS_FALLBACK} for themes...")
        t1, fin1 = chat(INSIGHTS_FALLBACK, pass1_messages, max_out=900)

    themes = []
    if t1:
        try:
            data = json.loads(t1)
            themes = data.get("themes", [])
        except Exception as e:
            print(f"[warn] Could not parse Pass 1 JSON: {e}")

    if not themes:
        print("\n📊 Retrospective Overview:\n")
        print("⚠️ No themes returned after primary and fallback.")
        exit(0)

    # Step 4: Pass 2 narrative with caps
    pass2_messages = [
        {"role": "system", "content": "You are an expert Agile facilitator. Output clear, concise Markdown."},
        {"role": "user", "content":
            "Write a compact retrospective from the THEMES_JSON. "
            "Sections and hard caps:\n"
            "## Highlights (max 5 bullets)\n"
            "## Themes (use the given titles; for each, 2 to 3 bullets explaining impact)\n"
            "## Risks (max 5 bullets)\n"
            "## Action Items (max 7 items; owner optional; keep each to 1 line)\n"
            "Keep total length tight. Do not restate every issue, just synthesize.\n\n"
            f"THEMES_JSON:\n{json.dumps({'themes': themes}, ensure_ascii=False)}"
        }
    ]
    t2, fin2 = chat(INSIGHTS_PRIMARY, pass2_messages, max_out=1100)
    if not t2:
        print(f"[info] Primary model empty (finish_reason={fin2}). Falling back to {INSIGHTS_FALLBACK} for narrative...")
        t2, fin2 = chat(INSIGHTS_FALLBACK, pass2_messages, max_out=1100)

    # Step 5: print nicely
    print("\n📊 Retrospective Overview:\n")
    for th in themes:
        title = th.get("title", "(untitled)")
        items = th.get("issues", [])[:5]
        print(f"• {title}")
        for it in items:
            print(f"  - {it.get('key')} - {it.get('url')}")
    print("\n📝 Facilitator Summary & Actions:\n")
    print(t2 if t2 else "⚠️ No narrative returned after primary and fallback.")

else:
    print("❌ Invalid mode. Exiting.")