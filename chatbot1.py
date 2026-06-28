from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import re
import time
from difflib import SequenceMatcher
from io import StringIO

import pandas as pd
import requests
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
CORS(app)

OUTPUT_FILE = "output.json"
SHEET_ID = "1DXrsx4g7WH-E-40YKPn1gLlh0exozlHbkVUonbL7ABg"
SHEET_GID = "999155847"
SHEET_BASE_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq"
SKIP_SCORE_FIELDS = {
    "Timestamp",
    "Score",
    "Calculated Score",
    "Email Address",
    "Logical Reasoning",
    "Memory & Attention",
    "Memory and attention",
    "Mathematical Skills",
    "Mathematical skills",
    "Verbal Reasoning",
}
DOMAIN_ALIASES = {
    "Logical Reasoning": ["logical", "logic", "reasoning"],
    "Memory & Attention": ["memory", "attention", "focus"],
    "Mathematical Skills": ["math", "maths", "mathematical", "mathematics", "numbers"],
    "Verbal Reasoning": ["verbal", "language", "word", "words"],
}

# =================== NORMALIZATION ===================
def normalize_text(text):
    """Improved normalization that handles your specific cases"""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("'", "").replace("’", "")
    # Keep essential symbols and normalize whitespace
    text = re.sub(r'[^\w\s⬝]', ' ', text)  # Replace most punctuation with spaces
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    return text.lower().strip()

# =================== LOAD SCORES ===================
def load_scores():
    if not os.path.exists(OUTPUT_FILE):
        return []
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_scores(scores):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
        json.dump(scores, file, indent=2, ensure_ascii=False)

def build_sheet_url(output_type):
    params = {
        "tqx": f"out:{output_type}",
        "gid": SHEET_GID,
        "headers": "1",
        "cache_bust": str(time.time_ns()),
    }
    return SHEET_BASE_URL, params

def refresh_scores_from_sheet():
    """Pull latest Google Form responses into output.json."""
    try:
        records = fetch_sheet_records_from_gviz_json()

        if not records:
            records = fetch_sheet_records_from_csv()

        if not records:
            return False, "Google Sheet did not return any student records; keeping existing output.json."

        field_count = len(records[0])
        if field_count < 10:
            return False, f"Google Sheet returned only {field_count} columns; keeping existing output.json."

        save_scores(records)
        return True, f"Updated output.json dynamically with {len(records)} latest form responses."
    except Exception as exc:
        return False, f"Could not refresh Google Sheet data: {exc}"

def fetch_sheet_records_from_gviz_json():
    url, params = build_sheet_url("json")
    response = requests.get(url, params=params, timeout=15, headers={"Cache-Control": "no-cache"})
    response.raise_for_status()

    text = response.text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return []

    payload = json.loads(text[start:end + 1])
    table = payload.get("table", {})
    columns = table.get("cols", [])
    rows = table.get("rows", [])
    headers = [str(column.get("label") or f"Column {index + 1}").strip() for index, column in enumerate(columns)]

    records = []
    for row in rows:
        cells = row.get("c", [])
        record = {}
        for index, header in enumerate(headers):
            cell = cells[index] if index < len(cells) else None
            value = ""
            if cell:
                value = cell.get("f", cell.get("v", ""))
            record[header] = "" if value is None else str(value).strip()

        if any(value != "" for value in record.values()):
            records.append(record)

    return records

def fetch_sheet_records_from_csv():
    url, params = build_sheet_url("csv")
    response = requests.get(url, params=params, timeout=15, headers={"Cache-Control": "no-cache"})
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), dtype=str).fillna("")
    df.columns = [str(column).strip() for column in df.columns]
    return df.to_dict(orient="records")

def get_sheet_status():
    refreshed, refresh_message = refresh_scores_from_sheet()
    scores = load_scores()

    if not scores:
        return (
            f"{refresh_message}\n\n"
            f"Connected Sheet ID: {SHEET_ID}\n"
            f"Connected GID: {SHEET_GID or 'default first sheet'}\n"
            f"Rows in output.json: 0"
        )

    first_student = scores[0]
    last_student = scores[-1]

    return (
        f"{refresh_message}\n\n"
        f"Connected Sheet ID: {SHEET_ID}\n"
        f"Connected GID: {SHEET_GID or 'default first sheet'}\n"
        f"Rows fetched from connected sheet: {len(scores)}\n"
        f"First response: {first_student.get('Timestamp', 'unknown')} | {first_student.get('Email Address', 'no email')}\n"
        f"Latest response: {last_student.get('Timestamp', 'unknown')} | {last_student.get('Email Address', 'no email')}\n\n"
        "If your Google Form shows a different response count, this backend is connected to a different response sheet/tab."
    )

def refresh_and_score_records():
    refreshed, refresh_message = refresh_scores_from_sheet()
    scores = load_scores()

    for student in scores:
        total_score, domain_scores, _ = score_student(student)
        student["Calculated Score"] = str(total_score)
        student["Logical Reasoning"] = str(domain_scores["Logical Reasoning"])
        student["Memory & Attention"] = str(domain_scores["Memory & Attention"])
        student["Mathematical Skills"] = str(domain_scores["Mathematical Skills"])
        student["Verbal Reasoning"] = str(domain_scores["Verbal Reasoning"])

    if scores:
        save_scores(scores)

    return scores, refreshed, refresh_message

# =================== QUESTION & ANSWER KEYS ===================
QUESTION_CATEGORIES = {
    # Logical Reasoning
    "if.*3.*2.*8.*7.*6": "Logical Reasoning",
    "identify.*not belong": "Logical Reasoning",
    "deep.*space.*mission": "Logical Reasoning",
    "village.*two kinds": "Logical Reasoning",
    "clock.*chimes": "Logical Reasoning",
    
    # Memory & Attention
    "james.*orchard": "Memory & Attention",
    "managing.*fruit market": "Memory & Attention",
    "enter.*café": "Memory & Attention",
    "phone.*number.*board": "Memory & Attention",
    "researcher.*statement": "Memory & Attention",
    
    # Mathematical Skills
    "missing.*sequence 2 6 12": "Mathematical Skills",
    "missing.*sequence 3 6 11": "Mathematical Skills",
    "buckets.*coins": "Mathematical Skills",
    "next.*1 1 2 3 5 8": "Mathematical Skills",
    "detective.*crime scene": "Mathematical Skills",
    
    # Verbal Reasoning
    "10 minutes.*test": "Verbal Reasoning",
    "one item.*deserted island": "Verbal Reasoning",
    "word.*not belong": "Verbal Reasoning",
    "proverb.*watched pot": "Verbal Reasoning",
    "rearrange.*words": "Verbal Reasoning"
}

ANSWER_SCORES = {
    # Logical Reasoning
    "if.*3.*2.*8.*7.*6": {"correct": "48", "medium": "47"},
    "identify.*not belong": {"correct": "carrot", "medium": "banana"},
    "deep.*space.*mission": {
        "correct": "planet x is closer to the star than planet y", 
        "medium": "planet y is moving faster in its orbit than planet x"
    },
    "village.*two kinds": {"correct": "ben", "medium": "alex"},
    "clock.*chimes": {"correct": "156", "medium": "180"},
    
    # Memory & Attention
    "james.*orchard": {"correct": "tree", "medium": "house"},
    "managing.*fruit market": {"correct": "3", "medium": "2"},
    "enter.*café": {"correct": "credit card machines", "medium": "donuts"},
    "phone.*number.*board": {"correct": "0", "medium": "3"},
    "researcher.*statement": {"correct": "3", "medium": "4"},
    
    # Mathematical Skills
    "missing.*sequence 2 6 12": {"correct": "30", "medium": "24"},
    "missing.*sequence 3 6 11": {"correct": "38", "medium": "35"},
    "buckets.*coins": {"correct": "20", "medium": "15"},
    "next.*1 1 2 3 5 8": {"correct": "13", "medium": "11"},
    "detective.*crime scene": {"correct": "2710", "medium": "3010"},
    
    # Verbal Reasoning
    "10 minutes.*test": {"correct": "prioritize the easiest and quickest questions", "medium": "attempt only the hardest questions"},
    "one item.*deserted island": {"correct": "a knife", "medium": "a book"},
    "word.*not belong": {"correct": "sphere", "medium": "triangle"},
    "proverb.*watched pot": {
        "correct": "if youre impatient time feels slower", 
        "medium": "water takes longer to boil if you stare at it"
    },
    "rearrange.*words": {
        "correct": "those who hesitate rarely go ahead", 
        "medium": "rarely go ahead those who hesitate"
    }
}

IMPROVEMENT_TIPS = {
    "Logical Reasoning": [
        "Practice pattern recognition exercises",
        "Try solving logic puzzles regularly",
        "Work on identifying relationships between concepts",
        "Practice breaking down complex problems into smaller parts"
    ],
    "Memory & Attention": [
        "Try mindfulness exercises to improve focus",
        "Practice memory techniques like chunking",
        "Work on active listening skills",
        "Try memory games and concentration exercises"
    ],
    "Mathematical Skills": [
        "Practice number sequence problems",
        "Work on basic arithmetic regularly",
        "Try solving word problems to apply math concepts",
        "Practice identifying mathematical patterns"
    ],
    "Verbal Reasoning": [
        "Read diverse materials to expand vocabulary",
        "Practice identifying word relationships",
        "Work on understanding context in reading",
        "Try solving word puzzles and anagrams"
    ]
}

# =================== CORE SCORING FUNCTION ===================
def score_student(student):
    """Score an individual student and return detailed results"""
    total_score = 0
    domain_scores = {
        "Logical Reasoning": 0,
        "Memory & Attention": 0,
        "Mathematical Skills": 0,
        "Verbal Reasoning": 0
    }
    
    question_details = []
    
    for question, answer in student.items():
        if question in SKIP_SCORE_FIELDS:
            continue
            
        normalized_q = normalize_text(question)
        student_answer = normalize_text(str(answer))
        
        # Find matching question pattern
        matched_key = None
        for q_pattern in QUESTION_CATEGORIES:
            if re.search(q_pattern, normalized_q):
                matched_key = q_pattern
                break
        
        if matched_key:
            domain = QUESTION_CATEGORIES[matched_key]
            correct = normalize_text(str(ANSWER_SCORES[matched_key]["correct"]))
            medium = normalize_text(str(ANSWER_SCORES[matched_key]["medium"]))
            
            if student_answer == correct:
                score = 10
                feedback = "Excellent!"
            else:
                score = 0
                feedback = "Incorrect"
            
            domain_scores[domain] += score
            total_score += score
            
            question_details.append({
                "question": question,
                "domain": domain,
                "student_answer": answer,
                "correct_answer": ANSWER_SCORES[matched_key]["correct"],
                "score": score,
                "feedback": feedback
            })
    
    return total_score, domain_scores, question_details

def get_official_sheet_score(student, fallback_score):
    raw_score = str(student.get("Score", "")).strip()
    match = re.search(r'\d+(?:\.\d+)?', raw_score)
    if not match:
        return fallback_score, f"{fallback_score}/200"

    value = float(match.group(0))
    score_number = int(value) if value.is_integer() else value
    return score_number, f"{score_number}/200"

# =================== APRIORI ANALYSIS ===================
def build_transactions(scores):
    """Build transactions for Apriori analysis"""
    transactions = []
    per_student_low = []
    
    for student in scores:
        _, domain_scores, _ = score_student(student)
        low_domains = [f"Low_{domain}" for domain, score in domain_scores.items() if score < 25]
        medium_domains = [f"Medium_{domain}" for domain, score in domain_scores.items() if 25 <= score < 40]
        high_domains = [f"High_{domain}" for domain, score in domain_scores.items() if score >= 40]
        
        transaction = low_domains + medium_domains + high_domains
        transactions.append(transaction)
        per_student_low.append(low_domains)
    
    return transactions, per_student_low

def get_apriori_rules(transactions, min_support=0.2, min_confidence=0.6):
    """Generate association rules using Apriori algorithm"""
    if not transactions or len(transactions) < 3:
        return pd.DataFrame()
    
    # One-hot encode transactions
    all_items = sorted(set(item for t in transactions for item in t))
    encoded_rows = [{item: (item in t) for item in all_items} for t in transactions]
    df = pd.DataFrame(encoded_rows)
    
    if df.empty or df.sum().sum() == 0:
        return pd.DataFrame()
    
    try:
        freq_items = apriori(df, min_support=min_support, use_colnames=True)
        if freq_items.empty:
            return pd.DataFrame()
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
        return rules
    except Exception as e:
        print(f"Apriori error: {e}")
        return pd.DataFrame()

# =================== INDIVIDUAL STUDENT ANALYSIS ===================
def analyze_individual_student(student, all_rules, student_index, all_students):
    """Analyze individual student with personalized feedback"""
    email = student.get("Email Address", "No email provided")
    calculated_score, domain_scores, question_details = score_student(student)
    total_score, total_score_label = get_official_sheet_score(student, calculated_score)
    
    # Generate domain feedback
    feedback = []
    improvement_areas = []
    
    domain_feedback = {}
    for domain, score in domain_scores.items():
        if score >= 40:
            level = "Excellent"
            message = "Strong performance. Keep practicing to maintain this level."
        elif score >= 25:
            level = "Good"
            message = "Good base. A little more targeted practice can improve consistency."
        else:
            level = "Needs improvement"
            message = "This is a priority area. Use the improvement tips below."
            improvement_areas.append(domain)
        feedback.append(f"{domain}: {score}/50 - {level}")
        domain_feedback[domain] = {
            "score": f"{score}/50",
            "level": level,
            "message": message
        }
    
    # Add improvement tips for low-scoring domains
    improvement_tips = {}
    if improvement_areas:
        for area in improvement_areas:
            improvement_tips[area] = IMPROVEMENT_TIPS[area]
    
    # Overall assessment
    if total_score >= 180:
        overall_feedback = "Excellent cognitive ability 🎯"
    elif total_score >= 120:
        overall_feedback = "Good cognitive skills 👍"
    else:
        overall_feedback = "Needs improvement 📚"
    
    apriori_insights = build_student_apriori_insights(domain_scores, all_rules)
    
    return {
        "Student Email": email,
        "Logical Reasoning": f"{domain_scores['Logical Reasoning']}/50",
        "Memory & Attention": f"{domain_scores['Memory & Attention']}/50",
        "Mathematical Skills": f"{domain_scores['Mathematical Skills']}/50",
        "Verbal Reasoning": f"{domain_scores['Verbal Reasoning']}/50",
        "Total Score": total_score_label,
        "Calculated Score": f"{calculated_score}/200",
        "Overall Feedback": overall_feedback,
        "Domain Feedback": domain_feedback,
        "Improvement Tips": improvement_tips,
        "Apriori Insights": apriori_insights,
        "Question Details": question_details
    }

# =================== MAIN ANALYSIS FUNCTION ===================
def analyze_scores_with_apriori():
    """Main function to analyze all students with Apriori insights"""
    scores, refreshed, refresh_message = refresh_and_score_records()
    if not scores:
        return "No student data found."
    
    # Generate Apriori rules for all students
    transactions, per_student_low = build_transactions(scores)
    rules = get_apriori_rules(transactions, min_support=0.2, min_confidence=0.6)
    
    # Analyze each student individually
    individual_results = []
    for i, student in enumerate(scores):
        result = analyze_individual_student(student, rules, i, scores)
        individual_results.append(result)
    
    return {
        "refresh_status": refresh_message,
        "refreshed": refreshed,
        "students_analyzed": len(scores),
        "results": individual_results
    }

def build_student_apriori_insights(domain_scores, all_rules, max_insights=3):
    if all_rules.empty:
        return []

    student_domains = []
    focus_levels = []
    for domain, score in domain_scores.items():
        if score < 25:
            level = "Low"
        elif score < 40:
            level = "Medium"
        else:
            level = "High"

        student_domains.append(f"{level}_{domain}")
        if level != "High":
            focus_levels.append(f"{level}_{domain}")

    # Avoid giving every strong student the same generic high-performance rules.
    if not focus_levels:
        return []

    student_domain_set = set(student_domains)
    candidates = []

    for _, rule in all_rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])

        if not antecedents.issubset(student_domain_set):
            continue

        related_to_growth_area = bool((antecedents | consequents) & set(focus_levels))
        if not related_to_growth_area:
            continue

        candidates.append(rule)

    candidates = sorted(candidates, key=lambda rule: (rule['confidence'], rule['support']), reverse=True)
    insights = []

    for rule in candidates[:max_insights]:
        antecedents = format_rule_items(rule['antecedents'])
        consequents = format_rule_items(rule['consequents'])
        insights.append(
            f"Students with {antecedents} often also show {consequents} "
            f"(confidence: {rule['confidence']:.2f}). Use this as a cohort pattern, not an individual diagnosis."
        )

    return insights

def format_rule_items(items):
    return ", ".join(str(item).replace("_", " ") for item in sorted(items))

# =================== GLOBAL RULES FUNCTION ===================
def generate_global_rules():
    """Generate global association rules for all students"""
    scores, refreshed, refresh_message = refresh_and_score_records()
    if not scores:
        return "No student data available for association mining."
    
    transactions, _ = build_transactions(scores)
    rules = get_apriori_rules(transactions, min_support=0.2, min_confidence=0.6)
    
    if rules.empty:
        return "No significant association rules found."
    
    # Sort by confidence and get top rules
    top_rules = rules.sort_values(by="confidence", ascending=False).head(10)
    
    results = []
    results.append("Cognitive Pattern Insights (Association Rules):")
    results.append("=" * 50)
    
    for _, rule in top_rules.iterrows():
        antecedents = [str(item) for item in rule['antecedents']]
        consequents = [str(item) for item in rule['consequents']]
        results.append(f"• If student has {', '.join(antecedents)}")
        results.append(f"  → Then they likely have {', '.join(consequents)}")
        results.append(f"  (Confidence: {rule['confidence']:.2f}, Support: {rule['support']:.2f})")
        results.append("")
    
    results.append(refresh_message)
    results.append(f"Total students analyzed: {len(scores)}")
    results.append(f"Total rules generated: {len(rules)}")
    
    return "\n".join(results)

def get_student_score(user_message):
    scores, refreshed, refresh_message = refresh_and_score_records()
    if not scores:
        return "No student data found."

    selected_student = find_requested_student(scores, user_message)
    if isinstance(selected_student, str):
        return f"{refresh_message}\n\n{selected_student}"

    calculated_score, domain_scores, _ = score_student(selected_student)
    total_score, total_score_label = get_official_sheet_score(selected_student, calculated_score)
    email = selected_student.get("Email Address", "latest form response")

    return (
        f"{refresh_message}\n\n"
        f"Score for {email}:\n"
        f"- Logical Reasoning: {domain_scores['Logical Reasoning']}/50\n"
        f"- Memory & Attention: {domain_scores['Memory & Attention']}/50\n"
        f"- Mathematical Skills: {domain_scores['Mathematical Skills']}/50\n"
        f"- Verbal Reasoning: {domain_scores['Verbal Reasoning']}/50\n"
        f"- Total Score: {total_score_label}"
    )

def find_requested_student(scores, user_message):
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_message)

    if email_match:
        requested_email = email_match.group(0).lower()
        for student in reversed(scores):
            if str(student.get("Email Address", "")).lower() == requested_email:
                return student

        suggestion = find_closest_email(scores, requested_email)
        if suggestion:
            return f"I could not find {requested_email}. Did you mean {suggestion}?"
        return f"I could not find a form response for {requested_email}."

    return scores[-1]

def find_closest_email(scores, requested_email):
    best_email = None
    best_ratio = 0
    for student in scores:
        email = str(student.get("Email Address", "")).lower()
        if not email:
            continue
        ratio = SequenceMatcher(None, requested_email, email).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_email = email

    return best_email if best_ratio >= 0.72 else None

def get_student_analysis(user_message):
    scores, refreshed, refresh_message = refresh_and_score_records()
    if not scores:
        return "No student data found."

    selected_student = find_requested_student(scores, user_message)
    if isinstance(selected_student, str):
        return f"{refresh_message}\n\n{selected_student}"

    transactions, _ = build_transactions(scores)
    rules = get_apriori_rules(transactions, min_support=0.2, min_confidence=0.6)
    student_index = scores.index(selected_student)

    return {
        "refresh_status": refresh_message,
        "refreshed": refreshed,
        "students_analyzed": 1,
        "results": [analyze_individual_student(selected_student, rules, student_index, scores)]
    }

def get_class_summary():
    scores, refreshed, refresh_message = refresh_and_score_records()
    if not scores:
        return "No student data found."

    domain_totals = {
        "Logical Reasoning": 0,
        "Memory & Attention": 0,
        "Mathematical Skills": 0,
        "Verbal Reasoning": 0
    }
    student_summaries = []

    for student in scores:
        calculated_score, domain_scores, _ = score_student(student)
        total_score, _ = get_official_sheet_score(student, calculated_score)
        email = student.get("Email Address", "No email provided")
        for domain, score in domain_scores.items():
            domain_totals[domain] += score
        student_summaries.append({
            "email": email,
            "total": total_score,
            "domain_scores": domain_scores
        })

    domain_averages = {
        domain: round(total / len(scores), 1)
        for domain, total in domain_totals.items()
    }
    overall_average = round(sum(student["total"] for student in student_summaries) / len(scores), 1)
    strongest_domain = max(domain_averages, key=domain_averages.get)
    weakest_domain = min(domain_averages, key=domain_averages.get)
    support_students = sorted(
        [student for student in student_summaries if student["total"] < 120],
        key=lambda student: student["total"]
    )[:5]

    return {
        "type": "class_summary",
        "refresh_status": refresh_message,
        "students_analyzed": len(scores),
        "class_average": overall_average,
        "domain_averages": domain_averages,
        "strongest_domain": {
            "name": strongest_domain,
            "average": domain_averages[strongest_domain]
        },
        "weakest_domain": {
            "name": weakest_domain,
            "average": domain_averages[weakest_domain]
        },
        "support_students": [
            {
                "email": student["email"],
                "total": student["total"],
                "weakest_domain": min(student["domain_scores"], key=student["domain_scores"].get)
            }
            for student in support_students
        ]
    }

# =================== NLP INTENT LAYER ===================
def detect_domain(message):
    normalized_message = normalize_text(message)
    for domain, aliases in DOMAIN_ALIASES.items():
        if any(alias in normalized_message for alias in aliases):
            return domain
    return None

def extract_score_threshold(message, default_threshold=120):
    match = re.search(r'(?:below|under|less than|lower than|<)\s*(\d+)', message)
    if match:
        return int(match.group(1))
    return default_threshold

def fuzzy_contains(message, phrases, threshold=0.78):
    normalized_message = normalize_text(message)
    for phrase in phrases:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase in normalized_message:
            return True
        ratio = SequenceMatcher(None, normalized_message, normalized_phrase).ratio()
        if ratio >= threshold:
            return True
    return False

def detect_intent(message):
    normalized_message = normalize_text(message)
    has_email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message) is not None

    if fuzzy_contains(normalized_message, ["sheet status", "form status", "data status", "response count", "sheet count"]):
        return "sheet_status"
    if fuzzy_contains(normalized_message, ["class summary", "class dashboard", "dashboard", "class performance", "overall performance", "average score"]):
        return "class_summary"
    if "association" in normalized_message or "apriori" in normalized_message or "pattern" in normalized_message:
        return "association_rules"
    if any(term in normalized_message for term in ["weak", "low", "below", "under", "less than", "need help", "support", "struggling"]):
        return "weak_students"
    if has_email and any(term in normalized_message for term in ["analyze", "analysis", "report", "details", "detail"]):
        return "student_analysis"
    if any(term in normalized_message for term in ["latest", "recent", "new response", "new student", "full analysis", "report", "analyze student"]):
        return "student_analysis"
    if any(term in normalized_message for term in ["score", "marks", "mark", "result"]):
        return "student_score"
    return None

def get_weak_students_report(user_message):
    scores, refreshed, refresh_message = refresh_and_score_records()
    if not scores:
        return "No student data found."

    domain = detect_domain(user_message)
    threshold = extract_score_threshold(user_message)
    weak_students = []

    for student in scores:
        calculated_score, domain_scores, _ = score_student(student)
        official_score, _ = get_official_sheet_score(student, calculated_score)
        email = student.get("Email Address", "No email provided")

        if domain:
            domain_score = domain_scores[domain]
            if domain_score < 25:
                weak_students.append((email, official_score, domain, domain_score))
        elif official_score < threshold:
            weakest_domain = min(domain_scores, key=domain_scores.get)
            weak_students.append((email, official_score, weakest_domain, domain_scores[weakest_domain]))

    weak_students.sort(key=lambda item: (item[1], item[3]))

    if domain:
        title = f"Students needing support in {domain}"
    else:
        title = f"Students below {threshold}/200"
    return {
        "type": "weak_students",
        "refresh_status": refresh_message,
        "title": title,
        "domain": domain,
        "threshold": threshold,
        "total_matches": len(weak_students),
        "students": [
            {
                "email": email,
                "total": total,
                "domain": weakest_domain,
                "domain_score": domain_score
            }
            for email, total, weakest_domain, domain_score in weak_students[:10]
        ],
        "remaining_count": max(0, len(weak_students) - 10)
    }

def handle_nlp_query(user_message):
    intent = detect_intent(user_message)

    if intent == "sheet_status":
        return get_sheet_status()
    if intent == "class_summary":
        return get_class_summary()
    if intent == "association_rules":
        return generate_global_rules()
    if intent == "weak_students":
        return get_weak_students_report(user_message)
    if intent == "student_analysis":
        return get_student_analysis(user_message)
    if intent == "student_score":
        return get_student_score(user_message)
    return None

# =================== FLASK ROUTES ===================
@app.route('/', methods=['GET'])
def home():
    return send_from_directory(app.root_path, 'index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return send_from_directory(app.root_path, 'index.html')
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    user_message = data.get("message", "").strip().lower()

    if user_message == "analyze scores":
        # Return individual student results with Apriori insights
        results = analyze_scores_with_apriori()
        return jsonify({"response": results})
    
    elif user_message == "association rules":
        # Return global association rules
        rules_text = generate_global_rules()
        return jsonify({"response": rules_text})

    elif user_message.startswith("my score") or user_message.startswith("latest score"):
        score_text = get_student_score(user_message)
        return jsonify({"response": score_text})

    elif (
        user_message.startswith("my analysis")
        or user_message.startswith("analyze latest")
        or user_message.startswith("analyze student")
    ):
        results = get_student_analysis(user_message)
        return jsonify({"response": results})

    elif user_message in ["sheet status", "form status", "data status"]:
        return jsonify({"response": get_sheet_status()})

    elif user_message in ["class summary", "dashboard", "class dashboard"]:
        return jsonify({"response": get_class_summary()})

    nlp_response = handle_nlp_query(user_message)
    if nlp_response is not None:
        return jsonify({"response": nlp_response})

    # Help responses
    response = {
        "hi": "Hello! I can analyze cognitive assessment scores with Apriori analysis. Try 'analyze scores' for individual results or 'association rules' for pattern insights.",
        "hello": "Hi there! I can analyze cognitive assessment scores with Apriori analysis. Try 'analyze scores' for individual results or 'association rules' for pattern insights.",
        "help": "I can analyze cognitive assessment scores using Apriori algorithm. Here's what you can ask:\n"
                "- 'analyze scores': View detailed individual student results with personalized feedback\n"
                "- 'association rules': Find hidden patterns in cognitive performance\n"
                "- 'my score': Refresh form data and show the latest submitted score\n"
                "- 'my score your@email.com': Refresh form data and show that email's score\n"
                "- 'my analysis': Refresh form data and show the latest student's full analysis\n"
                "- 'analyze student your@email.com': Refresh form data and show that student's full analysis\n"
                "- 'sheet status': Show which Google Sheet is connected and how many rows it returns\n"
                "- 'class summary': Show class averages, strongest/weakest domain, and support list\n"
                "- Natural language: Try 'who is weak in math?', 'show students below 100', or 'latest student report'\n"
                "- 'about': Learn about this assessment\n"
                "- 'domains': See what cognitive domains we test",
        "about": "This is a Cognitive Assessment Analysis Bot that uses NLP intent detection, entity extraction, and Association Rule Mining (Apriori algorithm) to:\n"
                 "- Evaluate logical reasoning\n"
                 "- Assess memory and attention\n"
                 "- Analyze mathematical skills\n"
                 "- Test verbal reasoning abilities\n"
                 "- Understand natural language queries like weak students, latest reports, and score lookups\n\n"
                 "The assessment consists of 20 questions across 4 domains, with personalized feedback based on cohort patterns.",
        "domains": "We assess these cognitive domains:\n"
                   "🔹 Logical Reasoning (5 questions)\n"
                   "🔹 Memory & Attention (5 questions)\n"
                   "🔹 Mathematical Skills (5 questions)\n"
                   "🔹 Verbal Reasoning (5 questions)\n\n"
                   "Each domain is scored out of 50 points.",
        "scoring": "Scoring works as follows:\n"
                   "- Correct answer: 10 points\n"
                   "- Incorrect: 0 points\n\n"
                   "Total possible score: 200 points. The displayed total score matches the Google Sheet score.",
        "thank you": "You're welcome! Happy to help with cognitive assessment analysis anytime.",
        "bye": "Goodbye! Have a great day!"
    }.get(user_message, "I can analyze scores or show association rules. Try 'analyze scores' for individual results or 'association rules' for pattern insights.")

    return jsonify({"response": response})

if __name__ == '__main__':
    print("Starting Cognitive Analysis Server with Apriori Algorithm...")
    print("Features:")
    print("- Individual student analysis with personalized feedback")
    print("- Association rule mining for pattern discovery")
    print("- Domain-specific improvement recommendations")
    app.run(debug=True, port=5000, use_reloader=False)
