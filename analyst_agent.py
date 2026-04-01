import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=2000
)

# =========================================================
# 🔥 INSIGHT ENGINE
# =========================================================
def build_data_insights(results, eda_info=None, feature_importance=None):
    insights = []

    target = results.get('target')

    # 🔹 Feature Importance Insights
    if feature_importance:
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for feat, val in sorted_features[:5]:
            strength = "strong" if val > 0.15 else "moderate"
            insights.append(
                f"{feat} is a {strength} driver of {target} with importance score {val:.3f}."
            )

    # 🔹 Correlation Insights
    if eda_info and eda_info.get("high_correlation_pairs"):
        for pair in eda_info["high_correlation_pairs"][:3]:
            insights.append(
                f"High correlation relationship exists between {pair}."
            )

    # 🔹 Fallback safety
    if not insights:
        insights.append("No strong statistical signals detected, suggesting weak feature relationships.")

    return "\n".join(insights)

def safe_json_load(raw_content):
    try:
        # Remove markdown wrappers
        clean = raw_content.replace("```json", "").replace("```", "").strip()

        # Extract JSON object
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")

        json_text = match.group()

        # Remove control characters
        json_text = re.sub(r'[\x00-\x1F]+', ' ', json_text)

        # Fix common JSON issues
        json_text = re.sub(r',\s*}', '}', json_text)  # trailing comma
        json_text = re.sub(r',\s*]', ']', json_text)

        return json.loads(json_text)

    except Exception as e:
        print("❌ JSON PARSE FAILED")
        print("RAW OUTPUT (truncated):\n", raw_content[:1000])
        raise e
# =========================================================
# 🔥 MAIN REPORT GENERATION
# =========================================================
def generate_full_report_narrative(results, eda_info=None, feature_importance=None):
    """
    Generates HIGH-QUALITY executive report with real data-driven insights.
    """

    target = results.get('target')
    score = 0
    model = "Universal Engine"
    
    metrics = results.get('metrics', {})
    if metrics:
        for m_name, m_score in metrics.items():
            try:
                val = float(m_score)
                if val > score:
                    score = val
                    model = m_name.replace('_score', '').upper()
            except (ValueError, TypeError):
                continue
    
    # Fallback to the top-level keys if metrics loop didn't find anything
    if score == 0:
        score = results.get('score', 0)
        model = results.get('best_model', 'Unknown Model')

    # 🔥 Build real insights
    data_insights = build_data_insights(results, eda_info, feature_importance)

    prompt = f"""
    You are a senior data strategist and business consultant.

    [BUSINESS CONTEXT]
    Target Variable: {target}
    Model Used: {model}
    Model Confidence: {score:.2%}

    [DATA-DRIVEN INSIGHTS]
    {data_insights}

    [STRICT OUTPUT FORMAT]
    Return ONLY valid JSON:

    {{
        "summary": "...",
        "data_logic": "...",
        "methodology": "...",
        "insights": "...",
        "visual_analysis": "...",
        "roadmap": "...",

        "business_impact": "...",
        "roi_simulation": "...",
        "risk_analysis": "...",
        "do_nothing": "...",
        "kpi_dashboard": "..."
    }}

    [CRITICAL INSTRUCTIONS]

    1. ALL statements must be based on the insights provided.
    2. DO NOT invent new features.
    3. Convert analytical signals into business reasoning.

    -----------------------------------------
    CORE SECTIONS (EXISTING BEHAVIOR)
    -----------------------------------------

    4. INSIGHTS must explain how features influence the target.

    5. VISUAL_ANALYSIS must explain:
    - distributions
    - relationships
    - variability
    - anomalies

    6. ROADMAP:
        - Provide 5 to 6 SPECIFIC business actions
        - Each action MUST be on a NEW LINE
        - Use numbered format EXACTLY like:

        (1) First action.
        (2) Second action.
        (3) Third action.

        - DO NOT write in paragraph form

    -----------------------------------------
    🔥 NEW BOARDROOM SECTIONS
    -----------------------------------------

    7. SUMMARY (EXECUTIVE LEVEL):
        - First line MUST be a single powerful executive headline
        - MUST contain EXACTLY 3 insights
        - MUST be written as a SINGLE STRING (NOT bullets)
        - Separate insights using "•" or "." inside the string
        - Example:

        "summary": "Insight 1... Insight 2... Insight 3..."

    -----------------------------------------

    8.BUSINESS_IMPACT:
        - MUST be 6 to 7 numbered points
        - Each point MUST be 1 to 2 sentences MAX
        - Each point MUST contain ONLY ONE idea
        - No long paragraphs
        - No repetition
        - No filler phrases like "the model can also help"

    -----------------------------------------   

    9. ROI_SIMULATION:
        - Estimate potential gains (efficiency, revenue, cost reduction)
        - If revenue not present → give qualitative ROI

    -----------------------------------------

    10. RISK_ANALYSIS:
        MUST include:
        - data quality risk
        - model limitations
        - external uncertainty
        - recommendation to mitigate risk

    -----------------------------------------

    11. DO_NOTHING_SCENARIO:
        MUST explain:
        - what happens if no action is taken
        - missed opportunities
        - performance stagnation
        - competitive disadvantage

    -----------------------------------------

    12. KPI_DASHBOARD:
        - Provide 5 to 7 KPIs to track success
        - Each KPI must be on a new line

    -----------------------------------------

    [IMPORTANT RULES]

    - Each new section must be 5 to 7 sentences
    - No one-liners
    - No vague phrases like "this shows trends"
    - Must sound like a consulting report (McKinsey style)
    - Must be actionable and business-focused
    - Avoid repetition across sections

    -----------------------------------------

    [STRICT JSON RULES - CRITICAL]

    - Return ONLY valid JSON
    - NO trailing commas
    - NO explanations outside JSON
    - NO markdown formatting (no ```)
    - DO NOT use bullet symbols like *, -, or numbered lists inside JSON values
    - ALL values must be valid JSON strings (wrapped in double quotes)
    - Use plain sentences only

    If JSON is invalid, the response will be rejected.

    """
    try:
        print("⏳ Generating top-tier AI narrative...")

        response = llm.invoke(prompt)
        content = response.content.strip()

        try:
            return safe_json_load(content)
        except Exception:
            return fallback_response(target)

    except Exception as e:
        print(f"❌ AI Narrative Error: {e}")
        return fallback_response(target)


# =========================================================
# 🔥 FALLBACK (SAFETY NET)
# =========================================================
def fallback_response(target):
    return {
        "summary": f"The predictive system provides directional insights into {target}, highlighting key influencing factors.",
        "data_logic": "Data was cleaned, transformed, and validated to ensure analytical consistency.",
        "methodology": "A machine learning model was applied to capture complex feature interactions.",
        "insights": "Key drivers influence performance, but deeper analysis is required for precision optimization.",
        "visual_analysis": "Visual patterns suggest variability and relationships across core variables.",
        "roadmap": "Focus on optimizing high-impact drivers and continuously monitor performance trends."
    }