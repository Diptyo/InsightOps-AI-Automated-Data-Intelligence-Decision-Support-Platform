import os
import json
import pandas as pd
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class VisualizerAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def plan_visualizations(self, df: pd.DataFrame, target: str, feature_importances: dict = None, problem_type: str = 'regression'):
        """
        Plans dynamic visualizations tailored specifically to the problem type (Regression vs Classification).
        """
        dtypes = df.dtypes.astype(str).to_dict()
        
        # Prepare importance string, filtering for relevant features
        importance_text = "Not provided"
        if feature_importances:
            significant = {k: v for k, v in feature_importances.items() if v > 0.01}
            sorted_imp = sorted(significant.items(), key=lambda x: x[1], reverse=True)
            importance_text = ", ".join([f"{k}: {v:.4f}" for k, v in sorted_imp])

        # PROBLEM-SPECIFIC INSTRUCTIONS
        if problem_type == 'classification':
            viz_guidelines = """
            FOR CLASSIFICATION:
            - Use 'boxplot' or 'violin' to show how numeric features differ across target classes.
            - Use 'bar' charts for categorical feature distributions.
            - Focus on class separation and overlap.
            """
        else:
            viz_guidelines = """
            FOR REGRESSION:
            - Use 'scatter' plots with the target on the Y-axis to show linear/non-linear trends.
            - Use 'line' charts if there is a time-based component (e.g., Year, Month).
            - Focus on correlation and variance.
            """

        prompt = f"""
        You are a Senior Data Scientist.
        
        Dataset Target: '{target}'
        Problem Type: {problem_type.upper()}
        Available Columns: {dtypes}
        Top Features: {importance_text}

        {viz_guidelines}

        TASK:
        Generate a JSON list of visualization plans. 
        - Prioritize features with high importance scores.
        - Each 'description' MUST be 6-8 sentences explaining: 
          1) The trend shown 
          2) The statistical reasoning 
          3) The business strategic implication.

        OUTPUT FORMAT (STRICT JSON):
        [
            {{
                "type": "scatter|boxplot|hist|correlation_matrix|bar|line",
                "x": "column_name",
                "y": "column_name",
                "description": "Deep business insight..."
            }}
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            raw_content = response.choices[0].message.content.strip()
            clean_json = raw_content.replace('```json', '').replace('```', '').strip()
            match = re.search(r'\[.*\]', clean_json, re.DOTALL)
            
            if not match:
                raise ValueError("No valid JSON found")

            return json.loads(match.group())
            
        except Exception as e:
            print(f"Visualizer Agent Error: {e}")
            return [
                {"type": "correlation_matrix", "description": "Analyzing variable relationships to identify key drivers."},
                {"type": "hist", "x": target, "description": f"Distribution analysis of {target}."}
            ]