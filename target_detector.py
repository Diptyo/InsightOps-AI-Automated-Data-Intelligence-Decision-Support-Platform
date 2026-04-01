import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class TargetDetectorAgent:
    def __init__(self):
        """Initialize the Groq client using the secured API key."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
            
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def _get_system_prompt(self):
        """The core instructions for the AI's reasoning."""
        return """
        You are an Expert Data Architect and ML Strategy Agent. 
        Your goal is to analyze a dataset's metadata and determine the most logical 'Target Variable'.
        
        Rules for selection:
        1. Identify the 'Outcome': The variable that is a result of other features.
        2. Avoid Metadata: Do not pick 'ID', 'Name', 'Timestamp', or 'Notes'.
        3. Statistical Value: The column must have predictive importance.
        4. Data Integrity: Ignore columns with zero variance or status flags that don't change.
        
        Output Format:
        Return ONLY the exact string of the column name. No explanation, no quotes.
        """

    def detect(self, df: pd.DataFrame):
        """Perform semantic analysis to find the target variable."""
        try:
            column_names = df.columns.tolist()
            context = {
                "columns": column_names,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample_data": df.head(3).to_dict()
            }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"Analyze this dataset context and pick the target: {context}"}
                ],
                temperature=0.1
            )

            raw_output = response.choices[0].message.content.strip()
            detected_target = raw_output.replace('"', '').replace("'", "").strip().strip('.')

            # Validation Layer
            if detected_target in column_names:
                # Check for variance and actual content
                if df[detected_target].nunique() > 1 and df[detected_target].notnull().sum() > 0:
                    print(f"✅ AI detected target: {detected_target}")
                    return detected_target
            
            print(f"⚠️ Agent suggested '{detected_target}', but it failed validation. Using fallback.")
            return self._heuristic_fallback(df)
            
        except Exception as e:
            print(f"❌ Target Detector Error: {e}")
            return self._heuristic_fallback(df)

    def _heuristic_fallback(self, df: pd.DataFrame):
        """If AI fails, pick the last numeric column (statistically likely to be a target)."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            return numeric_cols[-1]
        return df.columns[-1]

    def get_problem_type(self, df: pd.DataFrame, target: str):
        """
        Determines if the task is Regression or Classification.
        This is now a top-level method of the class.
        """
        if target not in df.columns:
            return 'regression'
            
        unique_count = df[target].nunique()
        data_type = df[target].dtype

        # 1. Categorical data is always classification
        if data_type == 'object' or data_type == 'category' or data_type == 'bool':
            return 'classification'
        
        # 2. Heuristic: Low unique count in numeric data usually implies categories/classes
        if unique_count < 20: 
            return 'classification'
        
        return 'regression'