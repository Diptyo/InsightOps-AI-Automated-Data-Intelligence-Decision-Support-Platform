import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class DataJoinerAgent:
    def __init__(self):
        """Initialize with Groq client from environment."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def _get_metadata(self, df: pd.DataFrame, name: str):
        """Extracts minimal metadata for the LLM to save tokens."""
        return {
            "dataset_name": name,
            "columns": df.columns.tolist(),
            "sample": df.head(2).to_dict(),
            "shape": df.shape
        }

    def analyze_integration_strategy(self, dfs: list, names: list):
        """Uses AI to determine the most efficient multi-key join strategy."""
        metadata = [self._get_metadata(df, n) for df, n in zip(dfs, names)]
        
        prompt = f"""
        You are a Senior Data Architect. Analyze these {len(dfs)} datasets:
        {json.dumps(metadata, indent=2)}

        TASK: Determine how to combine these datasets into one Master Table.
        
        CRITICAL JOIN LOGIC:
        1. 'CONCAT': Only if columns are identical (stacking rows).
        2. 'MERGE': If datasets share common IDs. You MUST identify ALL common columns to prevent a "Many-to-Many" explosion.
           Example: If 'Store' and 'Date' appear in both, the join_keys MUST be ["Store", "Date"].
        3. 'NONE': If no logical link exists.

        Return ONLY a JSON object:
        {{
            "strategy": "MERGE" | "CONCAT" | "NONE",
            "join_keys": ["list", "of", "required", "keys"],
            "confidence_score": 0.0-1.0,
            "reason": "Explain why these specific keys prevent data duplication."
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"} 
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Joiner Agent Error: {e}")
            return {"strategy": "NONE", "join_keys": []}

    def combine(self, dfs: list, names: list):
        """Executes a memory-efficient join using the AI's strategy."""
        if not dfs: return None
        if len(dfs) == 1: return dfs[0]

        plan = self.analyze_integration_strategy(dfs, names)
        strategy = plan.get('strategy', 'NONE')
        keys = plan.get('join_keys', [])
        
        print(f"🤖 AI Join Strategy: {strategy}")
        print(f"🧐 Reason: {plan.get('reason', 'No reason provided')}")

        if strategy == "CONCAT":
            return pd.concat(dfs, ignore_index=True)
        
        elif strategy == "MERGE":
            # 1. Start with the 'Transaction/Fact' table (the one with the most rows)
            # This is vital for maintaining the integrity of the sales data.
            dfs_sorted = sorted(dfs, key=len, reverse=True)
            result_df = dfs_sorted[0]
            
            for i in range(1, len(dfs_sorted)):
                other_df = dfs_sorted[i]
                
                # 2. Identify keys that actually exist in BOTH dataframes
                valid_keys = [k for k in keys if k in result_df.columns and k in other_df.columns]
                
                if valid_keys:
                    print(f"🔗 Merging on: {valid_keys}")
                    # 3. Use 'left' join to keep all primary records and add secondary features
                    result_df = pd.merge(result_df, other_df, on=valid_keys, how="left")
                else:
                    # Fallback: Find any common columns if the AI's keys weren't found
                    common = list(set(result_df.columns) & set(other_df.columns))
                    if common:
                        print(f"🔄 AI keys failed. Falling back to common columns: {common}")
                        result_df = pd.merge(result_df, other_df, on=common, how="left")
            
            return result_df

        return dfs[0]