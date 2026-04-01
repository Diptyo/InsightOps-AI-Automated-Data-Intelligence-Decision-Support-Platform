import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
from dotenv import load_dotenv

# Specialized Tools
from tools.data_joiner import DataJoinerAgent
from tools.preprocessing_agent import PreprocessingAgent 
from tools.target_detector import TargetDetectorAgent
from tools.visualizer_agent import VisualizerAgent
from tools.visualization import execute_ai_viz_plan
import tools.visualization as viz_module
from tools.automl_pipeline import run_pipeline 
from agents.analyst_agent import generate_full_report_narrative
from tools.report_generator import ReportGenerator 

load_dotenv()

class AIProjectEngine:
    def __init__(self):
        self.joiner = DataJoinerAgent()
        self.preprocessor = PreprocessingAgent()
        self.detector = TargetDetectorAgent()
        self.viz_planner = VisualizerAgent()
        self.reporter = ReportGenerator() 
        
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def _log_data_snapshot(self, message, df, log_filename):
        with open(f"logs/{log_filename}.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"EVENT: {message}\n")
            f.write(f"{'='*60}\n")
            buffer = StringIO()
            df.info(buf=buffer)
            f.write(buffer.getvalue())
            f.write("\nDATA PREVIEW (TOP 5 ROWS):\n")
            f.write(df.head(5).to_string())
            f.write("\n\n")

    def run_engine_web(self, file_paths, manual_target=None, session_id=None, progress_callback=None):

        self._clear_session_visuals(session_id)
        
        def update_progress(status):
            if progress_callback:
                progress_callback(status)

        log_name = session_id if session_id else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. LOAD DATA
        update_progress("Loading Uploaded Files...")
        datasets = [pd.read_csv(p, low_memory=False) for p in file_paths]
        filenames = [os.path.basename(p) for p in file_paths]
        
        # 2. JOIN DATASETS
        update_progress("Joining Datasets...")
        master_df = self.joiner.combine(datasets, filenames)

        # 3. AI TARGET & PROBLEM DETECTION
        # We detect the target and the type (Regression/Classification) from the joined data
        target = manual_target if manual_target in master_df.columns else self.detector.detect(master_df)
        
        if not target:
            update_progress("Error: No valid target detected.")
            return None
            
        problem_type = self.detector.get_problem_type(master_df, target)
        print(f"DEBUG: Target='{target}', Type='{problem_type}'")

        # 4. PREPROCESS & CLEAN
        # Now we clean and handle the label encoding for the specific target
        update_progress("Cleaning and Optimizing Data...")
        master_df = self.preprocessor.process(master_df, target_col=target)

        # 5. ML PIPELINE
        update_progress(f"Executing AutoML ({problem_type}) for '{target}'...")
        eda_info = self.get_eda_info(master_df)
        
        # IMPORTANT: Pass problem_type to your pipeline
        ml_results = run_pipeline(master_df, target, problem_type=problem_type)
        ml_results['target'] = target
        ml_results['problem_type'] = problem_type

        # 6. DYNAMIC VISUALIZATION
        update_progress("AI is identifying essential data visualizations...")
        viz_plan = self.viz_planner.plan_visualizations(
            master_df, 
            target, 
            ml_results.get('feature_importance_dict'),
            problem_type=problem_type  # <--- Pass the new variable here
        )
        execute_ai_viz_plan(master_df, viz_plan)
        
        # 7. STRATEGIC NARRATIVE
        update_progress("Drafting Executive Strategy Narrative...")
        analysis_narrative_json = generate_full_report_narrative(
            results=ml_results,
            eda_info=eda_info,
            feature_importance=ml_results.get("feature_importance_dict")
        )
        
        # 8. FINAL REPORT GENERATION
        update_progress("Finalizing Professional PDF Report...")
        report_path = self.reporter.generate_document(
            ml_results=ml_results,
            ai_narrative=analysis_narrative_json,
            session_id=log_name,
            visual_description=viz_module.VISUAL_DESCRIPTIONS
        )
        
        update_progress("Complete")

        # --- UPDATED RETURN FOR WEB VIEW ---
        return {
            "pdf_filename": os.path.basename(report_path),
            "narrative": analysis_narrative_json,
            "visuals": viz_module.VISUAL_DESCRIPTIONS,
            "session_id": log_name,
            "target": target,
            "score": ml_results.get('score', 0),
            "best_model": ml_results.get('best_model', 'Unknown Engine'),
            "metrics": ml_results.get('metrics', {})
        }

    def get_eda_info(self, df):
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty: return {'high_correlation_pairs': []}
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [f"{r} & {c}" for c in upper.columns for r in upper.index if upper.loc[r, c] > 0.9]
        return {'high_correlation_pairs': high_corr, 'stats': df.describe().to_dict()}
    
    def _clear_session_visuals(self, session_id):
        """Removes any existing images for this session to prevent duplicates."""
        if not os.path.exists("visuals"):
            os.makedirs("visuals")
            return
            
        # Find all files in 'visuals/' that start with the session_id or are leftover PNGs
        # Note: If you want a global wipe, use glob.glob("visuals/*.png")
        files = glob.glob(f"visuals/*.png") 
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Could not delete old chart {f}: {e}")

if __name__ == "__main__":
    engine = AIProjectEngine()
    # Ensure you pass a LIST of actual CSV paths
    engine.run_engine_web(["your_data.csv"])