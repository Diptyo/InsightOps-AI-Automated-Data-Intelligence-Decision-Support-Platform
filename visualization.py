import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 🔥 GLOBAL STORE
VISUAL_DESCRIPTIONS = {}

# 🔥 EXECUTIVE STYLE SETTINGS
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

def normalize_type(v_type):
    if not v_type: return None
    v_type = v_type.lower().strip()
    mapping = {
        "scatter plot": "scatter", "scatter": "scatter",
        "bar chart": "bar", "bar": "bar",
        "line graph": "line", "line chart": "line",
        "histogram": "hist", "distribution": "hist", "hist": "hist",
        "heatmap": "correlation_matrix", "correlation heatmap": "correlation_matrix",
        "boxplot": "boxplot", "box plot": "boxplot"
    }
    return mapping.get(v_type, v_type)

def _add_title_and_subtitle(title, subtitle):
    """
    FIX: Using a single title with a newline (\n) ensures that 
    tight_layout treats both lines as one unit, preventing overlap.
    """
    plt.title(f"{title}\n{subtitle}", loc='left', weight='bold', fontsize=12, pad=20)

def execute_ai_viz_plan(df, plan):
    global VISUAL_DESCRIPTIONS
    VISUAL_DESCRIPTIONS = {}

    if not os.path.exists("visuals"):
        os.makedirs("visuals")

    for i, task in enumerate(plan):
        v_type = normalize_type(task.get('type'))
        x_col = task.get('x') or task.get('column')
        y_col = task.get('y') or task.get('target')
        description = task.get("description", "No insight available.")
        
        target_hue = y_col if v_type in ['scatter', 'boxplot'] else None
        file_path = f"visuals/{i+1}_{v_type}_{x_col}.png"
        
        plt.figure()

        try:
            # --- BOXPLOT ---
            if v_type == 'boxplot':
                sample_size = min(len(df), 10000)
                sns.boxplot(data=df.sample(sample_size), x=x_col, y=y_col, palette="viridis")
                plt.xticks(rotation=45)
                _add_title_and_subtitle(f"{x_col} Variance by {y_col}", "Visualizing class separation and outliers")

            # --- SCATTER ---
            elif v_type == 'scatter':
                sample_size = min(len(df), 5000)
                df_sample = df.sample(sample_size)
                sns.scatterplot(data=df_sample, x=x_col, y=y_col, hue=target_hue, alpha=0.6, palette="coolwarm")
                _add_title_and_subtitle(f"Correlation: {x_col} vs {y_col}", "Identifying clusters and trend directions")

            # --- BAR CHART ---
            elif v_type == 'bar':
                top_vals = df[x_col].value_counts().head(12)
                sns.barplot(x=top_vals.values, y=top_vals.index.astype(str), palette="magma")
                _add_title_and_subtitle(f"Volume Analysis: {x_col}", "Ranking the most frequent occurrences")

            # --- CORRELATION MATRIX ---
            elif v_type == 'correlation_matrix':
                numeric_df = df.select_dtypes(include=['number'])
                if numeric_df.shape[1] > 1:
                    corr = numeric_df.corr()
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
                    plt.xticks(rotation=45, ha='right')
                    _add_title_and_subtitle("Feature Interdependence", "Heatmap of numerical relationships")

            # --- HISTOGRAM ---
            elif v_type == 'hist':
                sns.histplot(df[x_col].dropna(), kde=True, color="#4e79a7")
                if df[x_col].max() > 10000: plt.yscale('log')
                _add_title_and_subtitle(f"Distribution of {x_col}", "Frequency and spread analysis")

            # --- LINE CHART ---
            elif v_type == 'line':
                sample = df.sort_values(x_col).head(5000)
                sns.lineplot(data=sample, x=x_col, y=y_col, color="#e15759")
                _add_title_and_subtitle(f"Timeline: {y_col} over {x_col}", "Sequential pattern and trend tracking")

            # 🔥 THE CRITICAL FIX: Reserve space at the top (top=0.88)
            plt.tight_layout(rect=[0, 0.03, 1, 0.88])
            plt.savefig(file_path)
            
            VISUAL_DESCRIPTIONS[os.path.basename(file_path)] = description
            print(f"📈 Chart Saved: {file_path}")

        except Exception as e:
            print(f"⚠️ Visualization Error ({v_type}): {e}")
        finally:
            plt.close()