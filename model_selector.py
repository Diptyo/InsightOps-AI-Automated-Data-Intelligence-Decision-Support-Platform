"""
Select best performing model from a dictionary of tuples.
Format: { "ModelName": (score, model_object) }
"""

def select_best_model(results):
    """
    Analyzes results and returns the name, the numerical score, 
    and the actual trained model object.
    """
    if not results:
        print("⚠️ No model results found to select from.")
        return None, 0, None

    # max() looks at the first element of the tuple (the score) to find the winner
    best_model_name = max(results, key=lambda k: results[k][0])
    
    # Unpack the winning tuple
    best_score, best_model_obj = results[best_model_name]

    print(f"🏆 Best Model Selected: {best_model_name} ({best_score:.4f})")
    
    return best_model_name, best_score, best_model_obj