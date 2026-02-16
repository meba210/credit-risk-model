import joblib

def save_model(model, file_path: str):
    """Save model to disk."""
    joblib.dump(model, file_path)

def load_model(file_path: str):
    """Load model from disk."""
    return joblib.load(file_path)
