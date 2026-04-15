import pickle
import os
 
def load_model(path: str = "model/model_final.pkl"):
    """
    Load model pack dari file pickle.
    Return None jika file tidak ditemukan (tidak crash).
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
 
def save_model(model_pack: dict, path: str = "model/model_final.pkl"):
    """Simpan model pack ke file pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_pack, f)