import torch

def save_index(index_path, embeddings, timestamps, video_name, sample_fps):
    """
    Saves the VRAM index (tensors and metadata) to disk.
    Tensors are moved to CPU during serialization for safety and compatibility.
    """
    data = {
        "embeddings": embeddings.cpu(),
        "timestamps": timestamps.cpu(),
        "video_name": video_name,
        "sample_fps": sample_fps
    }
    torch.save(data, index_path)
    print(f"[index] Index sauvegardé dans {index_path} ({embeddings.shape[0]} frames indexées)")

def load_index(index_path, device="cuda"):
    """
    Loads the index from disk and transfers tensors back to VRAM.
    """
    data = torch.load(index_path, map_location=device)
    embeddings = data["embeddings"].to(device)
    timestamps = data["timestamps"].to(device)
    print(f"[index] Index chargé : {data['video_name']} ({embeddings.shape[0]} frames, {data['sample_fps']} FPS)")
    return {
        "embeddings": embeddings,
        "timestamps": timestamps,
        "video_name": data["video_name"],
        "sample_fps": data["sample_fps"]
    }
