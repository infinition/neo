import torch
import numpy as np

def get_matching_clips(timestamps, similarities, threshold=0.22, max_gap=3.0):
    """
    Clusters matching frame timestamps into continuous video segments.
    Groups frames together if they are separated by less than `max_gap` seconds.
    """
    mask = similarities >= threshold
    matching_indices = torch.where(mask)[0]
    
    if len(matching_indices) == 0:
        return []
    
    matching_timestamps = timestamps[matching_indices].cpu().numpy()
    matching_sims = similarities[matching_indices].cpu().numpy()
    
    clips = []
    current_start = matching_timestamps[0]
    current_end = matching_timestamps[0]
    max_sim = matching_sims[0]
    
    for t, sim in zip(matching_timestamps[1:], matching_sims[1:]):
        if t - current_end <= max_gap:
            # Extend current segment
            current_end = t
            max_sim = max(max_sim, sim)
        else:
            # Store completed segment
            clips.append({
                "start": float(current_start),
                "end": float(current_end),
                "max_similarity": float(max_sim)
            })
            current_start = t
            current_end = t
            max_sim = sim
            
    # Add final segment
    clips.append({
        "start": float(current_start),
        "end": float(current_end),
        "max_similarity": float(max_sim)
    })
    
    # Sort segments by maximum relevance score (similarity)
    clips = sorted(clips, key=lambda x: x["max_similarity"], reverse=True)
    return clips


def query_index(index_data, query_text, encoder, threshold=0.22, max_gap=3.0):
    """
    Performs similarity search on GPU and returns matching clips.
    """
    # 1. Encode text query to normalized vector in VRAM
    text_emb = encoder.encode_text([query_text])  # shape (1, 512)
    
    embeddings = index_data["embeddings"]  # (N, 512)
    timestamps = index_data["timestamps"]  # (N,)
    
    # 2. Perform matrix multiplication for instant cosine similarity
    similarities = torch.matmul(embeddings, text_emb.t()).squeeze(-1)  # (N,)
    
    # 3. Group frames and build temporal segments
    clips = get_matching_clips(timestamps, similarities, threshold, max_gap)
    return clips
