import sys
from pathlib import Path

import torch

# The perception-engine example modules are not a package; add them to the path.
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "perception-engine"))
from models import CLIPEncoder
from index import load_index

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = CLIPEncoder(device=device)
index_data = load_index("assets/indexes/demo.neo", device=device)

embeddings = index_data["embeddings"]

queries = ["living room", "kitchen", "sofa", "television", "submarine", "airplane", "snow"]

for q in queries:
    text_emb = encoder.encode_text([q])
    sims = torch.matmul(embeddings, text_emb.t()).squeeze(-1)
    print(f"Query: '{q}'")
    print(f"  Min: {sims.min().item():.3f} | Max: {sims.max().item():.3f} | Mean: {sims.mean().item():.3f} | Std: {sims.std().item():.3f}")
