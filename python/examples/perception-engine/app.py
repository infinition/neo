import streamlit as st
import torch
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import os

from models import CLIPEncoder
from index import load_index
from query import query_index
from extract import extract_clip

# Set page styling
st.set_page_config(
    page_title="Neo — Video Perception Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules matching our design guidelines
st.markdown("""
<style>
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    .css-1d391kg {
        background-color: #1f2833;
    }
    .stButton>button {
        background-color: #45f3ff;
        color: #0b0c10;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1f2833;
        color: #45f3ff;
        box-shadow: 0 0 10px #45f3ff;
    }
    .result-card {
        background-color: #1f2833;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #45f3ff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------- Caching Resource
@st.cache_resource
def get_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CLIPEncoder(model_name="openai/clip-vit-base-patch32", device=device)

# Load the encoder (runs once at startup)
encoder = get_encoder()

# ---------------------------------------------------- Sidebar UI
st.sidebar.title("🛠️ Configuration")

# Scan for index files (run the app from the repo root: indexes live in assets/indexes/)
neo_files = sorted(set(glob.glob("assets/indexes/*.neo") + glob.glob("*.neo")))
if not neo_files:
    st.sidebar.warning("Aucun index .neo trouvé dans les dossiers de ressources.")
    selected_index = st.sidebar.text_input("Chemin vers l'index (.neo)", "")
else:
    selected_index = st.sidebar.selectbox("Choisir l'index vidéo", neo_files)

# Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")
threshold = st.sidebar.slider("Seuil de similarité", 0.10, 0.40, 0.23, step=0.01)
max_gap = st.sidebar.slider("Écart max de fusion (s)", 1.0, 10.0, 3.0, step=0.5)
max_clips = st.sidebar.slider("Nombre max de clips à afficher", 1, 10, 3)

# ---------------------------------------------------- Main UI Header
st.title("👁️ Neo — GPU-Native Video Perception Engine")
st.write("Recherche sémantique instantanée dans le flux vidéo décodé et découpé sur GPU.")

if not selected_index or not Path(selected_index).exists():
    st.info("Veuillez sélectionner ou renseigner un fichier d'index `.neo` dans le panneau de configuration à gauche.")
else:
    index_path = Path(selected_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load index data (cached dynamically per index path change)
    index_data = load_index(index_path, device=device)
    embeddings = index_data["embeddings"]
    timestamps = index_data["timestamps"]
    video_file_name = index_data["video_name"]
    sample_fps = index_data["sample_fps"]
    
    st.success(f"Index chargé : **{video_file_name}** ({embeddings.shape[0]} frames indexées à {sample_fps} FPS)")

    # ---------------------------------------------------- Search Input
    query_text = st.text_input("🔍 De quoi vous rappelez-vous dans la vidéo ? (ex: 'living room', 'person', 'sofa', 'plant')", "living room")
    
    if query_text:
        # Perform query
        # 1. Encode text query to normalized vector
        text_emb = encoder.encode_text([query_text])
        
        # 2. Vector dot-product on GPU
        similarities = torch.matmul(embeddings, text_emb.t()).squeeze(-1)
        sims_np = similarities.cpu().numpy()
        ts_np = timestamps.cpu().numpy()
        
        # 3. Cluster into temporal clips
        clips = query_index(
            index_data=index_data,
            query_text=query_text,
            encoder=encoder,
            threshold=threshold,
            max_gap=max_gap
        )
        
        # ---------------------------------------------------- Similarity Timeline Chart
        st.subheader("📈 Ligne temporelle de similarité sémantique")
        
        chart_data = pd.DataFrame({
            "Temps (s)": ts_np,
            "Similarité sémantique": sims_np,
            "Seuil de détection": np.full_like(sims_np, threshold)
        }).set_index("Temps (s)")
        
        st.line_chart(chart_data)

        # ---------------------------------------------------- Query Results
        st.subheader(f"🎞️ Moments détectés ({len(clips)} correspondances)")
        
        if not clips:
            st.info("Aucun segment de la vidéo ne correspond à cette recherche avec le seuil actuel. Essayez d'abaisser le seuil dans la barre latérale.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Liste des correspondances")
                for idx, clip in enumerate(clips[:max_clips]):
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>#{idx+1} Segment détecté</h4>
                        <p>⏱️ <b>{clip['start']:.1f}s - {clip['end']:.1f}s</b> (Durée: {clip['end']-clip['start']:.1f}s)</p>
                        <p>🎯 Score de ressemblance : <b>{clip['max_similarity']:.3f}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Visionneuse d'extraits vidéo")
                # Dropdown to select which matching segment to play
                clip_choices = [f"#{idx+1} ({clip['start']:.1f}s - {clip['end']:.1f}s) - Score {clip['max_similarity']:.3f}" 
                                for idx, clip in enumerate(clips[:max_clips])]
                
                selected_clip_idx = st.selectbox("Choisir l'extrait à jouer", range(len(clip_choices)), format_func=lambda x: clip_choices[x])
                target_clip = clips[selected_clip_idx]
                
                # Try to resolve original video path
                video_path = None
                for candidate_dir in (Path("assets/videos"), index_path.parent, Path(".")):
                    test_path = candidate_dir / video_file_name
                    if test_path.exists():
                        video_path = test_path
                        break
                
                if not video_path:
                    st.warning(f"Vidéo source '{video_file_name}' introuvable. Veuillez renseigner le chemin complet ci-dessous pour extraire le clip.")
                    user_video_path = st.text_input("Chemin vers la vidéo d'origine", "")
                    if user_video_path and Path(user_video_path).exists():
                        video_path = Path(user_video_path)

                if video_path:
                    # Target clip output path
                    safe_query_name = "".join(c if c.isalnum() else "_" for c in query_text).strip("_")
                    output_name = f"clip_{selected_clip_idx+1}_{safe_query_name}_{target_clip['start']:.1f}s_to_{target_clip['end']:.1f}s.mp4"
                    output_path = Path("./clips") / output_name
                    
                    # If not already extracted, extract it
                    if not output_path.exists():
                        with st.spinner("Extraction de l'extrait en cours sur GPU (NVENC)..."):
                            success = extract_clip(
                                video_path=video_path,
                                start_time=target_clip['start'],
                                end_time=target_clip['end'],
                                output_path=output_path
                            )
                            if not success:
                                st.error("L'extraction du clip a échoué.")
                    
                    if output_path.exists():
                        # Play the extracted clip
                        st.video(str(output_path))
                        st.caption(f"Lecture de l'extrait : {output_name}")
                    else:
                        st.error("Impossible d'afficher la vidéo de l'extrait.")
