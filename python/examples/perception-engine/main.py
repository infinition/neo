import argparse
import sys
from pathlib import Path
import torch

from models import CLIPEncoder
from ingest import ingest_video
from index import save_index, load_index
from query import query_index
from extract import extract_clip

def do_ingest(args):
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        sys.exit("Erreur : CUDA n'est pas disponible. Ce moteur de perception exige un GPU NVIDIA.")

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Erreur : Le fichier vidéo '{video_path}' n'existe pas.")

    # Determine index output path
    if args.index_out:
        index_path = Path(args.index_out)
    else:
        index_path = video_path.with_suffix(".neo")

    # Load CLIP encoder
    encoder = CLIPEncoder(model_name=args.model, device="cuda")

    # Run ingestion
    embeddings, timestamps = ingest_video(
        video_path=video_path,
        encoder=encoder,
        mode=args.mode,
        sample_fps=args.sample_fps,
        batch_size=args.batch_size
    )

    # Save index
    save_index(
        index_path=index_path,
        embeddings=embeddings,
        timestamps=timestamps,
        video_name=video_path.name,
        sample_fps=args.sample_fps
    )

def do_query(args):
    index_path = Path(args.index)
    if not index_path.exists():
        sys.exit(f"Erreur : Le fichier d'index '{index_path}' n'existe pas.")

    # Ensure CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Warning] CUDA indisponible, exécution sur CPU.")

    # Load CLIP encoder
    encoder = CLIPEncoder(model_name=args.model, device=device)

    # Load index
    index_data = load_index(index_path, device=device)

    # Perform query
    clips = query_index(
        index_data=index_data,
        query_text=args.text,
        encoder=encoder,
        threshold=args.threshold,
        max_gap=args.max_gap
    )

    # Print results
    print(f"\n========== RÉSULTATS DE RECHERCHE (Query: '{args.text}') ==========")
    if not clips:
        print("Aucun segment correspondant trouvé.")
    else:
        for idx, clip in enumerate(clips):
            print(f"[{idx+1}] Temps : {clip['start']:.1f}s - {clip['end']:.1f}s | Score : {clip['max_similarity']:.3f}")
    print("==================================================================\n")

    # Optional clip extraction
    if args.extract and clips:
        # Resolve source video file path
        video_file_name = index_data["video_name"]
        video_path = None
        
        if args.video:
            video_path = Path(args.video)
        else:
            # Check if video is in the same directory as the index
            test_path = index_path.parent / video_file_name
            if test_path.exists():
                video_path = test_path
            else:
                # Check current directory
                test_path = Path(".") / video_file_name
                if test_path.exists():
                    video_path = test_path

        if not video_path or not video_path.exists():
            print(f"[Warning] Impossible d'extraire les clips : vidéo source '{video_file_name}' introuvable. "
                  f"Veuillez spécifier le chemin avec --video.")
            return

        print(f"[extract] Début de l'extraction des clips (max {args.max_clips})...")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Safe text for filename
        safe_query_name = "".join(c if c.isalnum() else "_" for c in args.text).strip("_")
        
        for idx, clip in enumerate(clips[:args.max_clips]):
            output_name = f"clip_{idx+1}_{safe_query_name}_{clip['start']:.1f}s_to_{clip['end']:.1f}s.mp4"
            output_path = out_dir / output_name
            extract_clip(
                video_path=video_path,
                start_time=clip['start'],
                end_time=clip['end'],
                output_path=output_path
            )

def main():
    parser = argparse.ArgumentParser(
        description="Neo — GPU-Native Video Perception Engine (Phase 1 MVP)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: Ingest
    parser_ingest = subparsers.add_parser("ingest", help="Indexe une vidéo en VRAM et sauvegarde l'index")
    parser_ingest.add_argument("--video", required=True, help="Chemin vers la vidéo à indexer")
    parser_ingest.add_argument("--index-out", help="Chemin de sortie de l'index (défaut: même nom avec extension .neo)")
    parser_ingest.add_argument("--mode", choices=["neo", "baseline"], default="neo", help="Mode d'ingestion (neo = zero-copy VRAM)")
    parser_ingest.add_argument("--sample-fps", type=float, default=1.0, help="Nombre d'images à indexer par seconde de vidéo (défaut: 1.0)")
    parser_ingest.add_argument("--batch-size", type=int, default=8, help="Taille des batchs d'inférence GPU (défaut: 8)")
    parser_ingest.add_argument("--model", default="openai/clip-vit-base-patch32", help="Modèle CLIP à utiliser (défaut: openai/clip-vit-base-patch32)")

    # Sub-command: Query
    parser_query = subparsers.add_parser("query", help="Recherche sémantique sur une vidéo indexée")
    parser_query.add_argument("--index", required=True, help="Chemin vers le fichier d'index .neo")
    parser_query.add_argument("--text", required=True, help="Texte de recherche sémantique")
    parser_query.add_argument("--threshold", type=float, default=0.22, help="Seuil de similarité cosinus minimum (défaut: 0.22)")
    parser_query.add_argument("--max-gap", type=float, default=3.0, help="Écart max en secondes pour regrouper les frames en un segment (défaut: 3.0)")
    parser_query.add_argument("--model", default="openai/clip-vit-base-patch32", help="Modèle CLIP à utiliser (défaut: openai/clip-vit-base-patch32)")
    
    # Clip extraction arguments
    parser_query.add_argument("--extract", action="store_true", help="Active l'extraction vidéo des segments correspondants")
    parser_query.add_argument("--video", help="Optionnel: Chemin vers la vidéo source d'origine si non trouvée automatiquement")
    parser_query.add_argument("--out-dir", default="./clips", help="Dossier de sortie des clips extraits (défaut: ./clips)")
    parser_query.add_argument("--max-clips", type=int, default=3, help="Nombre maximal de clips à extraire (défaut: 3)")

    args = parser.parse_args()

    if args.command == "ingest":
        do_ingest(args)
    elif args.command == "query":
        do_query(args)

if __name__ == "__main__":
    main()
