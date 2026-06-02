//! neo-studio — standalone GUI binary.
//!
//! Launch with no arguments and import a clip from the UI:
//!   neo-studio
//!
//! Or pass a clip to open it immediately:
//!   neo-studio path\to\clip.h264
//!
//! Optional env: NEO_STUDIO_SHADERS (scratch dir for the active shader),
//! NEO_STUDIO_FPS (playback fps), NEO_STUDIO_NO_VSYNC=1.

use neo_lab::StudioOptions;
use std::path::PathBuf;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neo_lab=info,neo_hwaccel=info".into()),
        )
        .init();

    let input = std::env::args().nth(1).map(PathBuf::from);
    let shaders_dir = std::env::var("NEO_STUDIO_SHADERS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("shaders-studio"));
    let fps = std::env::var("NEO_STUDIO_FPS").ok().and_then(|s| s.parse().ok());
    let no_vsync = std::env::var("NEO_STUDIO_NO_VSYNC").is_ok();

    let opts = StudioOptions {
        input,
        shaders_dir,
        fps,
        no_vsync,
    };

    if let Err(e) = neo_lab::run_studio(opts) {
        eprintln!("neo-studio failed: {e}");
        std::process::exit(1);
    }
}
