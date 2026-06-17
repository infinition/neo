//! Subcommand implementations. Each module was a standalone demo binary
//! before the workspace unification; the pipeline code is unchanged, only
//! the clap entry points were converted to `Args` + `run()`.

pub mod desktop;
pub mod filter_live;
pub mod infer_bench;
pub mod mosaic;
pub mod rife;
pub mod stream;
