//! TCP frame streaming: shared protocol + zero-copy NVDEC source, with
//! `send` / `recv` subcommands. `rife` and `filter-live` reuse the same
//! protocol and stream modules.

pub mod metrics;
pub mod protocol;
pub mod recv;
pub mod send;
pub use neo_hwaccel::zerocopy_stream;
