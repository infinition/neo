use thiserror::Error;

/// Unified error type for neo-moe.
#[derive(Error, Debug)]
pub enum MoeError {
    #[error("GGUF parse error: {0}")]
    GgufParse(String),

    #[error("Expert {layer}/{expert} not found in weight map")]
    ExpertNotFound { layer: u32, expert: u32 },

    #[error("CUDA error: {0:?}")]
    Cuda(cudarc::driver::DriverError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("VRAM pool exhausted — increase vram_resident_experts")]
    PoolExhausted,

    #[error("Expert size mismatch: expected {expected} bytes, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("Prefetch queue full — NVMe I/O cannot keep up with inference pace")]
    PrefetchQueueFull,

    #[error("Engine not initialised — call ExpertStream::init() first")]
    NotInitialised,

    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for MoeError {
    fn from(e: anyhow::Error) -> Self {
        MoeError::Other(e.to_string())
    }
}

impl From<cudarc::driver::DriverError> for MoeError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        MoeError::Cuda(e)
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, MoeError>;
