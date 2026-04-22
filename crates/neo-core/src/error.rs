use thiserror::Error;

pub type NeoResult<T> = Result<T, NeoError>;

#[derive(Debug, Error)]
pub enum NeoError {
    // GPU errors
    #[error("GPU device not found: {0}")]
    NoGpuDevice(String),
    #[error("GPU out of memory: requested {requested} bytes, available {available} bytes")]
    GpuOutOfMemory { requested: u64, available: u64 },
    #[error("GPU buffer error: {0}")]
    GpuBuffer(String),
    #[error("GPU compute error: {0}")]
    GpuCompute(String),

    // Codec errors
    #[error("unsupported codec: {0}")]
    UnsupportedCodec(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("encode error: {0}")]
    Encode(String),

    // Pipeline errors
    #[error("pipeline error: {0}")]
    Pipeline(String),
    #[error("node not found: {0}")]
    NodeNotFound(String),
    #[error("incompatible formats: {from} -> {to}")]
    IncompatibleFormats { from: String, to: String },

    // Inference errors
    #[error("model load error: {0}")]
    ModelLoad(String),
    #[error("inference error: {0}")]
    Inference(String),

    // I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("DirectStorage not available on this platform")]
    DirectStorageUnavailable,

    // Format errors
    #[error("unsupported pixel format: {0}")]
    UnsupportedPixelFormat(String),
    #[error("invalid frame dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    // Hardware acceleration errors
    #[error("hardware acceleration unavailable: {0}")]
    HwAccelUnavailable(String),
    #[error("CUDA error: {0}")]
    Cuda(String),
}
