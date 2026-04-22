use neo_core::format::{CodecId, PixelFormat};
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// Hardware acceleration API to use for decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwAccelApi {
    /// NVIDIA NVDEC (Windows/Linux).
    Nvdec,
    /// Intel/AMD VAAPI (Linux).
    Vaapi,
    /// Apple VideoToolbox (macOS/iOS).
    VideoToolbox,
    /// Microsoft DirectX Video Acceleration (Windows).
    Dxva2,
    /// Software fallback (CPU — should be avoided).
    Software,
    /// Auto-detect best available.
    Auto,
}

/// Configuration for creating a decoder.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub codec: CodecId,
    pub width: u32,
    pub height: u32,
    pub hw_accel: HwAccelApi,
    /// Output pixel format (typically NV12 from hardware decoders).
    pub output_format: PixelFormat,
    /// Number of surfaces (decoded frame buffers) to allocate.
    pub surface_count: u32,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            codec: CodecId::H264,
            width: 1920,
            height: 1080,
            hw_accel: HwAccelApi::Auto,
            output_format: PixelFormat::Nv12,
            surface_count: 8,
        }
    }
}

/// The decoder trait — platform implementations provide the actual decoding.
///
/// All implementations MUST keep decoded frames in VRAM. The `GpuFrame`
/// returned should reference GPU buffers, not CPU memory.
pub trait Decoder: Send {
    /// Send a compressed packet to the decoder.
    fn send_packet(&mut self, data: &[u8], pts: i64, dts: i64) -> NeoResult<()>;

    /// Receive a decoded frame (if available).
    fn receive_frame(&mut self) -> NeoResult<Option<GpuFrame>>;

    /// Flush the decoder (signal end of stream).
    fn flush(&mut self) -> NeoResult<()>;

    /// Get the output pixel format.
    fn output_format(&self) -> PixelFormat;

    /// Get the hardware acceleration API in use.
    fn hw_accel(&self) -> HwAccelApi;
}

/// Software decoder fallback (CPU-based, for development/testing).
///
/// In production, this should never be in the hot path.
pub struct SoftwareDecoder {
    config: DecoderConfig,
    frame_counter: u64,
}

impl SoftwareDecoder {
    pub fn new(config: DecoderConfig) -> NeoResult<Self> {
        tracing::warn!(
            codec = %config.codec,
            "Using software decoder — performance will be degraded"
        );
        Ok(Self {
            config,
            frame_counter: 0,
        })
    }
}

impl Decoder for SoftwareDecoder {
    fn send_packet(&mut self, _data: &[u8], _pts: i64, _dts: i64) -> NeoResult<()> {
        // Software decode would happen here.
        // For now, this is a stub that produces empty frames.
        Ok(())
    }

    fn receive_frame(&mut self) -> NeoResult<Option<GpuFrame>> {
        use neo_core::color::ColorDesc;
        use neo_core::frame::FrameFlags;
        use neo_core::timestamp::Timestamp;

        let id = self.frame_counter;
        self.frame_counter += 1;

        let frame = GpuFrame {
            id,
            width: self.config.width,
            height: self.config.height,
            pixel_format: self.config.output_format,
            color: ColorDesc::default(),
            pts: Timestamp::new(id as i64, 1, 30),
            dts: Timestamp::new(id as i64, 1, 30),
            duration: Timestamp::new(1, 1, 30),
            planes: vec![],
            tensor_desc: None,
            flags: FrameFlags {
                keyframe: id == 0,
                ..Default::default()
            },
        };
        Ok(Some(frame))
    }

    fn flush(&mut self) -> NeoResult<()> {
        Ok(())
    }

    fn output_format(&self) -> PixelFormat {
        self.config.output_format
    }

    fn hw_accel(&self) -> HwAccelApi {
        HwAccelApi::Software
    }
}

/// Create the best available decoder for this platform.
pub fn create_decoder(config: DecoderConfig) -> NeoResult<Box<dyn Decoder>> {
    match config.hw_accel {
        HwAccelApi::Auto => {
            // TODO: Probe for NVDEC, then VAAPI, then VideoToolbox, then software.
            tracing::info!("Auto-detecting hardware decoder...");
            // For now, fall back to software.
            Ok(Box::new(SoftwareDecoder::new(config)?))
        }
        HwAccelApi::Software => Ok(Box::new(SoftwareDecoder::new(config)?)),
        other => {
            tracing::warn!(api = ?other, "Hardware decoder not yet implemented, falling back to software");
            Ok(Box::new(SoftwareDecoder::new(config)?))
        }
    }
}
