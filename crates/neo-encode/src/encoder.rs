use neo_core::format::{CodecId, PixelFormat};
use neo_core::frame::GpuFrame;
use neo_core::NeoResult;

/// Rate control mode for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControl {
    /// Constant bitrate.
    Cbr { bitrate: u64 },
    /// Variable bitrate.
    Vbr { target: u64, max: u64 },
    /// Constant quality (CRF/CQP).
    ConstantQuality { quality: u32 },
    /// Lossless.
    Lossless,
}

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub codec: CodecId,
    pub width: u32,
    pub height: u32,
    pub frame_rate: (u32, u32),
    pub pixel_format: PixelFormat,
    pub rate_control: RateControl,
    /// Encoding preset (0 = slowest/best quality, 10 = fastest/lowest quality).
    pub preset: u32,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// Use hardware encoder (NVENC, VideoToolbox, etc.).
    pub hw_encode: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: CodecId::H265,
            width: 1920,
            height: 1080,
            frame_rate: (30, 1),
            pixel_format: PixelFormat::Nv12,
            rate_control: RateControl::ConstantQuality { quality: 23 },
            preset: 5,
            gop_size: 60,
            hw_encode: true,
        }
    }
}

/// Encoded packet output.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    pub data: Vec<u8>,
    pub pts: i64,
    pub dts: i64,
    pub keyframe: bool,
    pub size: usize,
}

/// The encoder trait — encodes GpuFrames from VRAM into compressed bitstream.
pub trait Encoder: Send {
    /// Send a frame to the encoder (frame data is in VRAM).
    fn send_frame(&mut self, frame: &GpuFrame) -> NeoResult<()>;

    /// Receive an encoded packet (if available).
    fn receive_packet(&mut self) -> NeoResult<Option<EncodedPacket>>;

    /// Flush the encoder (signal end of stream, get remaining packets).
    fn flush(&mut self) -> NeoResult<()>;

    /// Get encoder info string.
    fn info(&self) -> String;
}

/// Software encoder fallback.
pub struct SoftwareEncoder {
    config: EncoderConfig,
    packet_counter: u64,
}

impl SoftwareEncoder {
    pub fn new(config: EncoderConfig) -> NeoResult<Self> {
        tracing::warn!("Using software encoder — consider enabling hardware encoding");
        Ok(Self {
            config,
            packet_counter: 0,
        })
    }
}

impl Encoder for SoftwareEncoder {
    fn send_frame(&mut self, _frame: &GpuFrame) -> NeoResult<()> {
        // Software encode would happen here
        self.packet_counter += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> NeoResult<Option<EncodedPacket>> {
        if self.packet_counter == 0 {
            return Ok(None);
        }
        self.packet_counter -= 1;
        Ok(Some(EncodedPacket {
            data: vec![0u8; 8192], // Synthetic encoded data
            pts: self.packet_counter as i64,
            dts: self.packet_counter as i64,
            keyframe: self.packet_counter == 0,
            size: 8192,
        }))
    }

    fn flush(&mut self) -> NeoResult<()> {
        Ok(())
    }

    fn info(&self) -> String {
        format!(
            "SoftwareEncoder({}, {}x{}, {:?})",
            self.config.codec, self.config.width, self.config.height, self.config.rate_control
        )
    }
}

/// Create the best available encoder.
pub fn create_encoder(config: EncoderConfig) -> NeoResult<Box<dyn Encoder>> {
    if config.hw_encode {
        tracing::info!("Hardware encoder not yet implemented, falling back to software");
    }
    Ok(Box::new(SoftwareEncoder::new(config)?))
}
