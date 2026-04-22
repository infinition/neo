use crate::encoder::EncodedPacket;
use neo_core::format::ContainerFormat;
use neo_core::NeoResult;
use std::path::PathBuf;

/// Muxer configuration.
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    pub container: ContainerFormat,
    pub output_path: PathBuf,
}

/// Muxer — writes encoded packets into a container file.
pub trait Muxer: Send {
    /// Write a packet to the output.
    fn write_packet(&mut self, packet: &EncodedPacket) -> NeoResult<()>;

    /// Finalize the container (write trailer, close file).
    fn finalize(&mut self) -> NeoResult<()>;
}

/// Stub muxer — writes raw packets to a file.
pub struct RawMuxer {
    config: MuxerConfig,
    buffer: Vec<u8>,
    packet_count: u64,
}

impl RawMuxer {
    pub fn new(config: MuxerConfig) -> NeoResult<Self> {
        Ok(Self {
            config,
            buffer: Vec::new(),
            packet_count: 0,
        })
    }
}

impl Muxer for RawMuxer {
    fn write_packet(&mut self, packet: &EncodedPacket) -> NeoResult<()> {
        self.buffer.extend_from_slice(&packet.data);
        self.packet_count += 1;
        Ok(())
    }

    fn finalize(&mut self) -> NeoResult<()> {
        std::fs::write(&self.config.output_path, &self.buffer)?;
        tracing::info!(
            path = %self.config.output_path.display(),
            packets = self.packet_count,
            bytes = self.buffer.len(),
            "Muxer finalized"
        );
        Ok(())
    }
}
