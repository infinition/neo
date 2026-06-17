use neo_core::format::{CodecId, ContainerFormat};
use neo_core::NeoResult;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Information about a single stream within a media file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    pub index: u32,
    pub stream_type: StreamType,
    pub codec: CodecId,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frame_rate: Option<(u32, u32)>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub bitrate: Option<u64>,
    pub duration: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamType {
    Video,
    Audio,
    Subtitle,
    Data,
}

/// Full media file information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaInfo {
    pub container: ContainerFormat,
    pub duration: Option<Duration>,
    pub bitrate: Option<u64>,
    pub streams: Vec<StreamInfo>,
}

impl MediaInfo {
    /// Get the first video stream.
    pub fn video_stream(&self) -> Option<&StreamInfo> {
        self.streams.iter().find(|s| s.stream_type == StreamType::Video)
    }

    /// Get the first audio stream.
    pub fn audio_stream(&self) -> Option<&StreamInfo> {
        self.streams.iter().find(|s| s.stream_type == StreamType::Audio)
    }
}

/// A compressed packet read from the container.
#[derive(Debug, Clone)]
pub struct Packet {
    pub stream_index: u32,
    pub data: Vec<u8>,
    pub pts: i64,
    pub dts: i64,
    pub duration: i64,
    pub keyframe: bool,
}

/// Demuxer — reads compressed packets from a container file.
///
/// The demuxer handles container parsing (MP4, MKV, etc.) and
/// produces compressed packets that are sent to the decoder.
pub trait Demuxer: Send {
    /// Get media information.
    fn info(&self) -> &MediaInfo;

    /// Read the next packet.
    fn read_packet(&mut self) -> NeoResult<Option<Packet>>;

    /// Seek to a timestamp.
    fn seek(&mut self, timestamp: Duration) -> NeoResult<()>;

    /// Reset to the beginning.
    fn reset(&mut self) -> NeoResult<()>;
}

/// Stub demuxer for development — produces synthetic packets.
pub struct StubDemuxer {
    info: MediaInfo,
    packet_count: u64,
    max_packets: u64,
}

impl StubDemuxer {
    pub fn new(width: u32, height: u32, fps: u32, duration_secs: f64) -> Self {
        let total_frames = (fps as f64 * duration_secs) as u64;
        Self {
            info: MediaInfo {
                container: ContainerFormat::Mp4,
                duration: Some(Duration::from_secs_f64(duration_secs)),
                bitrate: Some(10_000_000),
                streams: vec![StreamInfo {
                    index: 0,
                    stream_type: StreamType::Video,
                    codec: CodecId::H264,
                    width: Some(width),
                    height: Some(height),
                    frame_rate: Some((fps, 1)),
                    sample_rate: None,
                    channels: None,
                    bitrate: Some(10_000_000),
                    duration: Some(Duration::from_secs_f64(duration_secs)),
                }],
            },
            packet_count: 0,
            max_packets: total_frames,
        }
    }
}

impl Demuxer for StubDemuxer {
    fn info(&self) -> &MediaInfo {
        &self.info
    }

    fn read_packet(&mut self) -> NeoResult<Option<Packet>> {
        if self.packet_count >= self.max_packets {
            return Ok(None);
        }
        let pts = self.packet_count as i64;
        self.packet_count += 1;
        Ok(Some(Packet {
            stream_index: 0,
            data: vec![0u8; 4096], // Synthetic compressed data
            pts,
            dts: pts,
            duration: 1,
            keyframe: pts == 0 || pts % 30 == 0,
        }))
    }

    fn seek(&mut self, _timestamp: Duration) -> NeoResult<()> {
        Ok(())
    }

    fn reset(&mut self) -> NeoResult<()> {
        self.packet_count = 0;
        Ok(())
    }
}
