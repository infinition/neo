use crate::demuxer::{MediaInfo, StreamInfo, StreamType};
use neo_core::format::{CodecId, ContainerFormat};
use neo_core::{NeoError, NeoResult};
use std::path::Path;
use std::time::Duration;

/// Probe a media file and return its metadata.
///
/// This is the equivalent of `ffprobe` — quickly reads the container
/// headers without decoding any frames.
pub fn probe_file(path: &Path) -> NeoResult<MediaInfo> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let container = match ext.as_str() {
        "mp4" | "m4v" => ContainerFormat::Mp4,
        "mkv" => ContainerFormat::Mkv,
        "webm" => ContainerFormat::Webm,
        "mov" => ContainerFormat::Mov,
        _ => {
            return Err(NeoError::UnsupportedCodec(format!(
                "unknown container: .{ext}"
            )))
        }
    };

    // TODO: Actually parse the container headers.
    // For now, return a placeholder based on file extension.
    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len();

    // Estimate: assume 10 Mbps bitrate for duration guess
    let estimated_duration = file_size as f64 / (10_000_000.0 / 8.0);

    Ok(MediaInfo {
        container,
        duration: Some(Duration::from_secs_f64(estimated_duration)),
        bitrate: Some(10_000_000),
        streams: vec![StreamInfo {
            index: 0,
            stream_type: StreamType::Video,
            codec: CodecId::H264,
            width: None,
            height: None,
            frame_rate: None,
            sample_rate: None,
            channels: None,
            bitrate: None,
            duration: Some(Duration::from_secs_f64(estimated_duration)),
        }],
    })
}
