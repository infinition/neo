//! Wire protocol for neo-stream.
//!
//! Each frame is sent as a header followed by the H.264 NAL payload.
//! The header carries a sender-side timestamp so the receiver can compute
//! one-way latency (accurate on the same machine, approximate across hosts).
//!
//! ```text
//! ┌──────────┬───────────┬──────────────────┬──────────────┐
//! │ magic(4) │ frame(4)  │ timestamp_us(8)  │ payload(4)   │
//! │ "NEOS"   │ u32 LE    │ u64 LE           │ u32 LE       │
//! └──────────┴───────────┴──────────────────┴──────────────┘
//!                          20 bytes total
//! ```

use std::io::{self, Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAGIC: u32 = 0x4E454F53; // "NEOS"
#[allow(dead_code)]
pub const HEADER_SIZE: usize = 20;

#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub frame_num: u32,
    pub timestamp_us: u64,
    pub payload_len: u32,
}

impl FrameHeader {
    pub fn new(frame_num: u32, payload_len: u32) -> Self {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        Self {
            frame_num,
            timestamp_us,
            payload_len,
        }
    }

    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&MAGIC.to_le_bytes())?;
        w.write_all(&self.frame_num.to_le_bytes())?;
        w.write_all(&self.timestamp_us.to_le_bytes())?;
        w.write_all(&self.payload_len.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> io::Result<Self> {
        let mut buf = [0u8; HEADER_SIZE];
        r.read_exact(&mut buf)?;
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad magic: 0x{magic:08X}"),
            ));
        }
        let frame_num = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let timestamp_us = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let payload_len = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        Ok(Self {
            frame_num,
            timestamp_us,
            payload_len,
        })
    }
}

/// Current timestamp in microseconds since epoch.
pub fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
