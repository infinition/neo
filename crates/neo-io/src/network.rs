use neo_core::{NeoError, NeoResult};
use std::net::SocketAddr;

/// Network protocol for streaming input/output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamProtocol {
    /// RTMP (Twitch, YouTube Live ingest).
    Rtmp,
    /// SRT (Secure Reliable Transport — low latency).
    Srt,
    /// WebRTC (browser-to-browser, ultra low latency).
    WebRtc,
    /// Raw TCP stream.
    Tcp,
    /// Raw UDP stream.
    Udp,
}

/// Network source — receives a live video stream from the network.
///
/// The received data is pushed into the pipeline as compressed packets,
/// which are then decoded directly into VRAM.
pub struct NetworkSource {
    protocol: StreamProtocol,
    addr: SocketAddr,
    connected: bool,
}

impl NetworkSource {
    /// Create a new network source listener.
    pub fn new(protocol: StreamProtocol, addr: SocketAddr) -> Self {
        Self {
            protocol,
            addr,
            connected: false,
        }
    }

    /// Start listening for incoming connections.
    pub async fn listen(&mut self) -> NeoResult<()> {
        tracing::info!(
            protocol = ?self.protocol,
            addr = %self.addr,
            "Network source listening"
        );
        // TODO: Implement actual protocol handling
        self.connected = true;
        Ok(())
    }

    /// Read the next chunk of compressed data from the network.
    pub async fn read_chunk(&mut self) -> NeoResult<Option<Vec<u8>>> {
        if !self.connected {
            return Err(NeoError::Io(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "not connected",
            )));
        }
        // TODO: Implement actual network read
        Ok(None)
    }

    pub fn protocol(&self) -> StreamProtocol {
        self.protocol
    }
}

/// Network sink — sends encoded video to a streaming server.
pub struct NetworkSink {
    protocol: StreamProtocol,
    url: String,
    connected: bool,
}

impl NetworkSink {
    pub fn new(protocol: StreamProtocol, url: &str) -> Self {
        Self {
            protocol,
            url: url.to_string(),
            connected: false,
        }
    }

    /// Connect to the streaming server.
    pub async fn connect(&mut self) -> NeoResult<()> {
        tracing::info!(
            protocol = ?self.protocol,
            url = %self.url,
            "Connecting to streaming server"
        );
        // TODO: Implement actual connection
        self.connected = true;
        Ok(())
    }

    /// Send encoded data to the server.
    pub async fn send(&mut self, data: &[u8]) -> NeoResult<()> {
        if !self.connected {
            return Err(NeoError::Io(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "not connected",
            )));
        }
        // TODO: Implement actual send
        Ok(())
    }
}
