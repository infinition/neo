use neo_core::{NeoError, NeoResult};
use std::path::{Path, PathBuf};

/// DirectStorage reader — bypasses CPU for NVMe → GPU transfers.
///
/// On Windows 11, uses the DirectStorage API.
/// On Linux, uses io_uring with registered buffers.
/// Falls back to mmap on unsupported platforms.
pub struct DirectStorageReader {
    path: PathBuf,
    available: bool,
}

impl DirectStorageReader {
    /// Try to create a DirectStorage reader for the given file.
    pub fn new(path: &Path) -> NeoResult<Self> {
        let available = Self::probe_support();
        if !available {
            tracing::warn!("DirectStorage not available, will use mmap fallback");
        }
        Ok(Self {
            path: path.to_path_buf(),
            available,
        })
    }

    /// Check if DirectStorage is supported on this platform.
    pub fn probe_support() -> bool {
        // TODO: Probe for Windows DirectStorage or Linux io_uring
        #[cfg(target_os = "windows")]
        {
            // Windows 11 with DirectStorage runtime
            false // Not yet implemented
        }
        #[cfg(target_os = "linux")]
        {
            // Check for io_uring support
            false // Not yet implemented
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            false
        }
    }

    /// Whether this reader is using true DirectStorage or a fallback.
    pub fn is_direct(&self) -> bool {
        self.available
    }

    /// Read a chunk of data. In the future, this writes directly to a GPU buffer.
    pub fn read_chunk(&self, offset: u64, size: usize) -> NeoResult<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; size];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Get file size.
    pub fn file_size(&self) -> NeoResult<u64> {
        Ok(std::fs::metadata(&self.path)?.len())
    }
}
