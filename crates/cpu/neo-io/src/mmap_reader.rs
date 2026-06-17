use neo_core::{NeoError, NeoResult};
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Memory-mapped file reader — zero-copy reads via the OS virtual memory system.
///
/// This is the fallback when DirectStorage is not available.
/// The OS maps the file into the process address space, and the GPU
/// can DMA from these pages with minimal CPU involvement.
pub struct MmapReader {
    path: PathBuf,
    mmap: Mmap,
    size: u64,
}

impl MmapReader {
    /// Open a file with memory mapping.
    pub fn open(path: &Path) -> NeoResult<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let size = metadata.len();

        // SAFETY: We're only reading. The file should not be modified while mapped.
        let mmap = unsafe { Mmap::map(&file)? };

        tracing::debug!(
            path = %path.display(),
            size_mb = size / (1024 * 1024),
            "Memory-mapped file opened"
        );

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            size,
        })
    }

    /// Get a slice of the mapped data.
    pub fn slice(&self, offset: u64, len: usize) -> &[u8] {
        let start = offset as usize;
        let end = (start + len).min(self.mmap.len());
        &self.mmap[start..end]
    }

    /// Get the entire mapped data.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// File size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Advise the OS to prefetch a range (for sequential reading).
    pub fn advise_sequential(&self) {
        // On Linux, this would call madvise(MADV_SEQUENTIAL).
        // On Windows, this would call PrefetchVirtualMemory.
        // For now, no-op.
    }
}
