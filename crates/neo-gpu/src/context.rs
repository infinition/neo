use neo_core::{NeoError, NeoResult};
use std::sync::Arc;
use tracing::info;

/// The GPU context — a single connection to a GPU device.
///
/// This is created once at startup and shared across the entire pipeline.
/// All VRAM operations go through this context.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.adapter_info.name)
            .field("backend", &self.adapter_info.backend)
            .finish()
    }
}

/// Options for GPU device selection.
#[derive(Debug, Clone)]
pub struct GpuOptions {
    /// Preferred backend (Vulkan, Metal, DX12, etc.). None = auto-detect.
    pub backend: Option<wgpu::Backends>,
    /// Preferred GPU power profile.
    pub power_preference: wgpu::PowerPreference,
    /// Required features.
    pub features: wgpu::Features,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            backend: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            features: wgpu::Features::empty(),
        }
    }
}

impl GpuOptions {
    /// Preset for the Stage 6 CUDA↔Vulkan interop pipeline: forces the
    /// Vulkan backend and requests the `VULKAN_EXTERNAL_MEMORY_WIN32`
    /// feature so shared memory handles can be exported from Vulkan and
    /// imported into CUDA.
    pub fn interop() -> Self {
        Self {
            backend: Some(wgpu::Backends::VULKAN),
            power_preference: wgpu::PowerPreference::HighPerformance,
            features: wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32,
        }
    }
}

impl GpuContext {
    /// Initialize the GPU context.
    ///
    /// Discovers the best available GPU and creates a device + queue.
    pub async fn new(options: &GpuOptions) -> NeoResult<Self> {
        let backends = options.backend.unwrap_or(wgpu::Backends::all());
        // wgpu 29: InstanceDescriptor lost its `Default` impl; we start from
        // the explicit `new_without_display_handle` constructor and override
        // the backends field.
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = backends;
        let instance = wgpu::Instance::new(instance_desc);

        // wgpu 29: request_adapter returns Result instead of Option.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| NeoError::NoGpuDevice(format!("no compatible GPU adapter: {e}")))?;

        let adapter_info = adapter.get_info();
        info!(
            gpu = %adapter_info.name,
            backend = ?adapter_info.backend,
            "GPU adapter selected"
        );

        let limits = adapter.limits();

        // wgpu 29: request_device takes a single DeviceDescriptor; the old
        // trace-path argument is gone.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("neo-ffmpeg"),
                required_features: options.features,
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| NeoError::NoGpuDevice(format!("failed to create device: {e}")))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits,
        })
    }

    /// Create a synchronous context (convenience wrapper).
    pub fn new_sync(options: &GpuOptions) -> NeoResult<Self> {
        pollster::block_on(Self::new(options))
    }

    /// Get the maximum buffer size supported by this GPU.
    pub fn max_buffer_size(&self) -> u64 {
        self.limits.max_buffer_size
    }

    /// Get the GPU name.
    pub fn gpu_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get the backend (Vulkan, Metal, DX12, etc.).
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }
}
