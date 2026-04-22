//! Live-reloadable WGSL compute-shader chain.
//!
//! ## Wire format
//!
//! Each user shader is a single `.wgsl` file under the watched directory.
//! The file must declare exactly the bind group layout that Neo Lab
//! provides:
//!
//! ```wgsl
//! struct Dims { width: u32, height: u32 };
//! @group(0) @binding(0) var<uniform> dims: Dims;
//! @group(0) @binding(1) var<storage, read>        in_buf:  array<u32>;
//! @group(0) @binding(2) var<storage, read_write>  out_buf: array<u32>;
//!
//! @compute @workgroup_size(8, 8, 1)
//! fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
//!     let x = gid.x;
//!     let y = gid.y;
//!     if (x >= dims.width || y >= dims.height) { return; }
//!     let idx = y * dims.width + x;
//!     let pixel = in_buf[idx];
//!     // unpack BGRA, do stuff, repack
//!     out_buf[idx] = pixel;
//! }
//! ```
//!
//! Pixels are packed `B | G<<8 | R<<16 | A<<24` little-endian (matches the
//! NV12→BGRA converter output and NVENC's `NV_ENC_BUFFER_FORMAT_ARGB`).
//!
//! ## Hot reload
//!
//! On every `poll_reload` the chain checks the `notify` channel for any
//! file event under the shaders directory. On change it re-globs the
//! folder, recompiles each `.wgsl`, and atomically swaps the active node
//! list. If a shader fails to compile, the previous version is kept and
//! the error is exposed via [`ShaderChain::last_error`] without aborting
//! anything.

use bytemuck::{Pod, Zeroable};
use notify::{Event, RecursiveMode, Watcher};
use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
};
use tracing::{debug, error, info, warn};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Dims {
    width: u32,
    height: u32,
}

/// One compiled user shader.
struct ShaderNode {
    name: String,
    pipeline: wgpu::ComputePipeline,
}

/// Output buffer parity after running the chain — tells the caller which
/// of the two ping-pong buffers holds the final result.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FinalBuffer {
    A,
    B,
}

pub struct ShaderChain {
    device: Arc<wgpu::Device>,
    width: u32,
    height: u32,

    /// Two BGRA storage buffers, identical layout, used as ping-pong.
    pub buf_a: wgpu::Buffer,
    pub buf_b: wgpu::Buffer,

    dims_buf: wgpu::Buffer,
    bind_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,

    /// Pre-built bind groups for both directions, reused across all
    /// shaders that match the standard layout.
    bind_a_to_b: wgpu::BindGroup,
    bind_b_to_a: wgpu::BindGroup,

    nodes: Vec<ShaderNode>,
    last_error: Option<String>,

    shaders_dir: PathBuf,
    watcher: notify::RecommendedWatcher,
    rx: mpsc::Receiver<notify::Result<Event>>,
}

impl ShaderChain {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        shaders_dir: PathBuf,
    ) -> Result<Self, String> {
        let bgra_size = (width as u64) * (height as u64) * 4;

        let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lab-bgra-a"),
            size: bgra_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lab-bgra-b"),
            size: bgra_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dims_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lab-dims"),
            size: std::mem::size_of::<Dims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dims_buf, 0, bytemuck::bytes_of(&Dims { width, height }));

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lab-chain-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lab-chain-pl"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let bind_a_to_b = build_bind_group(&device, &bind_layout, &dims_buf, &buf_a, &buf_b);
        let bind_b_to_a = build_bind_group(&device, &bind_layout, &dims_buf, &buf_b, &buf_a);

        // Set up the file watcher.
        let (tx, rx) = mpsc::channel();
        let mut watcher: notify::RecommendedWatcher = notify::recommended_watcher(
            move |res: notify::Result<Event>| {
                let _ = tx.send(res);
            },
        )
        .map_err(|e| format!("notify watcher init: {e}"))?;
        std::fs::create_dir_all(&shaders_dir)
            .map_err(|e| format!("create shaders dir {:?}: {e}", shaders_dir))?;
        watcher
            .watch(&shaders_dir, RecursiveMode::NonRecursive)
            .map_err(|e| format!("watch {:?}: {e}", shaders_dir))?;

        let mut chain = Self {
            device,
            width,
            height,
            buf_a,
            buf_b,
            dims_buf,
            bind_layout,
            pipeline_layout,
            bind_a_to_b,
            bind_b_to_a,
            nodes: Vec::new(),
            last_error: None,
            shaders_dir,
            watcher,
            rx,
        };
        chain.reload();
        Ok(chain)
    }

    /// Drain the watcher channel and reload if anything changed.
    /// Returns true if the shader chain was rebuilt.
    pub fn poll_reload(&mut self) -> bool {
        let mut should_reload = false;
        while let Ok(res) = self.rx.try_recv() {
            match res {
                Ok(_event) => should_reload = true,
                Err(e) => warn!("notify error: {e}"),
            }
        }
        if should_reload {
            self.reload();
        }
        should_reload
    }

    /// Force-rebuild the chain from the current state of `shaders_dir`.
    pub fn reload(&mut self) {
        let mut paths: Vec<PathBuf> = match std::fs::read_dir(&self.shaders_dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wgsl"))
                .collect(),
            Err(e) => {
                self.last_error = Some(format!("read_dir {:?}: {e}", self.shaders_dir));
                error!(error = ?e, "shaders dir read failed");
                return;
            }
        };
        paths.sort();

        let mut new_nodes = Vec::with_capacity(paths.len());
        for path in &paths {
            match self.compile_node(path) {
                Ok(node) => new_nodes.push(node),
                Err(err) => {
                    self.last_error = Some(format!("{}: {err}", path.display()));
                    error!(path = %path.display(), error = %err, "shader compile failed");
                    return; // keep the previous node list intact
                }
            }
        }
        info!(count = new_nodes.len(), "shader chain rebuilt");
        for n in &new_nodes {
            debug!(name = %n.name, "  - loaded");
        }
        self.nodes = new_nodes;
        self.last_error = None;
    }

    fn compile_node(&self, path: &Path) -> Result<ShaderNode, String> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("read: {e}"))?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("shader")
            .to_string();
        // wgpu 29: push_error_scope returns an ErrorScopeGuard whose
        // .pop() yields a future. We block on it after building the
        // pipeline so any compile/link error surfaces here as a Result.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&name),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(format!("{err}"));
        }
        Ok(ShaderNode { name, pipeline })
    }

    /// Record the chain into the given encoder. Caller is responsible
    /// for ensuring `buf_a` already contains the source pixels (typically
    /// written by the NV12→BGRA pre-pass). Returns which buffer holds
    /// the final result.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) -> FinalBuffer {
        let groups_x = (self.width + 7) / 8;
        let groups_y = (self.height + 7) / 8;
        // Even number of nodes → result in A. Odd → result in B.
        let mut current_is_a = true;
        for node in &self.nodes {
            let bind = if current_is_a {
                &self.bind_a_to_b
            } else {
                &self.bind_b_to_a
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&node.name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&node.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
            current_is_a = !current_is_a;
        }
        if current_is_a {
            FinalBuffer::A
        } else {
            FinalBuffer::B
        }
    }

    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    dims: &wgpu::Buffer,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lab-chain-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dims.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dst.as_entire_binding(),
            },
        ],
    })
}
