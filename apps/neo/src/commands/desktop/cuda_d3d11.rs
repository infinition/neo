//! Minimal FFI shim for `cuGraphicsD3D11RegisterResource`.
//!
//! `cudarc` 0.19 does not expose CUDA <-> D3D11 graphics interop, so we
//! resolve the single function we need at runtime from `nvcuda.dll` (which
//! cudarc has already loaded with libloading). Everything else (mapping,
//! sub-resource → CUarray, memcpy2D, unmap) lives in cudarc's public sys
//! bindings.

use cudarc::driver::sys::{CUgraphicsResource, CUresult};
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::sync::OnceLock;

/// `cuGraphicsD3D11RegisterResource(pCudaResource, pD3DResource, flags)`
pub type CuGraphicsD3D11RegisterResourceFn = unsafe extern "C" fn(
    *mut CUgraphicsResource,
    *mut c_void, // ID3D11Resource *
    u32,
) -> CUresult;

/// `CU_GRAPHICS_REGISTER_FLAGS_NONE`
pub const CU_GRAPHICS_REGISTER_FLAGS_NONE: u32 = 0x00;

struct Loaded {
    _lib: Library,
    register: CuGraphicsD3D11RegisterResourceFn,
}

static SHIM: OnceLock<Loaded> = OnceLock::new();

fn load() -> Result<&'static Loaded, String> {
    if let Some(s) = SHIM.get() {
        return Ok(s);
    }
    unsafe {
        let lib = Library::new("nvcuda.dll")
            .map_err(|e| format!("LoadLibrary(nvcuda.dll): {e}"))?;
        let sym: Symbol<CuGraphicsD3D11RegisterResourceFn> = lib
            .get(b"cuGraphicsD3D11RegisterResource\0")
            .map_err(|e| format!("GetProcAddress(cuGraphicsD3D11RegisterResource): {e}"))?;
        let register = *sym;
        // Best-effort init; if another thread won the race, we just throw ours away.
        let _ = SHIM.set(Loaded {
            _lib: lib,
            register,
        });
    }
    Ok(SHIM.get().expect("shim set"))
}

/// Safe-ish wrapper around `cuGraphicsD3D11RegisterResource`.
///
/// `d3d_resource` must be a valid `ID3D11Resource *` (raw COM pointer).
/// The returned `CUgraphicsResource` must later be released with
/// `cuGraphicsUnregisterResource`.
pub unsafe fn register_d3d11_resource(
    d3d_resource: *mut c_void,
    flags: u32,
) -> Result<CUgraphicsResource, String> {
    let shim = load()?;
    let mut res: CUgraphicsResource = std::ptr::null_mut();
    let r = (shim.register)(&mut res, d3d_resource, flags);
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("cuGraphicsD3D11RegisterResource: {r:?}"));
    }
    Ok(res)
}
