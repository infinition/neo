// Patched for Neo-FFmpeg: dynamic loading via libloading.
//
// We do NOT link to nvEncodeAPI / nvcuvid at build time. Instead, the sys
// modules below load `nvEncodeAPI64.dll` / `libnvidia-encode.so` (and the
// NVDEC counterpart) at runtime via libloading. This means the crate
// builds on any machine with rustc — the NVIDIA Video Codec SDK is no
// longer required to compile, only an NVIDIA driver is required to run.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
