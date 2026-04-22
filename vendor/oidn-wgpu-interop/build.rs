// all cfg aliases in here should match with the corresponding ones in wgpu-hal that gate things we use
fn main() {
    cfg_aliases::cfg_aliases! {
        dx12: { all(target_os = "windows", feature = "dx12") },
        vulkan: { all(not(target_arch = "wasm32"), feature = "vulkan") },
    }
}
