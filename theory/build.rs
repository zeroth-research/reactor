fn main() {
    // On macOS and Linux, with the `torch` feature, the test binaries of this crate need:
    // - an rpath to find libtorch at runtime: torch-sys only emits rustc-link-search
    //   (link-time), not -rpath (runtime);
    // - libpython at link time: the tests embed Python, but `pyo3-tch` forces pyo3's
    //   `extension-module` feature, so nothing links libpython and the linker rejects the
    //   undefined `Py*` symbols.
    // `rustc-link-arg` applies only to this crate's own linked targets (its test binaries),
    // not to dependents, so the `reactor` Python extension stays unlinked from libpython.
    #[cfg(all(any(target_os = "macos", target_os = "linux"), feature = "torch"))]
    {
        const SCRIPT: &str = "\
import os, pathlib, sysconfig, torch
print(pathlib.Path(torch.__file__).parent / 'lib')
base = sysconfig.get_config_var('PYTHONFRAMEWORKPREFIX') or sysconfig.get_config_var('LIBDIR')
print(os.path.join(base, sysconfig.get_config_var('LDLIBRARY')))
";
        let output = std::process::Command::new("python3")
            .args(["-c", SCRIPT])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8(o.stdout).unwrap();
                let mut lines = stdout.lines().map(str::trim);
                let torch_lib = lines.next().unwrap();
                let libpython = std::path::Path::new(lines.next().unwrap());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{torch_lib}");
                // On Linux, the PyTorch wheel's bundled libraries (e.g. OpenBLAS) depend on
                // siblings in `torch.libs` without a RUNPATH; GNU ld must be told where they are
                // to resolve those indirect dependencies at link time.
                #[cfg(target_os = "linux")]
                println!("cargo:rustc-link-arg=-Wl,-rpath-link,{torch_lib}/../../torch.libs");
                println!("cargo:rustc-link-arg={}", libpython.display());
                if let Some(dir) = libpython.parent() {
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
                }
            }
            _ => {
                println!(
                    "cargo:warning=theory build.rs: could not find torch/libpython via python3; test binaries may fail to link or load"
                );
            }
        }

        println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    }
}
