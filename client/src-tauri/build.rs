use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    tauri_build::build();

    // Compile Swift audio capture module
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let swift_src = manifest_dir.join("swift").join("AudioCapture.swift");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib_path = out_dir.join("libAudioCapture.a");

    let status = Command::new("swiftc")
        .args([
            "-emit-library",
            "-static",
            "-module-name",
            "AudioCapture",
            "-o",
            lib_path.to_str().unwrap(),
            swift_src.to_str().unwrap(),
            "-Osize",
        ])
        .status()
        .expect("Failed to run swiftc. Is Xcode installed?");

    assert!(status.success(), "Swift compilation failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=AudioCapture");
    println!("cargo:rustc-link-lib=framework=CoreAudio");
    println!("cargo:rustc-link-lib=framework=AudioToolbox");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rerun-if-changed=swift/AudioCapture.swift");

    // Link Swift runtime
    let swift_libs = Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .expect("Failed to find SDK path");
    let sdk_path = String::from_utf8(swift_libs.stdout).unwrap().trim().to_string();

    // Find Swift lib directory
    let toolchain = Command::new("xcrun")
        .args(["--find", "swiftc"])
        .output()
        .expect("Failed to find swiftc");
    let swiftc_path = PathBuf::from(String::from_utf8(toolchain.stdout).unwrap().trim().to_string());
    let swift_lib_dir = swiftc_path
        .parent().unwrap()  // bin
        .parent().unwrap()  // usr
        .join("lib").join("swift").join("macosx");

    println!("cargo:rustc-link-search=native={}", swift_lib_dir.display());
    println!("cargo:rustc-link-search=native={}/usr/lib/swift", sdk_path);
    println!("cargo:rustc-link-lib=dylib=swiftCore");

    // Set rpath so the binary can find Swift runtime dylibs
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", swift_lib_dir.display());
}
