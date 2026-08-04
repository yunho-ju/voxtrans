//! Proves `cargo test` runs at all against the library crate.
//!
//! Feature 002 adds the client's first tests. Before it there was no `tests/`
//! directory and no `[dev-dependencies]`, so every claim about client
//! behaviour was unverifiable. This file exists so that the scaffolding itself
//! has a Red/Green history separate from the behaviour under repair — if the
//! harness breaks later, the failure points here rather than at whatever test
//! happened to run first.

/// The library crate is linkable from an integration test.
///
/// This is not a tautology: `voxtrans_client_lib` declares
/// `crate-type = ["lib", "cdylib", "staticlib"]` and links a static Swift
/// archive built by `build.rs`. If the Swift compilation or the link step
/// regresses, this test fails to build, which is the signal we want.
#[test]
fn library_crate_links_from_an_integration_test() {
    let devices = voxtrans_client_lib::audio_output::list_output_devices();

    // Deliberately not asserting a device count: the set depends on what is
    // plugged into the machine, and a test that depends on ambient hardware
    // state is the failure mode feature 001 spent three review rounds
    // removing. What is asserted is that the call crosses the Swift FFI
    // boundary and returns an owned, well-formed structure.
    for (name, uid) in &devices {
        assert!(!name.is_empty(), "device name must not be empty: {devices:?}");
        assert!(!uid.is_empty(), "device uid must not be empty: {devices:?}");
    }
}
