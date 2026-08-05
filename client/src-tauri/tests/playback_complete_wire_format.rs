//! T032 Critic (major): `audio_output.rs`'s poll loop is the only place
//! `playback_complete`'s JSON is built (`contracts/playback-protocol.md`:
//! `{"type": "playback_complete", "segment_id": <int>}`), but nothing
//! observed that string's actual content -- renaming the field to
//! `"segmentId"` left both `cargo test` (21 passing) and the full server
//! suite (119 passing) green (Critic reproduction: the server's
//! `_handle_control` treats an unrecognized field as a missing
//! `segment_id`, which is already a documented no-op per FR-012, so nothing
//! anywhere noticed). At runtime this degrades silently to the deadline
//! fallback -- exactly the failure mode FR-004 exists to bound, but bounding
//! it is not the same as this shipping correctly. This test drives the real
//! `start_tts_playback` (same real-device pattern as
//! `tts_playback_stop_responsiveness.rs`) and asserts the wire string
//! itself matches the contract shape.

use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Duration;
use tokio::sync::mpsc;

// Generous relative to how quickly one short chunk should actually play and
// its tts_end be processed by the poll loop (well under 1s observed) -- see
// `tts_playback_stop_responsiveness.rs`'s identical 5s choice for the same
// class of bounded wait.
const PLAYBACK_COMPLETE_TIMEOUT: Duration = Duration::from_secs(5);

const SEGMENT_ID: i64 = 42;

#[test]
fn playback_complete_wire_string_matches_the_contract_shape() {
    // Empty UID = system default output -- every Mac has one, so this does
    // not need a hardware-presence skip the way a specific named device
    // would (same convention as `tts_playback_stop_responsiveness.rs`).
    let outputs = voxtrans_client_lib::audio_output::list_output_devices();
    if outputs.is_empty() {
        eprintln!(
            "no output devices enumerated on this machine -- skipping (no real audio hardware \
             to drive `start_tts_playback` against)"
        );
        return;
    }

    let (tts_tx, tts_rx) = mpsc::channel::<Vec<u8>>(4);
    let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
    let (tts_end_tx, tts_end_rx) = mpsc::channel::<i64>(1);
    let (playback_complete_tx, mut playback_complete_rx) = mpsc::channel::<String>(4);

    let (done_tx, done_rx) = std_mpsc::channel::<Result<(), String>>();
    thread::spawn(move || {
        let result = voxtrans_client_lib::audio_output::start_tts_playback(
            String::new(),
            String::new(),
            tts_rx,
            stop_rx,
            tts_end_rx,
            playback_complete_tx,
        );
        let _ = done_tx.send(result);
    });

    // Let audio_output_start settle before enqueuing -- same margin as
    // `tts_playback_stop_responsiveness.rs`.
    thread::sleep(Duration::from_millis(300));

    // One short chunk (content is irrelevant to this test -- T025's own
    // hardware harness, test_playback_completion.py, already verifies real
    // audio content and completion timing; this test only cares about the
    // wire string `playback_complete` produces).
    let silence = vec![0.0f32; 2_400]; // 0.1s at 24kHz mono
    let bytes: Vec<u8> = silence.iter().flat_map(|s| s.to_le_bytes()).collect();
    tts_tx
        .blocking_send(bytes)
        .expect("failed to send chunk to tts_rx");
    tts_end_tx
        .blocking_send(SEGMENT_ID)
        .expect("failed to send tts_end to tts_end_rx");

    // Relay the async receiver onto a std channel so this can be waited on
    // with a bounded, non-async timeout (same pattern as `done_rx` below).
    let (msg_tx, msg_rx) = std_mpsc::channel::<String>();
    thread::spawn(move || {
        if let Some(message) = playback_complete_rx.blocking_recv() {
            let _ = msg_tx.send(message);
        }
    });

    let message = match msg_rx.recv_timeout(PLAYBACK_COMPLETE_TIMEOUT) {
        Ok(message) => message,
        Err(_) => panic!(
            "no playback_complete message arrived within {PLAYBACK_COMPLETE_TIMEOUT:?} of \
             enqueuing one chunk and its tts_end"
        ),
    };

    let parsed: serde_json::Value = serde_json::from_str(&message)
        .unwrap_or_else(|e| panic!("expected valid JSON, got {message:?}: {e}"));
    assert_eq!(
        parsed.get("type").and_then(|v| v.as_str()),
        Some("playback_complete"),
        "expected the contract's literal \"type\" field, got {message:?}"
    );
    assert_eq!(
        parsed.get("segment_id").and_then(|v| v.as_i64()),
        Some(SEGMENT_ID),
        "expected the contract's \"segment_id\" field (not e.g. \"segmentId\"), got {message:?}"
    );

    stop_tx
        .blocking_send(())
        .expect("failed to send stop signal to start_tts_playback's stop_rx");
    match done_rx.recv_timeout(PLAYBACK_COMPLETE_TIMEOUT) {
        Ok(result) => assert!(
            result.is_ok(),
            "expected start_tts_playback to shut down cleanly after stop, got {result:?}"
        ),
        Err(_) => panic!(
            "start_tts_playback did not return within {PLAYBACK_COMPLETE_TIMEOUT:?} of the stop \
             signal"
        ),
    }
}
