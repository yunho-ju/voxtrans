//! T026 (US2 round 3): the exactly-once decision
//! `contracts/playback-protocol.md` requires ("Exactly once per
//! segment_id") lives in `playback_lifecycle::PlaybackLifecycle`, a pure
//! state machine with no FFI/WS dependency (see that module's doc comment
//! for why). Drives it directly, the same way `audio_output.rs`'s poll
//! loop will, without a Swift dylib or a live server -- see
//! `specs/002-audio-routing/briefs/US2-round3-T025-T026.md` "Rust 절반"
//! for the boundary this file covers vs. what WS wiring covers.

use voxtrans_client_lib::playback_lifecycle::PlaybackLifecycle;

/// data-model.md §3: "an empty queue alone does not mean playback
/// finished". Neither condition alone is enough -- only their conjunction
/// is.
#[test]
fn neither_condition_alone_completes_the_segment() {
    let mut lifecycle = PlaybackLifecycle::new();

    // Buffers drain to zero, but tts_end hasn't arrived yet -- the next
    // chunk might just not have arrived.
    lifecycle.observe_outstanding_buffers(3);
    lifecycle.observe_outstanding_buffers(0);
    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected no completion while stream_ended is still false"
    );

    // tts_end arrives, but buffers are still outstanding.
    lifecycle.observe_outstanding_buffers(2);
    lifecycle.on_stream_ended(7);
    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected no completion while buffers are still outstanding, even after tts_end"
    );

    // Now both hold.
    lifecycle.observe_outstanding_buffers(0);
    assert_eq!(
        lifecycle.take_completed_segment(),
        Some(7),
        "expected completion once stream_ended and scheduled_buffers == 0 both hold"
    );
}

/// contracts/playback-protocol.md: "Exactly once per segment_id" -- a
/// duplicate `tts_end` for the same segment must not produce a second
/// `playback_complete`.
#[test]
fn duplicate_tts_end_does_not_produce_a_second_completion() {
    let mut lifecycle = PlaybackLifecycle::new();
    lifecycle.observe_outstanding_buffers(0);
    lifecycle.on_stream_ended(7);
    lifecycle.on_stream_ended(7); // duplicate, e.g. a retransmit

    let mut completions = Vec::new();
    for _ in 0..3 {
        if let Some(id) = lifecycle.take_completed_segment() {
            completions.push(id);
        }
    }
    assert_eq!(
        completions,
        vec![7],
        "expected exactly one completion for segment 7 despite the duplicate tts_end"
    );
}

/// A `tts_end` retransmitted *after* the segment already completed and sent
/// (the case `on_stream_ended`'s duplicate guard exists for -- without it,
/// resetting `sent` back to `false` would let a later poll tick resend a
/// `playback_complete` the server already received) must not resend it.
#[test]
fn tts_end_repeated_after_the_segment_already_sent_does_not_resend() {
    let mut lifecycle = PlaybackLifecycle::new();
    lifecycle.observe_outstanding_buffers(0);
    lifecycle.on_stream_ended(7);
    assert_eq!(lifecycle.take_completed_segment(), Some(7));

    // A retransmitted or duplicate tts_end arrives for the segment already
    // reported complete.
    lifecycle.on_stream_ended(7);
    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected no resend for a tts_end repeated after the segment already sent"
    );
}

/// A late poll tick after a segment has already sent its
/// `playback_complete` must not re-fire it, even if outstanding-buffer
/// observations keep arriving at zero.
#[test]
fn repeated_polling_after_completion_yields_no_further_sends() {
    let mut lifecycle = PlaybackLifecycle::new();
    lifecycle.observe_outstanding_buffers(0);
    lifecycle.on_stream_ended(7);
    assert_eq!(lifecycle.take_completed_segment(), Some(7));

    lifecycle.observe_outstanding_buffers(0);
    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected no second completion for the same segment on a later poll tick"
    );
}

/// `audio_output_stop` does not wait for pending buffers
/// (`audio_output.rs`). A segment interrupted by stop before it satisfied
/// both completion conditions must not be reported complete -- the
/// server's deadline is the release path for it instead (`data-model.md`
/// §4, contracts/playback-protocol.md "Backward compatibility").
#[test]
fn stop_before_completion_suppresses_the_send_even_if_buffers_later_read_zero() {
    let mut lifecycle = PlaybackLifecycle::new();
    lifecycle.observe_outstanding_buffers(4);
    lifecycle.on_stream_ended(7);
    lifecycle.on_stop();

    // A stray late completion callback from Swift (fired after the engine
    // was already torn down) reports the queue as drained.
    lifecycle.observe_outstanding_buffers(0);
    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected stop to suppress the send for a segment that never actually finished draining"
    );
}

/// Critic round-3 blocker: `tts_end` arrives on a different channel than
/// audio chunks (`ws_client.rs` vs. `audio_output.rs`'s `tts_rx`), with no
/// ordering guarantee at the consumer's poll granularity. A segment short
/// enough that its last chunk and its `tts_end` are both observed within
/// the same tick must not be judged complete before that chunk is actually
/// handed to Swift (`audio_output_enqueue`) -- `observe_outstanding_buffers`
/// alone cannot see a chunk that has arrived on the Rust side but has not
/// yet been enqueued, so `on_chunk_arrived`/`on_chunk_enqueued` track that
/// window explicitly.
#[test]
fn stream_ended_observed_before_the_last_chunk_is_enqueued_does_not_complete_early() {
    let mut lifecycle = PlaybackLifecycle::new();
    lifecycle.observe_outstanding_buffers(0);

    // The last chunk has arrived on the Rust side (sitting in tts_rx) but
    // has not yet been handed to Swift this tick.
    lifecycle.on_chunk_arrived();
    // tts_end is observed in the same tick, before that chunk is enqueued --
    // exactly the interleaving the blocker describes.
    lifecycle.on_stream_ended(7);

    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected no completion while a received chunk has not yet been enqueued into Swift, \
         even though Swift's own outstanding count already reads 0"
    );

    // The chunk is now actually enqueued -- Swift's count reflects it.
    lifecycle.on_chunk_enqueued();
    lifecycle.observe_outstanding_buffers(1);
    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected no completion while the just-enqueued buffer is still outstanding in Swift"
    );

    // It finishes playing for real.
    lifecycle.observe_outstanding_buffers(0);
    assert_eq!(
        lifecycle.take_completed_segment(),
        Some(7),
        "expected completion once the chunk has both been enqueued and actually played back"
    );
}

/// Minor 1 (Critic round 3): a *different* segment's `tts_end` arriving
/// while the current segment has not yet completed overwrites the tracked
/// segment_id -- the previous segment's own `playback_complete` is silently
/// never sent. Documented here as an accepted, deadline-covered path (same
/// reasoning as `on_stop`), not a defect: the protocol's ordering invariant
/// means the server does not start a second segment until the first
/// releases, so this should not arise in practice, but the state machine
/// does not depend on that invariant holding to stay exactly-once-or-fewer
/// for each segment_id it does track.
#[test]
fn a_different_segments_tts_end_overwrites_an_incomplete_segment_without_double_sending() {
    let mut lifecycle = PlaybackLifecycle::new();

    lifecycle.observe_outstanding_buffers(3);
    lifecycle.on_stream_ended(1); // segment 1's stream closed, but buffers > 0

    // Segment 2's tts_end arrives before segment 1 ever completed.
    lifecycle.on_stream_ended(2);
    lifecycle.observe_outstanding_buffers(0);

    // Segment 1 never sends -- its tracking was overwritten, not completed.
    // Only segment 2 (the one now tracked) can complete.
    assert_eq!(
        lifecycle.take_completed_segment(),
        Some(2),
        "expected only segment 2 (the segment now tracked) to complete; segment 1's own \
         completion is lost, not double-sent -- accepted, since the server's deadline covers it"
    );
}

/// Two segments played back to back must each send their own
/// `playback_complete`, exactly once, with the correct segment_id -- not
/// conflated buffer counts or a stuck `sent` flag carried over from the
/// first segment.
#[test]
fn two_consecutive_segments_each_complete_exactly_once_with_their_own_id() {
    let mut lifecycle = PlaybackLifecycle::new();

    lifecycle.observe_outstanding_buffers(2);
    lifecycle.observe_outstanding_buffers(0);
    lifecycle.on_stream_ended(1);
    assert_eq!(lifecycle.take_completed_segment(), Some(1));

    // Second segment starts: buffers accumulate again before its own
    // tts_end arrives.
    lifecycle.observe_outstanding_buffers(5);
    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected no completion for the new segment while its buffers are still outstanding"
    );
    lifecycle.on_stream_ended(2);
    assert_eq!(
        lifecycle.take_completed_segment(),
        None,
        "expected no completion yet -- segment 2's buffers have not drained"
    );

    lifecycle.observe_outstanding_buffers(0);
    assert_eq!(
        lifecycle.take_completed_segment(),
        Some(2),
        "expected segment 2 to complete independently of segment 1's already-sent state"
    );
}
