//! Client-side completion tracking for the segment currently draining
//! through TTS playback (`data-model.md` §3 `PlaybackLifecycle`,
//! `contracts/playback-protocol.md` "When the client sends it").
//!
//! Deliberately free of FFI and I/O: it does not call into Swift, does not
//! own a WebSocket, and does not know about channels. It is driven by
//! events its caller (`audio_output::start_tts_playback`'s poll loop)
//! observes from elsewhere:
//!
//! - `observe_outstanding_buffers` -- Swift's own outstanding-buffer count
//!   (`AudioCapture.swift`'s `audio_output_outstanding_buffers`, T025) is
//!   the authoritative increment (on `scheduleBuffer`) / decrement (on
//!   `.dataPlayedBack`) source; this struct only reads the number Swift
//!   reports, it does not keep a second copy of that bookkeeping.
//! - `on_stream_ended` -- the server's `tts_end` for a `segment_id` arrived
//!   (parsed in `ws_client.rs`'s receive loop; Swift never sees a
//!   segment_id at all -- "segment 경계는 Rust가 안다").
//! - `on_chunk_arrived` / `on_chunk_enqueued` -- a chunk has been received
//!   on the Rust side (`tts_rx`) but not yet handed to Swift, and then has
//!   been. `tts_end` arrives on a different channel than audio chunks with
//!   no cross-channel ordering guarantee at the poll loop's granularity, so
//!   Swift's outstanding-buffer count alone cannot see a chunk that has not
//!   yet been enqueued (Critic round-3 blocker).
//! - `on_stop` -- the whole playback session is ending (`stop_capture`)
//!   before a segment satisfied both completion conditions.
//!
//! Kept as a pure state machine specifically so the exactly-once decision
//! the contract requires ("Exactly once per segment_id") is unit-testable
//! without a Swift dylib or a live server connection -- see
//! `specs/002-audio-routing/briefs/US2-round3-T025-T026.md`, "Rust 절반".
//! `tests/playback_lifecycle.rs` covers it; live WebSocket wiring
//! (`ws_client.rs`, `audio_output.rs`) is covered by compilation and the
//! existing test suite's regression, per that brief -- a real round trip to
//! the server is T031/T046's job, not this one's.

/// One playback segment's completion state, tracked at a time (the
/// ordering invariant in `contracts/playback-protocol.md` -- suppression
/// held from the moment a segment starts synthesizing until it releases --
/// means the server never starts a second segment while the first is still
/// suppressing, so only one segment is ever "current" here).
#[derive(Debug, Default)]
pub struct PlaybackLifecycle {
    segment_id: Option<i64>,
    scheduled_buffers: i64,
    stream_ended: bool,
    sent: bool,
    /// Audio chunks the Rust side has received (`tts_rx`) but not yet
    /// handed to Swift (`audio_output_enqueue`) -- see `on_chunk_arrived`.
    unenqueued_chunks: i64,
}

impl PlaybackLifecycle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records Swift's outstanding-buffer count as of this poll tick
    /// (`data-model.md` §3 `scheduled_buffers`). Not accumulated locally --
    /// Swift already owns both the increment and decrement sides, this is
    /// a read, not a second count that could drift from it.
    pub fn observe_outstanding_buffers(&mut self, count: i64) {
        self.scheduled_buffers = count;
    }

    /// The server's `tts_end` for `segment_id` has arrived: no more chunks
    /// are coming for it (`data-model.md` §3 `stream_ended`).
    ///
    /// A duplicate `tts_end` for the segment already being tracked is a
    /// no-op -- exactly one `playback_complete` must still result, not one
    /// per `tts_end` received. Only a *different* `segment_id` starts a new
    /// tracking window (and clears the previous segment's `sent` flag,
    /// which by then must already be true or moot -- see `on_stop`).
    pub fn on_stream_ended(&mut self, segment_id: i64) {
        if self.segment_id == Some(segment_id) && self.stream_ended {
            return;
        }
        self.segment_id = Some(segment_id);
        self.stream_ended = true;
        self.sent = false;
    }

    /// A chunk of TTS audio has arrived on the Rust side (`tts_rx`) but has
    /// not yet been handed to Swift's player node (`audio_output_enqueue`).
    ///
    /// Models the Critic round-3 blocker: `tts_end` (`on_stream_ended`)
    /// arrives on a *different* channel than audio chunks, with no
    /// ordering guarantee between them at the consumer's poll granularity
    /// (`audio_output.rs`'s `start_tts_playback` loop). A segment short
    /// enough that its last chunk and its `tts_end` are both observed in
    /// the same tick must not be judged complete before that chunk is
    /// actually enqueued -- `observe_outstanding_buffers` alone cannot see
    /// it, since Swift only learns of a chunk once it is handed to
    /// `audio_output_enqueue`.
    pub fn on_chunk_arrived(&mut self) {
        self.unenqueued_chunks += 1;
    }

    /// The chunk noted by `on_chunk_arrived` has now been handed to Swift.
    /// From this point, Swift's own outstanding-buffer count
    /// (`observe_outstanding_buffers`) is the sole authority for it.
    pub fn on_chunk_enqueued(&mut self) {
        self.unenqueued_chunks -= 1;
    }

    fn is_ready(&self) -> bool {
        self.segment_id.is_some()
            && self.stream_ended
            && self.scheduled_buffers == 0
            && self.unenqueued_chunks == 0
            && !self.sent
    }

    /// Returns the segment_id to send `playback_complete` for, the moment
    /// (and only the moment) both completion conditions first hold
    /// together. Returns `None` on every call before that and every call
    /// after -- the contract's "exactly once per segment_id"
    /// (`contracts/playback-protocol.md`) lives entirely in this one
    /// method: it flips `sent` the instant it returns `Some`, so no later
    /// observation of the same segment (a stale buffer-count read, a
    /// repeated poll tick) can produce a second `Some` for it.
    pub fn take_completed_segment(&mut self) -> Option<i64> {
        if self.is_ready() {
            self.sent = true;
            self.segment_id
        } else {
            None
        }
    }

    /// The whole playback session is stopping (`stop_capture`) before this
    /// segment satisfied both completion conditions. Per
    /// `contracts/playback-protocol.md`'s "backward compatibility" clause,
    /// a segment cut off this way is not falsely reported as complete --
    /// the server's own deadline (`data-model.md` §4) is the release path
    /// for it, the same one that covers a client that disconnects, stalls,
    /// or drops the message entirely. Marks the segment `sent` so a stray
    /// late buffer-count observation (e.g. a completion callback Swift
    /// fires after the engine has already been torn down) cannot
    /// retroactively complete it.
    pub fn on_stop(&mut self) {
        self.sent = true;
    }
}
