import Foundation
import CoreAudio
import AVFoundation
import AudioToolbox
import AppKit
import CoreGraphics

// C callback type: (data_ptr, num_samples, sample_rate, channels)
typealias AudioCallback = @convention(c) (UnsafePointer<Float>, Int, Int, Int) -> Void
// T035 (US4, research.md D6): reports a device-change rebuild that could
// not succeed, as a C string message, so the failure surfaces (FR-008)
// instead of the capture thread silently continuing to look healthy.
typealias AudioErrorCallback = @convention(c) (UnsafePointer<CChar>) -> Void

// ========== System Audio Capture (existing) ==========

// Global state for system audio capture
private var audioEngine: AVAudioEngine?
private var tapID: AudioObjectID = kAudioObjectUnknown
private var captureCallback: AudioCallback?
private var captureErrorCallback: AudioErrorCallback?
private var aggregateDeviceID: AudioObjectID = kAudioObjectUnknown

// T035: guards every read/write of the state above, since a device-change
// rebuild can now run on `sckDeviceChangeQueue` (a Core Audio-owned or
// NotificationCenter-owned thread) concurrently with `sck_start_capture`/
// `sck_stop_capture` being called from Rust's own capture thread.
private let sckStateLock = NSLock()

// Serial queue both device-change listeners below deliver onto, so a
// default-output-device change and an AVAudioEngineConfigurationChange
// firing close together cannot run their rebuilds concurrently with each
// other (only `sckStateLock` protects them from `sck_start_capture`/
// `sck_stop_capture`, which run on a different thread entirely).
private let sckDeviceChangeQueue = DispatchQueue(label: "com.voxtrans.sck.devicechange")
private var sckDefaultOutputListenerBlock: AudioObjectPropertyListenerBlock?
private var sckConfigChangeObserver: NSObjectProtocol?

// T037 Critic major 3: `engine.start()` reliably posts exactly one
// `AVAudioEngineConfigurationChange` notification for the engine that just
// started (measured: every single `buildSckCaptureGraph()` call, ~95ms
// after start). A *time-based* delay before registering the observer (this
// file's first attempt) permanently misses any genuine change that lands
// inside that window -- exactly the AirPods A2DP<->HFP profile switch
// this listener is the *only* signal for (the device identity does not
// change, only its rate, so `kAudioHardwarePropertyDefaultOutputDevice`
// never fires). Registering immediately and ignoring only the first
// notification actually delivered for this specific engine instance
// -- an identity-based guard, not a clock -- closes that window. Reset to
// `true` every time a build's config-change observer is (re)registered.
private var sckIgnoreNextConfigChangeNotification = false

// T037 Critic round 1 major 1 attempted "rebuild reported success is not
// the same as audio actually flowing" with a watchdog on *every* build,
// including the very first one. Critic round 2 blocker: that armed on
// fresh starts too, where silence is completely normal (listen mode
// started before a meeting has any audio at all) -- measured killing a
// legitimate, silent session at 2.3s. Round 2 design (confirmed, not
// re-discussed): the watchdog only ever arms for a *rebuild*, and only
// when audio was genuinely flowing in the moments right before that
// rebuild (`sckWasRecentlyFlowing`, using `sckRecentCallbackTimestamps`)
// -- never for the initial build. `sckBuildGeneration` is incremented on
// every successful `buildSckCaptureGraph()` call (initial start and every
// rebuild share this one function) and pins each armed watchdog and each
// poller/observer-fallback tick to the specific build that scheduled it.
// `sckReceivedAudioSinceLastBuild` is set by the tap callback the moment
// any audio actually arrives after the current build.
private var sckBuildGeneration: UInt64 = 0
private var sckReceivedAudioSinceLastBuild = false

// Ring buffer of recent tap-callback timestamps (CoreFoundation absolute
// time, seconds), used only to answer "was audio actually flowing in the
// last few seconds" at the moment a rebuild is about to happen -- the
// pure decision in `sckWasRecentlyFlowing` needs only the most recent
// entry, but a short buffer (rather than a single scalar) keeps the state
// shape self-documenting as "recent history", not a single ad hoc flag.
private var sckRecentCallbackTimestamps: [TimeInterval] = []
private let sckRecentCallbackTimestampCapacity = 8

// How far back, from the moment a rebuild is about to start, counts as
// "was recently flowing" for arming the post-rebuild watchdog below.
private let sckPreRebuildFlowingWindow: TimeInterval = 3.0

// How long an *armed* post-rebuild watchdog waits for at least one real
// callback before reporting a warning (T037 Critic round 2 major).
private let sckRebuildAudioWatchdogTimeout: TimeInterval = 4.0

// Found during this round's verification, under rapid repeated
// start/stop/device-change cycles (not visible in a single isolated
// trigger): a burst of several notifications -- both
// `kAudioHardwarePropertyDefaultOutputDevice` and
// `AVAudioEngineConfigurationChange` -- can arrive for what is, from the
// user's perspective, one logical change, each independently triggering
// its own `handleSckDeviceChange` call before this guard existed
// (reproduced directly: 15+ rebuilds in under 6 seconds). Debouncing --
// coalescing a burst into exactly one rebuild, run shortly after the
// *last* notification in the burst -- fixes this without the major-3
// flaw (a fixed delay *before arming*, which permanently drops a change
// landing inside it): every notification still schedules a rebuild -- a
// later, genuinely separate change simply gets its own rebuild once its
// own debounce window elapses undisturbed.
private var sckPendingRebuildWorkItem: DispatchWorkItem?
private let sckRebuildDebounceInterval: TimeInterval = 0.15

// T037 Critic round 2 major (ignore-first fallback): if the engine's own
// self-generated `AVAudioEngineConfigurationChange` notification never
// arrives at all within a bounded window (observed ~95ms in practice, but
// not a documented guarantee), `sckIgnoreNextConfigChangeNotification`
// would otherwise stay armed forever, silently swallowing the *next*
// genuine configuration change too. This fallback clears it automatically
// -- gated on the specific build's generation (`sckBuildGeneration`) so a
// *later* build's own fresh `true` can never be clobbered by a stale timer
// left over from an earlier one.
private let sckIgnoreNextConfigChangeFallbackTimeout: TimeInterval = 2.0

// T037 Critic round 2 major: a default-output change that only alters the
// *rate* of the already-default device (measured: exactly this shape for
// an AirPods A2DP<->HFP profile switch) fires neither
// `kAudioHardwarePropertyDefaultOutputDevice` (the device identity did not
// change) nor, on this machine, `AVAudioEngineConfigurationChange` --
// neither listener has anything to fire on. A poller is the only
// remaining option: while capturing, every `sckRatePollInterval`, compare
// the aggregate device's own nominal rate (set by the alignment step in
// `buildSckCaptureGraph`) against the system default output's *current*
// rate, read fresh each tick, and rebuild through the same debounced path
// on a mismatch. Applies whatever rate is actually read -- no hardcoded
// rate whitelist -- so it generalizes to rates this machine cannot
// reproduce (e.g. HFP's 16/24kHz), even though that generalization is
// unverified here (documented limitation, evidence/T037-round2.md).
private let sckRatePollInterval: TimeInterval = 2.0

/// Translates this process's own PID to a Core Audio process object via
/// `kAudioHardwarePropertyTranslatePIDToProcessObject` (research.md D5,
/// FR-006). `CATapDescription`'s `stereoGlobalTapButExcludeProcesses:`
/// initializer (used below) expects process *object* IDs, not PIDs --
/// passing `getpid()` directly would silently exclude nothing. Per the
/// CoreAudio header, a PID that does not
/// correspond to any process returns `kAudioObjectUnknown` rather than an
/// error; verified directly on this machine (a standalone probe, not part
/// of this suite -- see evidence/US2.md) that a genuine PID translates to a
/// valid, distinct object and a bogus one returns `kAudioObjectUnknown`.
private func ownProcessObjectID() -> AudioObjectID? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var pid = getpid()
    var objectID: AudioObjectID = kAudioObjectUnknown
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &address,
        UInt32(MemoryLayout<pid_t>.size),
        &pid,
        &size,
        &objectID
    )
    guard status == noErr, objectID != kAudioObjectUnknown else { return nil }
    return objectID
}

/// Returns the system's current default output device, or `nil` if it
/// cannot be resolved.
private func systemDefaultOutputDeviceID() -> AudioObjectID? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var deviceID: AudioObjectID = kAudioObjectUnknown
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID
    )
    guard status == noErr, deviceID != kAudioObjectUnknown else { return nil }
    return deviceID
}

/// Reads `kAudioDevicePropertyNominalSampleRate` for `deviceID`, or `nil`
/// if it cannot be read.
private func nominalSampleRate(of deviceID: AudioObjectID) -> Double? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyNominalSampleRate,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var rate: Double = 0
    var size = UInt32(MemoryLayout<Double>.size)
    let status = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &rate)
    guard status == noErr else { return nil }
    return rate
}

/// Sets `deviceID`'s own `kAudioDevicePropertyNominalSampleRate`.
@discardableResult
private func setNominalSampleRate(_ rate: Double, on deviceID: AudioObjectID) -> OSStatus {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyNominalSampleRate,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var value = rate
    return AudioObjectSetPropertyData(deviceID, &address, 0, nil, UInt32(MemoryLayout<Double>.size), &value)
}

/// Builds the tap + aggregate device + AVAudioEngine graph and starts it,
/// using whatever `captureCallback` is currently registered. Shared by the
/// initial `sck_start_capture` entry point and the device-change rebuild
/// path (T035, research.md D6) -- the graph is always torn down and rebuilt
/// from scratch rather than patched in place, since the tap, aggregate
/// device and engine are all bound to device identity at construction and
/// there is no partial-update path simpler than recreating them. Caller
/// must hold `sckStateLock`.
private func buildSckCaptureGraph() -> Bool {
    // Fresh build: whatever the *previous* graph's recent callback history
    // was, it says nothing about this new one -- `handleSckDeviceChange`
    // already read it (to decide whether to arm the post-rebuild watchdog)
    // before calling this function, so clearing it here cannot lose that
    // decision.
    sckRecentCallbackTimestamps.removeAll()

    // 1. Create a CATapDescription for global system audio tap, excluding
    // this app's own process object (FR-006, research.md D5) so the app
    // never hears its own synthesized TTS output back through the tap. If
    // translation fails for any reason, fall back to an empty exclude list
    // rather than failing capture outright -- self-exclusion is a
    // hardware-level hardening on top of the server-side suppression guard
    // (T029), not the only defense (research.md D5's "must fall back ...
    // if it does not [work]").
    // CATapDescription is a class (reference type) — no & needed
    let excludeProcesses: [AudioObjectID] = ownProcessObjectID().map { [$0] } ?? []
    if excludeProcesses.isEmpty {
        NSLog("VoxTrans: could not resolve own process object -- tap will not self-exclude")
    }
    let tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: excludeProcesses)
    tapDesc.uuid = UUID()

    // 2. Create the process tap via Core Audio
    var tapObjectID: AudioObjectID = kAudioObjectUnknown

    let tapStatus = AudioHardwareCreateProcessTap(tapDesc, &tapObjectID)
    guard tapStatus == noErr else {
        NSLog("VoxTrans: AudioHardwareCreateProcessTap failed: \(tapStatus)")
        return false
    }
    tapID = tapObjectID
    NSLog("VoxTrans: Created process tap ID: \(tapObjectID)")

    // 3. Create aggregate device that includes the tap
    let aggDesc: NSDictionary = [
        kAudioAggregateDeviceNameKey: "VoxTrans Tap",
        kAudioAggregateDeviceUIDKey: "com.voxtrans.tap.\(UUID().uuidString)",
        kAudioAggregateDeviceIsPrivateKey: 1,
        kAudioAggregateDeviceTapAutoStartKey: 1,
        kAudioAggregateDeviceTapListKey: [[
            "uid": tapDesc.uuid.uuidString,
        ]],
    ]

    var aggDeviceID: AudioObjectID = kAudioObjectUnknown
    let aggStatus = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggDeviceID)
    guard aggStatus == noErr else {
        NSLog("VoxTrans: AudioHardwareCreateAggregateDevice failed: \(aggStatus)")
        AudioHardwareDestroyProcessTap(tapID)
        tapID = kAudioObjectUnknown
        return false
    }
    aggregateDeviceID = aggDeviceID
    NSLog("VoxTrans: Created aggregate device ID: \(aggDeviceID)")

    // T037 Critic round 1 blocker: the aggregate device's own nominal rate
    // defaults to a fixed value (48000 Hz on this machine) independent of
    // whatever the system's actual default output device is currently
    // running at. When those differ, `engine.inputNode`'s format
    // (read below) mismatches the aggregate's real IO format and zero
    // callbacks are ever delivered -- even though the tap and aggregate
    // device both report `IsAlive`/`IsRunning` = 1 and `engine.start()`
    // does not throw. Reproduced and fixed with real hardware: a fresh
    // capture with the system default already at 44100 Hz produced zero
    // callbacks before this alignment, 29 in ~4s after it (same fix
    // confirmed at 32000 Hz).
    //
    // T037 round 2 minor: a nominal-rate change is not instantaneous at
    // the HAL level (the same characteristic this suite's own test
    // fixtures already encode, e.g. `mismatched_blackhole_rate`'s 0.2s
    // settle before its own readback) -- confirm via readback rather than
    // trusting a `noErr` status alone, retry once if it did not take, and
    // report (non-destructively -- see `scheduleRebuildAudioWatchdog`'s
    // WARN path) rather than silently proceeding if it still has not
    // confirmed. Capture still starts regardless: a failed alignment
    // leaves the pre-existing mismatch rather than making anything worse,
    // and either the post-rebuild watchdog (for a rebuild) or the
    // nominal-rate poller (for a later, independent rate drift) can still
    // observe and react to the resulting silence.
    if let defaultOutputID = systemDefaultOutputDeviceID(),
       let targetRate = nominalSampleRate(of: defaultOutputID) {
        var alignStatus = setNominalSampleRate(targetRate, on: aggDeviceID)
        Thread.sleep(forTimeInterval: 0.3)
        var confirmedRate = nominalSampleRate(of: aggDeviceID)
        if alignStatus != noErr || confirmedRate != targetRate {
            NSLog(
                "VoxTrans: aggregate rate alignment to \(targetRate) Hz did not confirm on the "
                    + "first try (status=\(alignStatus), confirmed=\(String(describing: confirmedRate))) -- retrying once"
            )
            alignStatus = setNominalSampleRate(targetRate, on: aggDeviceID)
            Thread.sleep(forTimeInterval: 0.3)
            confirmedRate = nominalSampleRate(of: aggDeviceID)
        }
        if alignStatus == noErr, confirmedRate == targetRate {
            NSLog("VoxTrans: Aligned aggregate device nominal rate to \(targetRate) Hz")
        } else {
            NSLog(
                "VoxTrans: Failed to confirm aggregate device nominal rate alignment to "
                    + "\(targetRate) Hz after retry (status=\(alignStatus), "
                    + "confirmed=\(String(describing: confirmedRate)))"
            )
            if let errorCB = captureErrorCallback {
                let message = "WARN: could not confirm audio capture's internal sample rate alignment (target \(targetRate) Hz) -- capture may be silent."
                message.withCString { cString in errorCB(cString) }
            }
        }
    } else {
        NSLog("VoxTrans: Could not resolve system default output device/rate for aggregate alignment")
    }

    // 4. Set up AVAudioEngine with the aggregate device as input
    let engine = AVAudioEngine()

    do {
        let audioUnit = engine.inputNode.audioUnit!

        var deviceID = aggDeviceID
        let setDeviceStatus = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            UInt32(MemoryLayout<AudioObjectID>.size)
        )
        guard setDeviceStatus == noErr else {
            NSLog("VoxTrans: Failed to set input device: \(setDeviceStatus)")
            tearDownSckGraph()
            return false
        }

        let inputFormat = engine.inputNode.outputFormat(forBus: 0)
        NSLog("VoxTrans: Input format: \(inputFormat)")

        let targetSampleRate: Double = 16000
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        )!

        let converter = AVAudioConverter(from: inputFormat, to: targetFormat)!

        engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            guard let cb = captureCallback else { return }

            // T037 major 1 / round 2: mark this build as having produced
            // real audio (checked by the post-rebuild watchdog) and record
            // the timestamp (checked by `sckWasRecentlyFlowing` before a
            // *later* rebuild decides whether to arm its own watchdog). A
            // plain locked write: this fires on every tap callback (~every
            // 85ms), so the lock must stay cheap, but correctness (not
            // losing a write) matters more than shaving a microsecond here.
            sckStateLock.lock()
            sckReceivedAudioSinceLastBuild = true
            sckRecentCallbackTimestamps.append(CFAbsoluteTimeGetCurrent())
            if sckRecentCallbackTimestamps.count > sckRecentCallbackTimestampCapacity {
                sckRecentCallbackTimestamps.removeFirst()
            }
            sckStateLock.unlock()

            let frameCount = buffer.frameLength
            let ratio = targetSampleRate / inputFormat.sampleRate
            let outputFrameCount = AVAudioFrameCount(Double(frameCount) * ratio)

            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
                return
            }

            var error: NSError?
            let convertStatus = converter.convert(to: outputBuffer, error: &error) { inNumPackets, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            guard convertStatus != .error, error == nil else {
                NSLog("VoxTrans: Conversion error: \(error?.localizedDescription ?? "unknown")")
                return
            }

            if let channelData = outputBuffer.floatChannelData?[0] {
                let sampleCount = Int(outputBuffer.frameLength)
                if sampleCount > 0 {
                    cb(channelData, sampleCount, Int(targetSampleRate), 1)
                }
            }
        }

        try engine.start()
        audioEngine = engine
        sckBuildGeneration += 1
        sckReceivedAudioSinceLastBuild = false
        registerSckConfigChangeObserver(for: engine)
        scheduleSckRatePoll(generation: sckBuildGeneration)
        NSLog("VoxTrans: Audio capture started via Core Audio Tap")
        return true

    } catch {
        NSLog("VoxTrans: Failed to start audio engine: \(error)")
        tearDownSckGraph()
        return false
    }
}

// MARK: - Permissions
//
// Only the microphone is preflighted here, and deliberately so.
//
// System audio is captured with a Core Audio process tap
// (`AudioHardwareCreateProcessTap`), not ScreenCaptureKit, so the permission it
// needs is "System Audio Recording Only" (`kTCCServiceAudioCapture`) rather
// than Screen Recording. `CGPreflightScreenCaptureAccess()` answers for Screen
// Recording, which this app never requests, so its answer says nothing about
// whether the tap will work.
//
// Core Audio taps have no preflight API of their own, and the only way to test
// one is to build a tap, which means touching the audio graph. So the
// system-audio permission is not guessed at up front: `sck_start_capture`
// reports its real failure, and that is what the UI surfaces.

/// 0 = denied/undetermined, 1 = granted.
@_cdecl("perm_check_microphone")
func permCheckMicrophone() -> Bool {
    return AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
}

/// Asks for microphone access, blocking until the user answers. Called off the
/// UI thread from Rust; the semaphore is why this must never run on main.
@_cdecl("perm_request_microphone")
func permRequestMicrophone() -> Bool {
    let status = AVCaptureDevice.authorizationStatus(for: .audio)
    if status == .authorized {
        return true
    }
    if status != .notDetermined {
        return false  // denied/restricted: only System Settings can change it
    }
    var granted = false
    let semaphore = DispatchSemaphore(value: 0)
    AVCaptureDevice.requestAccess(for: .audio) { allowed in
        granted = allowed
        semaphore.signal()
    }
    semaphore.wait()
    return granted
}

/// Opens the relevant System Settings pane. `kind` is "microphone" or
/// "system_audio". System audio lives under the Screen Recording pane on
/// macOS -- Apple lists "System Audio Recording Only" there, which is why the
/// anchor is `Privacy_ScreenCapture` even though this app never records the
/// screen.
@_cdecl("perm_open_settings")
func permOpenSettings(kind: UnsafePointer<CChar>) {
    let which = String(cString: kind)
    let anchor = which == "microphone"
        ? "Privacy_Microphone"
        : "Privacy_ScreenCapture"
    guard let url = URL(
        string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)"
    ) else { return }
    NSWorkspace.shared.open(url)
}

@_cdecl("sck_start_capture")
func sckStartCapture(callback: @escaping AudioCallback, errorCallback: @escaping AudioErrorCallback) -> Bool {
    sckStateLock.lock()
    defer { sckStateLock.unlock() }

    captureCallback = callback
    captureErrorCallback = errorCallback

    guard buildSckCaptureGraph() else {
        captureCallback = nil
        captureErrorCallback = nil
        return false
    }

    registerSckDefaultOutputListener()
    return true
}

@_cdecl("sck_stop_capture")
func sckStopCapture() {
    sckStateLock.lock()
    defer { sckStateLock.unlock() }

    tearDownSckGraph()
    unregisterSckDefaultOutputListener()
    sckPendingRebuildWorkItem?.cancel()
    sckPendingRebuildWorkItem = nil
    captureCallback = nil
    captureErrorCallback = nil
    NSLog("VoxTrans: Audio capture stopped")
}

/// Tears down the tap + aggregate device + engine, but leaves
/// `captureCallback`/`captureErrorCallback` and the default-output listener
/// registration alone -- both the rebuild path (which needs the callback to
/// keep delivering audio through the *new* graph) and a build failure
/// inside `buildSckCaptureGraph` (which needs to report through
/// `captureErrorCallback` when called from the rebuild path) rely on this.
/// Only `sck_stop_capture` clears those separately. Caller must hold
/// `sckStateLock`.
private func tearDownSckGraph() {
    if let engine = audioEngine {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        audioEngine = nil
    }
    unregisterSckConfigChangeObserver()

    if aggregateDeviceID != kAudioObjectUnknown {
        AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
        aggregateDeviceID = kAudioObjectUnknown
    }

    if tapID != kAudioObjectUnknown {
        AudioHardwareDestroyProcessTap(tapID)
        tapID = kAudioObjectUnknown
    }
}

/// Pure: true if the most recent entry in `recentTimestamps` (CoreFoundation
/// absolute-time seconds, same clock as `CFAbsoluteTimeGetCurrent()`) falls
/// within `window` seconds before `now`. Decides whether a rebuild should
/// arm the post-rebuild watchdog at all (T037 Critic round 2 blocker): if
/// audio was **not** already flowing right before the rebuild -- a quiet
/// room, or listen mode started before a meeting has produced any audio at
/// all, both completely normal -- silence afterward proves nothing and must
/// never be reported as a failure. Only a rebuild that interrupts audio
/// which *was* actually flowing is suspicious enough to warn about.
/// Verified via both of its branches behaviorally (real audio timing, not a
/// hermetic unit test -- this file has no isolated Swift test target, D9),
/// in `test_device_change.py`'s armed/not-armed watchdog tests.
private func sckWasRecentlyFlowing(recentTimestamps: [TimeInterval], now: TimeInterval, window: TimeInterval) -> Bool {
    guard let mostRecent = recentTimestamps.max() else { return false }
    return now - mostRecent <= window
}

/// T037 Critic round 2 major: schedules a one-shot check,
/// `sckRebuildAudioWatchdogTimeout` after a *rebuild* that just succeeded
/// -- never after the initial build (round 1's blocker: arming
/// unconditionally on every build, including the first, reported a
/// perfectly healthy silent listen-mode session as broken at 2.3s). Only
/// called by `handleSckDeviceChange` when `sckWasRecentlyFlowing` found
/// audio was genuinely flowing right before this rebuild. `generation` pins
/// the check to the specific build that scheduled it -- if a later rebuild
/// (or `sck_stop_capture`) has already superseded it by the time the timer
/// fires, this is a no-op.
///
/// Non-destructive: reports a **warning** (`"WARN: "`-prefixed message,
/// through the same `captureErrorCallback` a hard failure uses) and leaves
/// capture running exactly as it is -- Rust's `wait_for_stop_or_rebuild_error`
/// treats a `WARN:`-prefixed message as "surface and keep going", not "end
/// the session" (round 1's watchdog stopped capture outright on the same
/// signal, which is what made round 1's false positive destructive rather
/// than merely noisy).
private func scheduleRebuildAudioWatchdog(generation: UInt64) {
    sckDeviceChangeQueue.asyncAfter(deadline: .now() + sckRebuildAudioWatchdogTimeout) {
        sckStateLock.lock()
        let isCurrentBuild = generation == sckBuildGeneration && captureCallback != nil
        let receivedAudio = sckReceivedAudioSinceLastBuild
        let errorCB = captureErrorCallback
        sckStateLock.unlock()

        guard isCurrentBuild, !receivedAudio else { return }

        NSLog(
            "VoxTrans: system audio capture rebuilt but produced no audio within "
                + "\(sckRebuildAudioWatchdogTimeout)s despite audio flowing just before the "
                + "rebuild -- reporting as a warning (capture remains active)"
        )
        let message = "WARN: Audio capture rebuilt after a device change but has not produced audio within \(Int(sckRebuildAudioWatchdogTimeout))s. Capture is still active; restart it if the problem continues."
        if let errorCB = errorCB {
            message.withCString { cString in
                errorCB(cString)
            }
        }
    }
}

/// T037 Critic round 2 major: a default-output change that only alters the
/// already-default device's own rate (no listener fires for this shape --
/// see `sckRatePollInterval`'s doc comment) is caught by polling instead.
/// `generation` pins each tick to the build that started this poll chain;
/// a rebuild increments `sckBuildGeneration` and starts its own fresh chain
/// (from `buildSckCaptureGraph`), so a stale chain's next tick finds itself
/// superseded and stops rescheduling itself, with no separate cancellation
/// token needed (mirrors the watchdog's and ignore-first fallback's
/// generation pattern). Caller must hold `sckStateLock` is **not** required
/// -- this reads/writes only through the same lock internally, matching
/// every other cross-thread entry point in this file.
private func scheduleSckRatePoll(generation: UInt64) {
    sckDeviceChangeQueue.asyncAfter(deadline: .now() + sckRatePollInterval) {
        sckStateLock.lock()
        let stillCurrent = generation == sckBuildGeneration && captureCallback != nil
        let aggID = aggregateDeviceID
        sckStateLock.unlock()

        guard stillCurrent else { return }

        if let defaultOutputID = systemDefaultOutputDeviceID(),
           let currentDefaultRate = nominalSampleRate(of: defaultOutputID),
           let aggregateRate = nominalSampleRate(of: aggID),
           currentDefaultRate != aggregateRate {
            NSLog(
                "VoxTrans: nominal-rate poll detected a mismatch (default output = "
                    + "\(currentDefaultRate) Hz, aggregate = \(aggregateRate) Hz) -- rebuilding"
            )
            scheduleSckDeviceChangeRebuild(reason: "nominal rate poll mismatch")
            // A new poll chain starts from the rebuild's own successful
            // `buildSckCaptureGraph()` call -- rescheduling this stale
            // generation's own chain here would just be a wasted tick.
            return
        }

        scheduleSckRatePoll(generation: generation)
    }
}

/// Coalesces a burst of near-simultaneous device-change notifications
/// (either kind, or the rate poller) into a single `handleSckDeviceChange`
/// call, run `sckRebuildDebounceInterval` after the most recent one -- see
/// the `sckPendingRebuildWorkItem` doc comment for why. Every trigger site
/// in this file calls this instead of invoking `handleSckDeviceChange`
/// directly.
private func scheduleSckDeviceChangeRebuild(reason: String) {
    sckStateLock.lock()
    sckPendingRebuildWorkItem?.cancel()
    let workItem = DispatchWorkItem {
        handleSckDeviceChange(reason: reason)
    }
    sckPendingRebuildWorkItem = workItem
    sckStateLock.unlock()
    sckDeviceChangeQueue.asyncAfter(deadline: .now() + sckRebuildDebounceInterval, execute: workItem)
}

/// T035 (US4, research.md D6): fires whenever the system default output
/// device changes, this specific capture engine reports a Core Audio
/// configuration change, or the nominal-rate poller detects a mismatch --
/// the three mechanisms this file has for the ways the reported bug
/// (default output switching mid-session) could manifest. Rebuilds the
/// whole graph (tear down, then `buildSckCaptureGraph()` again) rather
/// than attempting a partial repair (research.md D6's rationale: the tap,
/// aggregate device and engine are all bound to device identity at
/// construction). A failed *build* is reported through `captureErrorCallback`
/// as a hard error (FR-008) rather than left to silently end the session;
/// a build that *succeeds* but was expected to keep producing audio it was
/// already producing is instead handed to `scheduleRebuildAudioWatchdog`,
/// conditionally (T037 Critic round 2 blocker -- see that function and
/// `sckWasRecentlyFlowing`).
private func handleSckDeviceChange(reason: String) {
    sckStateLock.lock()
    defer { sckStateLock.unlock() }

    // A listener can still be in-flight for a brief window during/after
    // `sck_stop_capture` -- only rebuild while a session is actually
    // running (`captureCallback` is cleared by `sck_stop_capture` under the
    // same lock, so this check and the teardown below are consistent).
    guard captureCallback != nil else { return }

    // T037 Critic round 2 blocker: decided *before* tearing down, from the
    // graph that is about to be replaced -- this is exactly "was audio
    // flowing right before this rebuild", not "is it flowing now" (which
    // would always be false mid-teardown) or "was it ever flowing" (which
    // would arm even for a session that went quiet minutes ago).
    let wasFlowing = sckWasRecentlyFlowing(
        recentTimestamps: sckRecentCallbackTimestamps,
        now: CFAbsoluteTimeGetCurrent(),
        window: sckPreRebuildFlowingWindow
    )

    NSLog("VoxTrans: system audio device change detected (\(reason)) -- rebuilding capture graph")
    tearDownSckGraph()
    if buildSckCaptureGraph() {
        NSLog("VoxTrans: system audio capture graph rebuilt successfully after \(reason)")
        if wasFlowing {
            scheduleRebuildAudioWatchdog(generation: sckBuildGeneration)
        }
    } else {
        NSLog("VoxTrans: system audio capture graph rebuild FAILED after \(reason)")
        let message = "Audio capture could not recover after a device change (\(reason)). Please restart capture."
        if let errorCB = captureErrorCallback {
            message.withCString { cString in
                errorCB(cString)
            }
        }
    }
}

/// Registers the `kAudioHardwarePropertyDefaultOutputDevice` listener once
/// per `sck_start_capture` session (idempotent -- a second call while one is
/// already registered is a no-op). Delivered on `sckDeviceChangeQueue`.
/// Caller must hold `sckStateLock`.
private func registerSckDefaultOutputListener() {
    guard sckDefaultOutputListenerBlock == nil else { return }

    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    let block: AudioObjectPropertyListenerBlock = { _, _ in
        scheduleSckDeviceChangeRebuild(reason: "default output device changed")
    }
    let status = AudioObjectAddPropertyListenerBlock(
        AudioObjectID(kAudioObjectSystemObject),
        &address,
        sckDeviceChangeQueue,
        block
    )
    if status == noErr {
        sckDefaultOutputListenerBlock = block
    } else {
        NSLog("VoxTrans: failed to register default-output-device listener: \(status)")
    }
}

/// Caller must hold `sckStateLock`.
private func unregisterSckDefaultOutputListener() {
    guard let block = sckDefaultOutputListenerBlock else { return }

    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    AudioObjectRemovePropertyListenerBlock(
        AudioObjectID(kAudioObjectSystemObject),
        &address,
        sckDeviceChangeQueue,
        block
    )
    sckDefaultOutputListenerBlock = nil
}

/// Scoped to a specific `AVAudioEngine` instance (`object: engine`), not
/// `nil` -- an unscoped observer would also fire for the *other* engines
/// this file owns (`mic_start_capture`'s, `audio_output_start`'s),
/// triggering unnecessary sck rebuilds on unrelated activity. Re-registered
/// on every successful `buildSckCaptureGraph()` call, since each rebuild
/// creates a new `AVAudioEngine` instance.
///
/// T037 Critic round 1 major 3: registers **immediately** (no delay) and
/// instead ignores exactly the first notification actually delivered for
/// this engine (`sckIgnoreNextConfigChangeNotification`) -- `engine.start()`
/// itself reliably posts one such notification for the engine that just
/// started (measured: every single `buildSckCaptureGraph()` call, ~95ms
/// after start), and an earlier, time-based version of this guard (a fixed
/// 0.5s delay before registering at all) permanently missed any genuine
/// change landing inside that window. (Measured in round 2: a rate-only
/// default-output change, e.g. an AirPods A2DP<->HFP profile switch, does
/// **not** reliably fire this notification on this machine either -- the
/// nominal-rate poller, `scheduleSckRatePoll`, is that transition's actual
/// coverage. This observer remains for whatever Core Audio configuration
/// changes genuinely do post it.) `sckIgnoreNextConfigChangeFallbackTimeout`
/// clears the ignore flag automatically if the expected self-notification
/// never arrives at all, so a missing notification can never leave a
/// *later*, genuine change permanently swallowed. Caller must hold
/// `sckStateLock`.
private func registerSckConfigChangeObserver(for engine: AVAudioEngine) {
    unregisterSckConfigChangeObserver()
    sckIgnoreNextConfigChangeNotification = true
    let myGeneration = sckBuildGeneration
    sckConfigChangeObserver = NotificationCenter.default.addObserver(
        forName: .AVAudioEngineConfigurationChange,
        object: engine,
        queue: nil
    ) { _ in
        sckStateLock.lock()
        let shouldIgnore = sckIgnoreNextConfigChangeNotification
        sckIgnoreNextConfigChangeNotification = false
        sckStateLock.unlock()

        guard !shouldIgnore else { return }
        scheduleSckDeviceChangeRebuild(reason: "AVAudioEngineConfigurationChange")
    }
    sckDeviceChangeQueue.asyncAfter(deadline: .now() + sckIgnoreNextConfigChangeFallbackTimeout) {
        sckStateLock.lock()
        if sckBuildGeneration == myGeneration {
            sckIgnoreNextConfigChangeNotification = false
        }
        sckStateLock.unlock()
    }
}

/// Caller must hold `sckStateLock`.
private func unregisterSckConfigChangeObserver() {
    sckIgnoreNextConfigChangeNotification = false
    guard let observer = sckConfigChangeObserver else { return }
    NotificationCenter.default.removeObserver(observer)
    sckConfigChangeObserver = nil
}

// ========== Microphone Capture ==========

private var micEngine: AVAudioEngine?
private var micCallback: AudioCallback?

@_cdecl("mic_start_capture")
func micStartCapture(callback: @escaping AudioCallback, deviceUID: UnsafePointer<CChar>) -> Bool {
    let uid = String(cString: deviceUID)
    micCallback = callback

    let engine = AVAudioEngine()

    do {
        let inputNode = engine.inputNode

        // An empty UID means "system default" (current behaviour, now an
        // explicit choice). A non-empty UID that cannot be resolved MUST
        // fail the start and MUST NOT fall back to the default -- that
        // silent fallback is the confirmed root cause this story fixes.
        // The device must be set on the input unit *before* the format is
        // read (contracts/audio-devices.md "Selection ordering"), mirroring
        // sck_start_capture above.
        if !uid.isEmpty {
            guard let deviceID = findAudioDeviceByUID(uid: uid) else {
                NSLog("VoxTrans Mic: Device not found for UID: \(uid)")
                micCallback = nil
                return false
            }
            let audioUnit = inputNode.audioUnit!
            var devID = deviceID
            let setDeviceStatus = AudioUnitSetProperty(
                audioUnit,
                kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global,
                0,
                &devID,
                UInt32(MemoryLayout<AudioObjectID>.size)
            )
            guard setDeviceStatus == noErr else {
                NSLog("VoxTrans Mic: Failed to set input device: \(setDeviceStatus)")
                micCallback = nil
                return false
            }
            NSLog("VoxTrans Mic: Set input device ID: \(deviceID) for UID: \(uid)")
        }

        let inputFormat = inputNode.outputFormat(forBus: 0)
        NSLog("VoxTrans Mic: Input format: \(inputFormat)")

        let targetSampleRate: Double = 16000
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        )!

        let converter = AVAudioConverter(from: inputFormat, to: targetFormat)!

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
            guard let cb = micCallback else { return }

            let frameCount = buffer.frameLength
            let ratio = targetSampleRate / inputFormat.sampleRate
            let outputFrameCount = AVAudioFrameCount(Double(frameCount) * ratio)

            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
                return
            }

            var error: NSError?
            let convertStatus = converter.convert(to: outputBuffer, error: &error) { inNumPackets, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            guard convertStatus != .error, error == nil else {
                NSLog("VoxTrans Mic: Conversion error: \(error?.localizedDescription ?? "unknown")")
                return
            }

            if let channelData = outputBuffer.floatChannelData?[0] {
                let sampleCount = Int(outputBuffer.frameLength)
                if sampleCount > 0 {
                    cb(channelData, sampleCount, Int(targetSampleRate), 1)
                }
            }
        }

        try engine.start()
        micEngine = engine
        NSLog("VoxTrans: Microphone capture started")
        return true

    } catch {
        NSLog("VoxTrans: Failed to start mic engine: \(error)")
        micStopCapture()
        return false
    }
}

@_cdecl("mic_stop_capture")
func micStopCapture() {
    if let engine = micEngine {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        micEngine = nil
    }
    micCallback = nil
    NSLog("VoxTrans: Microphone capture stopped")
}

// ========== Audio Output (TTS playback to BlackHole) ==========

private var outputEngine: AVAudioEngine?
private var playerNode: AVAudioPlayerNode?

// T025 (data-model.md §3 `PlaybackLifecycle.scheduled_buffers`):
// incremented when a buffer is scheduled, decremented once AVAudioEngine
// reports it has actually been heard (`.dataPlayedBack`, not the default
// `.dataConsumed`, which fires when the buffer merely reaches the renderer
// -- contracts/playback-protocol.md). Written from two different threads:
// `audioOutputEnqueue` runs on whatever thread Rust's `start_tts_playback`
// polling loop calls it from; the completion handler runs on
// AVAudioEngine's own internal completion queue. `outstandingLock`
// serializes both sides.
private let outstandingLock = NSLock()
private var outstandingBuffers: Int = 0

@_cdecl("audio_output_start")
func audioOutputStart(deviceUID: UnsafePointer<CChar>) -> Bool {
    let uid = String(cString: deviceUID)
    NSLog("VoxTrans Output: Starting on device UID: \(uid)")

    let engine = AVAudioEngine()
    let player = AVAudioPlayerNode()
    engine.attach(player)

    // The output device MUST be resolved and set on the output unit
    // *before* `engine.mainMixerNode` is ever touched
    // (contracts/audio-devices.md "Selection ordering", FR-005, research.md
    // D7): accessing the mixer lazily instantiates it and forms the
    // mixer->output connection at the then-current device's format, so
    // setting the device afterwards leaves the graph built for the wrong
    // device -- the confirmed cause of silent output loss (mirrors
    // mic_start_capture's ordering at :183-207 and sck_start_capture's at
    // :63-80). An empty UID means "system default" and is left as-is.
    if !uid.isEmpty {
        guard let deviceID = findAudioDeviceByUID(uid: uid) else {
            NSLog("VoxTrans Output: Device not found for UID: \(uid)")
            return false
        }
        let audioUnit = engine.outputNode.audioUnit!
        var devID = deviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &devID,
            UInt32(MemoryLayout<AudioObjectID>.size)
        )
        guard status == noErr else {
            NSLog("VoxTrans Output: Failed to set output device: \(status)")
            return false
        }
        NSLog("VoxTrans Output: Set output device ID: \(deviceID)")
    }

    // TTS output format: 24kHz mono float32 -- what audio_output_enqueue
    // builds buffers as (:345-368). Player -> mixer stays fixed at this
    // format regardless of device; the device's real format governs
    // mixer -> output instead, read only now that the device has already
    // been set above.
    let playerFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 24000,
        channels: 1,
        interleaved: false
    )!
    engine.connect(player, to: engine.mainMixerNode, format: playerFormat)

    let deviceFormat = engine.outputNode.outputFormat(forBus: 0)
    NSLog("VoxTrans Output: Output format: \(deviceFormat)")
    engine.connect(engine.mainMixerNode, to: engine.outputNode, format: deviceFormat)

    do {
        try engine.start()
        player.play()
        outputEngine = engine
        playerNode = player
        // Fresh session -- nothing from a previous session should carry
        // over into this one's completion count.
        outstandingLock.lock()
        outstandingBuffers = 0
        outstandingLock.unlock()
        NSLog("VoxTrans Output: Audio output started")
        return true
    } catch {
        NSLog("VoxTrans Output: Failed to start engine: \(error)")
        return false
    }
}

@_cdecl("audio_output_stop")
func audioOutputStop() {
    playerNode?.stop()
    if let engine = outputEngine {
        engine.stop()
    }
    outputEngine = nil
    playerNode = nil
    outstandingLock.lock()
    outstandingBuffers = 0
    outstandingLock.unlock()
    NSLog("VoxTrans Output: Audio output stopped")
}

// T025: Rust's `start_tts_playback` polls this every tick
// (`audio_output.rs`) to drive `PlaybackLifecycle` -- see
// `playback_lifecycle.rs`'s doc comment for why a polled counter was
// chosen over a Swift->Rust callback registration.
@_cdecl("audio_output_outstanding_buffers")
func audioOutputOutstandingBuffers() -> Int {
    outstandingLock.lock()
    defer { outstandingLock.unlock() }
    return outstandingBuffers
}

@_cdecl("audio_output_enqueue")
func audioOutputEnqueue(data: UnsafePointer<Float>, numSamples: Int) {
    guard let player = playerNode, let engine = outputEngine, engine.isRunning else {
        return
    }

    let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 24000,
        channels: 1,
        interleaved: false
    )!

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(numSamples)) else {
        return
    }
    buffer.frameLength = AVAudioFrameCount(numSamples)

    if let channelData = buffer.floatChannelData?[0] {
        channelData.update(from: data, count: numSamples)
    }

    outstandingLock.lock()
    outstandingBuffers += 1
    outstandingLock.unlock()

    // `.dataPlayedBack`, not the default `.dataConsumed`: the latter fires
    // when this buffer is handed to the renderer, not when it has actually
    // finished sounding, which is the exact bug (T014/US1) this feature's
    // completion tracking must not reproduce in a new place
    // (contracts/playback-protocol.md).
    player.scheduleBuffer(
        buffer,
        at: nil,
        options: [],
        completionCallbackType: .dataPlayedBack
    ) { _ in
        outstandingLock.lock()
        outstandingBuffers -= 1
        outstandingLock.unlock()
    }
}

@_cdecl("audio_list_output_devices")
func audioListOutputDevices() -> UnsafeMutablePointer<CChar>? {
    var propertyAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )

    var dataSize: UInt32 = 0
    var status = AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0, nil,
        &dataSize
    )
    guard status == noErr else { return nil }

    let deviceCount = Int(dataSize) / MemoryLayout<AudioObjectID>.size
    var deviceIDs = [AudioObjectID](repeating: 0, count: deviceCount)
    status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0, nil,
        &dataSize,
        &deviceIDs
    )
    guard status == noErr else { return nil }

    var results: [String] = []

    for deviceID in deviceIDs {
        // Check if device has output streams
        var streamAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioObjectPropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
        var streamSize: UInt32 = 0
        let streamStatus = AudioObjectGetPropertyDataSize(deviceID, &streamAddress, 0, nil, &streamSize)
        if streamStatus != noErr || streamSize == 0 {
            continue // Not an output device
        }

        // Get device name
        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var name: CFString = "" as CFString
        var nameSize = UInt32(MemoryLayout<CFString>.size)
        AudioObjectGetPropertyData(deviceID, &nameAddress, 0, nil, &nameSize, &name)

        // Get device UID
        var uidAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: CFString = "" as CFString
        var uidSize = UInt32(MemoryLayout<CFString>.size)
        AudioObjectGetPropertyData(deviceID, &uidAddress, 0, nil, &uidSize, &uid)

        let deviceName = name as String
        let deviceUID = uid as String

        if !deviceName.isEmpty && !deviceUID.isEmpty {
            results.append("\(deviceName)|\(deviceUID)")
        }
    }

    let joined = results.joined(separator: "\n")
    return strdup(joined)
}

@_cdecl("audio_list_input_devices")
func audioListInputDevices() -> UnsafeMutablePointer<CChar>? {
    var propertyAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )

    var dataSize: UInt32 = 0
    var status = AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0, nil,
        &dataSize
    )
    guard status == noErr else { return nil }

    let deviceCount = Int(dataSize) / MemoryLayout<AudioObjectID>.size
    var deviceIDs = [AudioObjectID](repeating: 0, count: deviceCount)
    status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0, nil,
        &dataSize,
        &deviceIDs
    )
    guard status == noErr else { return nil }

    var results: [String] = []

    for deviceID in deviceIDs {
        // Check if device has input streams
        var streamAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var streamSize: UInt32 = 0
        let streamStatus = AudioObjectGetPropertyDataSize(deviceID, &streamAddress, 0, nil, &streamSize)
        if streamStatus != noErr || streamSize == 0 {
            continue // Not an input device
        }

        // Get device name. Unlike audio_list_output_devices above (left
        // as-is; see evidence/US3.md), this reads into an `Unmanaged<CFString>?`
        // rather than taking the address of a `CFString` variable directly --
        // the latter is what the compiler warns about ("forming
        // 'UnsafeMutableRawPointer' to a variable of type 'CFString'; this is
        // likely incorrect because 'CFString' may contain an object
        // reference"), since `&name` there hands Core Audio a raw pointer to
        // a Swift-managed object reference. `Unmanaged<CFString>?` is a plain
        // struct wrapping a pointer, so `&name` here is not that -- ownership
        // is taken explicitly via `takeRetainedValue()` once Core Audio has
        // written the (already +1) CFStringRef into it.
        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var name: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<CFString?>.size)
        AudioObjectGetPropertyData(deviceID, &nameAddress, 0, nil, &nameSize, &name)

        // Get device UID (same pattern as the name read above).
        var uidAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: Unmanaged<CFString>?
        var uidSize = UInt32(MemoryLayout<CFString?>.size)
        AudioObjectGetPropertyData(deviceID, &uidAddress, 0, nil, &uidSize, &uid)

        let deviceName = (name?.takeRetainedValue() as String?) ?? ""
        let deviceUID = (uid?.takeRetainedValue() as String?) ?? ""

        if !deviceName.isEmpty && !deviceUID.isEmpty {
            results.append("\(deviceName)|\(deviceUID)")
        }
    }

    let joined = results.joined(separator: "\n")
    return strdup(joined)
}

/// Find an AudioObjectID by its UID string
private func findAudioDeviceByUID(uid: String) -> AudioObjectID? {
    var propertyAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )

    var dataSize: UInt32 = 0
    var status = AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0, nil,
        &dataSize
    )
    guard status == noErr else { return nil }

    let deviceCount = Int(dataSize) / MemoryLayout<AudioObjectID>.size
    var deviceIDs = [AudioObjectID](repeating: 0, count: deviceCount)
    status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0, nil,
        &dataSize,
        &deviceIDs
    )
    guard status == noErr else { return nil }

    for deviceID in deviceIDs {
        var uidAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceUID: CFString = "" as CFString
        var uidSize = UInt32(MemoryLayout<CFString>.size)
        AudioObjectGetPropertyData(deviceID, &uidAddress, 0, nil, &uidSize, &deviceUID)

        if (deviceUID as String) == uid {
            return deviceID
        }
    }

    return nil
}
