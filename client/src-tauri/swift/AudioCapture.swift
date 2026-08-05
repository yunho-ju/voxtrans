import Foundation
import CoreAudio
import AVFoundation
import AudioToolbox

// C callback type: (data_ptr, num_samples, sample_rate, channels)
typealias AudioCallback = @convention(c) (UnsafePointer<Float>, Int, Int, Int) -> Void

// ========== System Audio Capture (existing) ==========

// Global state for system audio capture
private var audioEngine: AVAudioEngine?
private var tapID: AudioObjectID = kAudioObjectUnknown
private var captureCallback: AudioCallback?
private var aggregateDeviceID: AudioObjectID = kAudioObjectUnknown

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

@_cdecl("sck_start_capture")
func sckStartCapture(callback: @escaping AudioCallback) -> Bool {
    captureCallback = callback

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
            cleanup()
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
        NSLog("VoxTrans: Audio capture started via Core Audio Tap")
        return true

    } catch {
        NSLog("VoxTrans: Failed to start audio engine: \(error)")
        cleanup()
        return false
    }
}

@_cdecl("sck_stop_capture")
func sckStopCapture() {
    cleanup()
    NSLog("VoxTrans: Audio capture stopped")
}

private func cleanup() {
    if let engine = audioEngine {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        audioEngine = nil
    }

    if aggregateDeviceID != kAudioObjectUnknown {
        AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
        aggregateDeviceID = kAudioObjectUnknown
    }

    if tapID != kAudioObjectUnknown {
        AudioHardwareDestroyProcessTap(tapID)
        tapID = kAudioObjectUnknown
    }

    captureCallback = nil
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
