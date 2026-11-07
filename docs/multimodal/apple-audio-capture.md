# Consent-bound Apple audio capture

`AmbientAudioCapture` binds an injected audio source to the existing [ambient consent state machine](ambient-session-policy.md). Its private session requires every declared participant's consent before starting. OS microphone permission is a separate requirement. Approving a permission prompt never starts recording or supplies participant consent.

The native `AppleMicrophoneSource` uses `AVAudioEngine` and `AVAudioApplication` on the supported Apple platforms. Construction installs lifecycle observers but does not create an audio engine, install a microphone tap, request permission or record. Permission requests and recording starts require explicit calls. The host application must supply `NSMicrophoneUsageDescription` and any platform sandbox microphone entitlement. On iOS, watchOS and visionOS, the host also owns audio-session category and activation; configure a recording-capable `AVAudioSession` only as part of the approved foreground capture flow, and restore the host's policy on shutdown. The library does not overwrite global audio-session settings.

## Capture boundaries

| Trigger | Capture behavior | Required next action |
| --- | --- | --- |
| Missing participant consent | Source is not started | Obtain consent from the complete declared cohort |
| Undetermined, denied or unavailable microphone permission | Start is refused | Explicit permission request or OS settings change, then explicit start |
| Pause | Source stopped; pending buffers cleared | Explicit `start(resuming: true)` |
| Interruption, device-route change or foreground loss | Source stopped; session interrupted | Recheck foreground/device/permission, then explicit resume |
| Interruption ends or permission becomes granted | Remains stopped | Explicit start/resume appropriate to session state |
| Participant change | Stop source, invalidate previous cohort, clear old data | Fresh consent from every new participant, then explicit start |
| Stop | Session stopped; cannot restart | Create a new session for new capture |
| Withdrawal, expiry or consumer failure | Terminal session; no later admission | New session with new consent if appropriate |

UIKit and AppKit foreground-loss notifications are connected by the native source. Audio-session interruptions and route changes are connected on iOS, watchOS and visionOS; engine configuration changes also interrupt capture. Hosts must forward any additional suspension boundary, including watch application lifecycle callbacks, through `capture.interrupt(.background)`. Waking the application never resumes capture. The original non-renewable session deadline is checked before every delivery and by an idle timer; process suspension may delay that timer. If automatic expiry is disabled, the host must schedule `expire()` reliably, including on wake.

Each recording attempt has a distinct epoch. Pausing, interrupting or withdrawing invalidates the previous epoch before stopping its source. Delayed samples or events from an old attempt cannot enter a resumed capture. Local consumer callbacks and lifecycle changes are serialized: a withdrawal waits for an already admitted short synchronous callback to finish, clears its owned buffer, then returns. It cannot preempt arbitrary application code that blocks forever. Callbacks must remain short and must not call capture lifecycle methods themselves.

## Local PCM and ownership

`CapturePCMBuffer` owns a mutable mono Float32 allocation with 1–16,384 frames, a sample rate from 8–192 kHz, and the original nonnegative sample offset. The native source accepts noninterleaved Float32 input with up to eight channels and averages channels into mono. It keeps at most one pending/delivered PCM buffer and one temporary mono conversion allocation. Both allocations are bounded; OS audio-engine storage is outside this inventory. Unsupported formats, nonfinite samples or invalid timing stop admission instead of producing empty successful audio.

The consumer borrows samples through `AmbientPCMConsumer.consume`. The session policy's byte limit applies before consumer admission. Owned samples are overwritten after each callback, including failure, rejection, stale callbacks, shutdown and deinitialization. This adapter retains no raw audio between calls and offers no audio export API. A stricter application must separately bound the native producer's temporary allocations and downstream model state.

Backpressure drops a callback instead of waiting on a busy consumer or accumulating an unbounded queue. Original source sample offsets and capture epochs remain visible, so the consumer can detect gaps and must not stitch them into supposedly continuous speech. No resampling, inferred acoustic alignment, speaker-role assignment or ASR model is introduced.

`snapshot()` exposes controlled reasons and counts: received, cleared, admitted and currently owned buffers, source cleanup status, permission and the underlying consent state. `cleanupComplete` requires stopped capture, no uncleared delivered buffers and successful source cleanup. A source cleanup failure is sticky and prevents restart. These claims cover owned mutable memory and the source's shutdown contract. They do not prove physical RAM erasure, OS buffer destruction or deletion of copies made by a consumer. Do not retain the borrowed pointer, log/copy/export its contents or send samples to a remote service. Model caches and downstream results need their own retention and privacy controls.

## Host integration

```swift
import OpenMedKit

// LocalASRConsumer is the application's short, synchronous local consumer.
let source = AppleMicrophoneSource()
let capture = try AmbientAudioCapture(
    sessionRef: opaqueSessionRef,
    participants: opaqueParticipantRefs,
    policy: AmbientSessionPolicy(revision: approvedPolicySHA256),
    source: source,
    consumer: LocalASRConsumer())

// From an explicit user action, request OS permission if needed.
capture.requestPermission { permission in
    // Update the visible UI only. Never start automatically here.
}
for participant in opaqueParticipantRefs {
    try capture.consent(participant)
}
// The host now checks permission/foreground and configures its audio session.
try capture.start()
try capture.pause()
// Later, from a separate explicit user action:
try capture.start(resuming: true)
capture.withdraw()
assert(capture.snapshot().cleanupComplete)
```

The application must keep visible recording controls and a live reference to `capture`, call the appropriate lifecycle method on every host boundary, and delete it deterministically when finished. This is a native capture binding, not a recorder UI, recording-law assessment, clinical approval or remote transport.

## Validation

`AmbientAudioCaptureTests` uses synthetic PCM and injected permission, source and consumer controls. It exercises consent and denial, pause/resume, all interruption types, participant changes, stale callbacks, original deadline expiry, permission revocation, startup failure, bounded buffers, consumer failure, cleanup failure, concurrent withdrawal and deinitialization. No test opens a microphone or requests system permission.

```bash
cd swift/OpenMedKit
swift test --disable-automatic-resolution --filter AmbientAudioCaptureTests
```

Apple references: [record permission](https://developer.apple.com/documentation/avfaudio/avaudioapplication/requestrecordpermission(completionhandler:)), [audio taps](https://developer.apple.com/documentation/avfaudio/avaudionode/installtap(onbus:buffersize:format:block:)), and [route changes](https://developer.apple.com/documentation/avfaudio/responding-to-audio-route-changes).
