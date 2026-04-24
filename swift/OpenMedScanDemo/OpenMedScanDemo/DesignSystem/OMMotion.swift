import SwiftUI

/// Motion tokens — three eases that match `--dur-micro/base/slow`
/// with the `cubic-bezier(0.2, 0.8, 0.2, 1)` curve (approximated by `.easeOut`).
public extension Animation {
    static var omMicro: Animation { .easeOut(duration: OM.Duration.micro) }
    static var omBase:  Animation { .easeOut(duration: OM.Duration.base) }
    static var omSlow:  Animation { .easeOut(duration: OM.Duration.slow) }
}
