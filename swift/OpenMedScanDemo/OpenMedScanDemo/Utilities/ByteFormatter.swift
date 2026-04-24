import Foundation

/// Friendly `"124.8 MB of 266.3 MB"` / `"62%"` formatting for downloads.
public enum ByteFormatter {
    private static let measurement: ByteCountFormatter = {
        let f = ByteCountFormatter()
        f.countStyle = .file
        f.allowedUnits = [.useMB, .useGB, .useKB]
        f.includesUnit = true
        f.allowsNonnumericFormatting = true
        return f
    }()

    public static func humanized(_ value: Int64) -> String {
        measurement.string(fromByteCount: max(0, value))
    }

    public static func humanized(_ value: Int64?) -> String {
        guard let value else { return "—" }
        return humanized(value)
    }

    public static func progressString(bytes: Int64, total: Int64?) -> String {
        let done = humanized(bytes)
        guard let total, total > 0 else { return done }
        return "\(done) of \(humanized(total))"
    }

    public static func rate(_ bytesPerSecond: Double?) -> String? {
        guard let bytesPerSecond, bytesPerSecond > 0 else { return nil }
        return "\(humanized(Int64(bytesPerSecond)))/s"
    }

    public static func percent(_ fraction: Double?) -> String {
        guard let fraction else { return "—" }
        return "\(Int((fraction * 100).rounded()))%"
    }
}
