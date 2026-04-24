import SwiftUI

/// Namespaces for non-color design tokens.
/// Values mirror `openmed-design-system/project/colors_and_type.css`.
public enum OM {
    /// Spacing scale in points. `space1 = 4pt`, doubling/jumping per `colors_and_type.css`.
    public enum Space {
        public static let s1: CGFloat = 4
        public static let s2: CGFloat = 8
        public static let s3: CGFloat = 12
        public static let s4: CGFloat = 16
        public static let s5: CGFloat = 24
        public static let s6: CGFloat = 32
        public static let s7: CGFloat = 48
        public static let s8: CGFloat = 64
        public static let s9: CGFloat = 96
        public static let s10: CGFloat = 128
    }

    /// Corner radii in points.
    public enum Radius {
        public static let sm: CGFloat = 4
        public static let md: CGFloat = 8
        public static let lg: CGFloat = 14
        public static let pill: CGFloat = 999
    }

    /// Motion durations in seconds. Matches `--dur-micro / --dur-base / --dur-slow`.
    public enum Duration {
        public static let micro: Double = 0.16
        public static let base: Double = 0.24
        public static let slow: Double = 0.42
    }

    /// Stroke and hairline widths.
    public enum Stroke {
        public static let hairline: CGFloat = 1
        public static let focusRing: CGFloat = 2
    }
}
