import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Wraps `UIFeedbackGenerator` so view code can fire haptics without
/// remembering to create + prepare a generator each time.
@MainActor
public enum HapticsCenter {
    public static func selection() {
        #if canImport(UIKit)
        let g = UISelectionFeedbackGenerator()
        g.prepare()
        g.selectionChanged()
        #endif
    }

    public static func impact(_ style: ImpactStyle = .soft) {
        #if canImport(UIKit)
        let g = UIImpactFeedbackGenerator(style: style.uiStyle)
        g.prepare()
        g.impactOccurred()
        #endif
    }

    public static func notify(_ type: NotificationType) {
        #if canImport(UIKit)
        let g = UINotificationFeedbackGenerator()
        g.prepare()
        g.notificationOccurred(type.uiType)
        #endif
    }

    public enum ImpactStyle {
        case soft, medium, rigid
        #if canImport(UIKit)
        fileprivate var uiStyle: UIImpactFeedbackGenerator.FeedbackStyle {
            switch self {
            case .soft: return .soft
            case .medium: return .medium
            case .rigid: return .rigid
            }
        }
        #endif
    }

    public enum NotificationType {
        case success, warning, error
        #if canImport(UIKit)
        fileprivate var uiType: UINotificationFeedbackGenerator.FeedbackType {
            switch self {
            case .success: return .success
            case .warning: return .warning
            case .error:   return .error
            }
        }
        #endif
    }
}
