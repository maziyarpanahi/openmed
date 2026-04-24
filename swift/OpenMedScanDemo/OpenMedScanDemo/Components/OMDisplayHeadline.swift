import SwiftUI

/// Display headline — Newsreader serif at `fs-display-*` scale, with
/// optional italic-accent words (teal). Inputs are `(text, accent)` spans
/// so callers stay declarative: `OMDisplayHeadline([.plain("Bring your clinical "), .accent("text"), .plain(".")])`.
public struct OMDisplayHeadline: View {
    public enum Span { case plain(String), accent(String) }

    public enum Scale {
        case xl, lg, md, sm

        fileprivate var size: CGFloat {
            switch self {
            case .xl: return 48       // compressed for iPhone from web 72pt
            case .lg: return 40
            case .md: return 32
            case .sm: return 26
            }
        }

        fileprivate var tracking: CGFloat {
            switch self {
            case .xl: return -0.9
            case .lg: return -0.7
            case .md: return -0.5
            case .sm: return -0.3
            }
        }

        fileprivate var lineSpacing: CGFloat {
            switch self {
            case .xl: return -2
            case .lg: return -1
            case .md: return -0.5
            case .sm: return 0
            }
        }
    }

    private let spans: [Span]
    private let scale: Scale

    public init(_ spans: [Span], scale: Scale = .md) {
        self.spans = spans
        self.scale = scale
    }

    public var body: some View {
        let merged: Text = spans.reduce(Text("")) { running, span in
            switch span {
            case .plain(let s):
                return running + Text(s)
                    .font(.om.display(scale.size, weight: .medium))
                    .foregroundColor(.omInk)
            case .accent(let s):
                return running + Text(s)
                    .font(.om.display(scale.size, weight: .medium, italic: true))
                    .foregroundColor(.omTealAccent)
            }
        }
        return merged
            .tracking(scale.tracking)
            .lineSpacing(scale.lineSpacing)
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)
    }
}
