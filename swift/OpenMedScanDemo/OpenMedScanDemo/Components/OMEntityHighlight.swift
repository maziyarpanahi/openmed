import SwiftUI

/// Renders body text with each detected entity wrapped in a paper-marker
/// highlight (yellow fill, ink underline, small mono bracket tag).
/// Uses `AttributedString` so the inline flow remains native-selectable.
public struct OMEntityHighlight: View {
    public let text: String
    public let entities: [DetectedEntity]
    public var bodyFont: Font = .om.body(16)
    public var showsLabels: Bool = true

    public init(text: String, entities: [DetectedEntity], bodyFont: Font = .om.body(16), showsLabels: Bool = true) {
        self.text = text
        self.entities = entities
        self.bodyFont = bodyFont
        self.showsLabels = showsLabels
    }

    public var body: some View {
        Text(attributed)
            .font(bodyFont)
            .foregroundStyle(Color.omInk)
            .lineSpacing(4)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var attributed: AttributedString {
        var out = AttributedString(text)
        let sorted = entities.sorted { $0.start < $1.start }
        let count = text.count

        for entity in sorted {
            let safeStart = min(max(entity.start, 0), count)
            let safeEnd = min(max(entity.end, safeStart), count)
            guard safeStart < safeEnd else { continue }
            let startIndex = out.index(out.startIndex, offsetByCharacters: safeStart)
            let endIndex = out.index(out.startIndex, offsetByCharacters: safeEnd)
            let range = startIndex..<endIndex
            out[range].backgroundColor = .omHighlight
            out[range].foregroundColor = .omInk
            out[range].underlineStyle = .single
        }

        if showsLabels {
            for entity in sorted.reversed() {
                let safeEnd = min(max(entity.end, 0), count)
                guard safeEnd <= count else { continue }
                let insertAt = out.index(out.startIndex, offsetByCharacters: safeEnd)
                var tag = AttributedString(" ⟨\(entity.category.displayName.uppercased())⟩")
                tag.font = .om.mono(9, weight: .semibold)
                tag.foregroundColor = entity.category.tone.accent
                out.insert(tag, at: insertAt)
            }
        }
        return out
    }
}
