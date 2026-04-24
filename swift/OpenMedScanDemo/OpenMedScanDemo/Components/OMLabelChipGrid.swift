import SwiftUI

/// Flow-layout grid of removable label chips plus an inline "+" affordance
/// that pops a text field for adding a new label. Comma or return commits.
public struct OMLabelChipGrid: View {
    @Binding public var labels: [String]
    public var maxLabels: Int = 32
    public var onChanged: (() -> Void)?

    @State private var draft: String = ""
    @FocusState private var draftFocused: Bool

    public init(labels: Binding<[String]>, maxLabels: Int = 32, onChanged: (() -> Void)? = nil) {
        self._labels = labels
        self.maxLabels = maxLabels
        self.onChanged = onChanged
    }

    public var body: some View {
        FlowLayout(spacing: 8, runSpacing: 8) {
            ForEach(Array(labels.enumerated()), id: \.offset) { index, label in
                OMChip(
                    label,
                    tone: .accent,
                    onRemove: { removeLabel(at: index) }
                )
            }
            addAffordance
        }
    }

    @ViewBuilder
    private var addAffordance: some View {
        HStack(spacing: 4) {
            Image(systemName: "plus")
                .font(.system(size: 11, weight: .semibold))
            TextField("add label", text: $draft)
                .focused($draftFocused)
                .font(.om.mono(12, weight: .medium))
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .frame(minWidth: 84, idealWidth: 120, maxWidth: 160)
                .submitLabel(.done)
                .onSubmit(commitDraft)
                .onChange(of: draft) { _, newValue in
                    guard newValue.contains(",") else { return }
                    commitDraft()
                }
        }
        .foregroundStyle(Color.omFgMuted)
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(Color.omBgElevated, in: Capsule(style: .continuous))
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(Color.omBorderStrong, style: StrokeStyle(lineWidth: OM.Stroke.hairline, dash: [3, 3]))
        )
    }

    private func commitDraft() {
        let cleaned = draft
            .split(whereSeparator: { $0 == "," || $0 == "\n" })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }
        guard !cleaned.isEmpty else { draft = ""; return }
        var next = labels
        for candidate in cleaned where !next.contains(candidate) && next.count < maxLabels {
            next.append(candidate)
        }
        labels = next
        draft = ""
        onChanged?()
        HapticsCenter.selection()
    }

    private func removeLabel(at index: Int) {
        guard labels.indices.contains(index) else { return }
        labels.remove(at: index)
        onChanged?()
        HapticsCenter.impact(.soft)
    }
}

// MARK: - Flow layout

/// Minimal flow layout so chips wrap naturally. Uses the SwiftUI `Layout`
/// protocol (iOS 16+). Deployment target is 17.0 — no back-compat needed.
public struct FlowLayout: Layout {
    public var spacing: CGFloat
    public var runSpacing: CGFloat

    public init(spacing: CGFloat = 8, runSpacing: CGFloat = 8) {
        self.spacing = spacing
        self.runSpacing = runSpacing
    }

    public func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let containerWidth = proposal.width ?? .infinity
        let rows = layoutRows(in: containerWidth, subviews: subviews)
        let height = rows.enumerated().reduce(0) { partial, pair in
            partial + pair.element.height + (pair.offset > 0 ? runSpacing : 0)
        }
        let width = min(containerWidth, rows.map(\.width).max() ?? 0)
        return CGSize(width: width, height: height)
    }

    public func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = layoutRows(in: bounds.width, subviews: subviews)
        var y = bounds.minY
        for row in rows {
            var x = bounds.minX
            for item in row.items {
                subviews[item.index].place(
                    at: CGPoint(x: x, y: y),
                    proposal: ProposedViewSize(width: item.size.width, height: item.size.height)
                )
                x += item.size.width + spacing
            }
            y += row.height + runSpacing
        }
    }

    private struct Row {
        var items: [RowItem]
        var width: CGFloat
        var height: CGFloat
    }
    private struct RowItem {
        let index: Int
        let size: CGSize
    }

    private func layoutRows(in maxWidth: CGFloat, subviews: Subviews) -> [Row] {
        var rows: [Row] = []
        var current: [RowItem] = []
        var currentWidth: CGFloat = 0
        var currentHeight: CGFloat = 0

        for (index, subview) in subviews.enumerated() {
            let size = subview.sizeThatFits(.unspecified)
            let projected = currentWidth + size.width + (current.isEmpty ? 0 : spacing)
            if projected > maxWidth && !current.isEmpty {
                rows.append(Row(items: current, width: currentWidth, height: currentHeight))
                current = []
                currentWidth = 0
                currentHeight = 0
            }
            if !current.isEmpty { currentWidth += spacing }
            current.append(RowItem(index: index, size: size))
            currentWidth += size.width
            currentHeight = max(currentHeight, size.height)
        }
        if !current.isEmpty {
            rows.append(Row(items: current, width: currentWidth, height: currentHeight))
        }
        return rows
    }
}
