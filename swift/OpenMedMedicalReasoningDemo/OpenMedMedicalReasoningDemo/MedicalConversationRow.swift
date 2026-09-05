import SwiftUI

struct MedicalConversationRow: View {
    let message: MedicalConversationMessage

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer(minLength: 52)
                Text(message.content)
                    .font(.system(size: 15, design: .rounded))
                    .foregroundStyle(.white)
                    .textSelection(.enabled)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        MedicalReasoningTheme.ink,
                        in: RoundedRectangle(cornerRadius: 18, style: .continuous)
                    )
            }

        case .assistant:
            assistantMessage
        }
    }

    private var assistantMessage: some View {
        VStack(alignment: .leading, spacing: 13) {
            HStack(spacing: 9) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(MedicalReasoningTheme.teal)
                        .frame(width: 30, height: 30)
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.white)
                }
                Text("LFM2.5")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(MedicalReasoningTheme.ink)
                if message.isGenerating {
                    MedicalStatusPill(
                        title: message.activity == .reasoning ? "Reasoning" : "Answering",
                        systemImage: message.activity == .reasoning ? "brain" : "text.cursor"
                    )
                }
            }

            if !message.reasoning.isEmpty || message.activity == .reasoning {
                MedicalReasoningDisclosure(message: message)
            }

            if !message.content.isEmpty {
                Text(Self.renderedMarkdown(message.content))
                    .font(.system(size: 15, design: .rounded))
                    .foregroundStyle(MedicalReasoningTheme.ink)
                    .lineSpacing(4)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } else if message.activity == .answering {
                Label("Writing response…", systemImage: "ellipsis")
                    .font(.system(size: 13, design: .rounded))
                    .foregroundStyle(MedicalReasoningTheme.secondaryInk)
            }

            if message.activity == .failed {
                Label(
                    "Response failed. Send your question again to retry; this unfinished turn is excluded from model history.",
                    systemImage: "exclamationmark.circle"
                )
                .font(.system(size: 13, design: .rounded))
                .foregroundStyle(MedicalReasoningTheme.coral)
            }
            if message.activity == .stopped {
                Label("Response stopped. You can send another message.", systemImage: "stop.circle")
                    .font(.system(size: 13, design: .rounded))
                    .foregroundStyle(MedicalReasoningTheme.secondaryInk)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    static func renderedMarkdown(_ text: String) -> AttributedString {
        // SwiftUI Text doesn't lay out AttributedString block presentation
        // intents. Preserve newlines so lists/paragraphs never run together.
        (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
    }
}

private struct MedicalReasoningDisclosure: View {
    let message: MedicalConversationMessage
    @State private var isExpanded: Bool

    init(message: MedicalConversationMessage) {
        self.message = message
        _isExpanded = State(initialValue: message.activity == .reasoning)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 9) {
                    if message.activity == .reasoning {
                        ProgressView()
                            .controlSize(.mini)
                            .tint(MedicalReasoningTheme.teal)
                    } else {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(MedicalReasoningTheme.teal)
                    }
                    Text(message.activity == .reasoning ? "Model reasoning…" : "Model reasoning")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(MedicalReasoningTheme.ink)
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                }
                .contentShape(Rectangle())
                .padding(.horizontal, 13)
                .padding(.vertical, 11)
            }
            .buttonStyle(.plain)

            if isExpanded {
                Divider()
                Group {
                    if message.reasoning.isEmpty {
                        Text("Starting the on-device reasoning trace…")
                            .italic()
                    } else {
                        Text(message.reasoning)
                            .textSelection(.enabled)
                    }
                }
                .font(.system(size: 13, design: .rounded))
                .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                .lineSpacing(3)
                .padding(13)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            MedicalReasoningTheme.surface,
            in: RoundedRectangle(cornerRadius: 15, style: .continuous)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 15, style: .continuous)
                .strokeBorder(MedicalReasoningTheme.border, lineWidth: 1)
        }
        .onChange(of: message.activity) { oldValue, newValue in
            guard oldValue != newValue, newValue == .complete || newValue == .stopped || newValue == .failed else { return }
            withAnimation(.easeInOut(duration: 0.2)) {
                isExpanded = false
            }
        }
    }
}
