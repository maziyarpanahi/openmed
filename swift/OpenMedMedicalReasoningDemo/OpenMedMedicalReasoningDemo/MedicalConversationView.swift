import SwiftUI

struct MedicalConversationView: View {
    private static let bottomAnchor = "medical-reasoning-chat-bottom"

    @ObservedObject var store: MedicalReasoningStore
    @FocusState private var composerFocused: Bool

    private let suggestions = [
        "Summarize the clinical timeline.",
        "What follow-up is documented?",
        "What evidence is missing?",
        "Separate documented facts from inference.",
    ]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 24) {
                        if store.messages.isEmpty {
                            emptyState
                        } else {
                            ForEach(store.messages) { message in
                                MedicalConversationRow(message: message)
                                    .id(message.id)
                            }
                        }

                        Color.clear
                            .frame(height: 1)
                            .id(Self.bottomAnchor)
                    }
                    .padding(.horizontal, MedicalReasoningTheme.pagePadding)
                    .padding(.top, 22)
                    .padding(.bottom, 16)
                }
                .scrollDismissesKeyboard(.interactively)
                .safeAreaInset(edge: .bottom, spacing: 0) { composer }
                .onAppear {
                    guard !store.messages.isEmpty else { return }
                    proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
                }
                .onChange(of: store.messages.count) { _, _ in
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
                    }
                }
                .onChange(of: latestContentLength) { _, _ in
                    proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
                }
            }
        }
    }

    private var header: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                MedicalReasoningBrand(compact: true)
                Spacer()

                Menu {
                    Button("Edit clinical context", action: store.editClinicalContext)
                    Button("Start a new case", action: store.startNewCase)
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(MedicalReasoningTheme.ink)
                        .frame(width: 40, height: 40)
                        .background(MedicalReasoningTheme.surfaceStrong, in: Circle())
                }
                .disabled(store.isGenerating)
                .accessibilityLabel("Conversation options")
            }

            HStack(spacing: 8) {
                MedicalStatusPill(
                    title: store.isLoadingModel ? "Loading LFM2.5" : "LFM2.5 · Local",
                    systemImage: store.isLoadingModel ? "hourglass" : "checkmark.circle.fill"
                )
                MedicalStatusPill(title: "Clinical context", systemImage: "doc.text.fill")
                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, MedicalReasoningTheme.pagePadding)
        .padding(.vertical, 12)
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 18) {
            ZStack {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(MedicalReasoningTheme.tealSoft)
                    .frame(width: 68, height: 68)
                Image(systemName: "text.bubble.fill")
                    .font(.system(size: 27, weight: .semibold))
                    .foregroundStyle(MedicalReasoningTheme.teal)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Ask about the supplied clinical evidence.")
                    .font(.system(size: 29, weight: .bold, design: .rounded))
                    .foregroundStyle(MedicalReasoningTheme.ink)
                Text("Follow-ups, corrections, acknowledgements, and ordinary multi-turn conversation are preserved. The model should say when the context does not contain an answer.")
                    .font(.system(size: 15, design: .rounded))
                    .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Label("On device · no cloud fallback", systemImage: "lock.fill")
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(MedicalReasoningTheme.teal)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 26)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 11) {
            if store.messages.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(suggestions, id: \.self) { suggestion in
                            Button {
                                store.draft = suggestion
                                submit()
                            } label: {
                                Text(suggestion)
                                    .font(.system(size: 12, weight: .medium, design: .rounded))
                                    .foregroundStyle(MedicalReasoningTheme.ink)
                                    .lineLimit(2)
                                    .multilineTextAlignment(.leading)
                                    .padding(.horizontal, 13)
                                    .padding(.vertical, 10)
                                    .background(
                                        MedicalReasoningTheme.surfaceStrong,
                                        in: RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    )
                                    .overlay {
                                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                                            .strokeBorder(MedicalReasoningTheme.border, lineWidth: 1)
                                    }
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .fixedSize(horizontal: false, vertical: true)
            }

            HStack(alignment: .bottom, spacing: 9) {
                TextField("Ask about the clinical context", text: $store.draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .foregroundStyle(MedicalReasoningTheme.ink)
                    .accessibilityIdentifier("chat.composer")
                    .font(.system(size: 15, design: .rounded))
                    .lineLimit(1...5)
                    .focused($composerFocused)
                    .padding(.horizontal, 15)
                    .padding(.vertical, 12)
                    .background(
                        MedicalReasoningTheme.surfaceStrong,
                        in: RoundedRectangle(cornerRadius: 20, style: .continuous)
                    )
                    .overlay {
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .strokeBorder(
                                composerFocused
                                    ? MedicalReasoningTheme.teal
                                    : MedicalReasoningTheme.border,
                                lineWidth: composerFocused ? 2 : 1
                            )
                    }
                    .submitLabel(.send)
                    .onSubmit(submit)

                Button(action: store.isGenerating ? store.stopGenerating : submit) {
                    Group {
                        if store.isGenerating {
                            Image(systemName: "stop.fill")
                                .font(.system(size: 14, weight: .bold))
                        } else {
                            Image(systemName: "arrow.up")
                                .font(.system(size: 15, weight: .bold))
                        }
                    }
                    .foregroundStyle(.white)
                    .frame(width: 45, height: 45)
                    .background(canSend || store.isGenerating ? MedicalReasoningTheme.ink : Color.gray, in: Circle())
                }
                .buttonStyle(.plain)
                .disabled(!canSend && !store.isGenerating)
                .accessibilityLabel(store.isGenerating ? "Stop response" : "Send")
                .accessibilityIdentifier("chat.sendOrStop")
            }

            Text("Model-generated reasoning and answers can be wrong. Verify clinical details against the source.")
                .font(.system(size: 10, design: .rounded))
                .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                .frame(maxWidth: .infinity, alignment: .center)
        }
        .padding(.horizontal, MedicalReasoningTheme.pagePadding)
        .padding(.top, 12)
        .padding(.bottom, 9)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) { Divider() }
    }

    private var canSend: Bool {
        !store.isGenerating
            && !store.isReleasingModel
            && !store.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var latestContentLength: Int {
        guard let latest = store.messages.last else { return 0 }
        return latest.content.count + latest.reasoning.count
    }

    private func submit() {
        guard canSend else { return }
        composerFocused = false
        Task { await store.sendMessage() }
    }
}

#Preview("Conversation") {
    MedicalConversationView(store: .previewConversation)
        .background(MedicalReasoningTheme.canvas)
}
