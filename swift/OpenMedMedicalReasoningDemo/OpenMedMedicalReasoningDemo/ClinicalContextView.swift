import SwiftUI

struct ClinicalContextView: View {
    @ObservedObject var store: MedicalReasoningStore
    @FocusState private var editorFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("CLINICAL CONTEXT")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .tracking(2.2)
                            .foregroundStyle(MedicalReasoningTheme.teal)
                        Text("Ground the conversation in a case.")
                            .font(.system(size: 34, weight: .bold, design: .rounded))
                            .foregroundStyle(MedicalReasoningTheme.ink)
                        Text("LFM2.5 receives this context with every turn. Keep the synthetic example or replace it with text that is already de-identified.")
                            .font(.system(size: 16, design: .rounded))
                            .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                            .lineSpacing(3)
                    }

                    safetyNotice
                    editor
                }
                .padding(.horizontal, MedicalReasoningTheme.pagePadding)
                .padding(.vertical, 22)
                .padding(.bottom, 110)
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) { continueButton }
    }

    private var header: some View {
        HStack {
            Button(action: store.backToModelSetup) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(MedicalReasoningTheme.ink)
                    .frame(width: 40, height: 40)
                    .background(MedicalReasoningTheme.surfaceStrong, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Back to model setup")

            Spacer()
            MedicalReasoningBrand(compact: true)
            Spacer()
            MedicalStatusPill(title: "De-ID only", systemImage: "lock.fill")
        }
        .padding(.horizontal, MedicalReasoningTheme.pagePadding)
        .padding(.vertical, 12)
    }

    private var safetyNotice: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.shield.fill")
                .foregroundStyle(MedicalReasoningTheme.coral)
            Text("This example does not run PII detection. Do not paste names, record numbers, contact details, or other direct identifiers.")
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                .fixedSize(horizontal: false, vertical: true)
        }
        .medicalCard()
    }

    private var editor: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("DE-IDENTIFIED CASE")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .tracking(1.4)
                    .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                Spacer()
                Text("\(store.clinicalContext.count) characters")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(MedicalReasoningTheme.secondaryInk)
            }

            TextEditor(text: $store.clinicalContext)
                .font(.system(size: 15, design: .rounded))
                .foregroundStyle(MedicalReasoningTheme.ink)
                .scrollContentBackground(.hidden)
                .focused($editorFocused)
                .frame(minHeight: 360)
                .padding(12)
                .background(
                    MedicalReasoningTheme.surfaceStrong,
                    in: RoundedRectangle(cornerRadius: 15, style: .continuous)
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 15, style: .continuous)
                        .strokeBorder(
                            editorFocused
                                ? MedicalReasoningTheme.teal
                                : MedicalReasoningTheme.border,
                            lineWidth: editorFocused ? 2 : 1
                        )
                }

            Button("Restore synthetic case", action: store.restoreSyntheticCase)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(MedicalReasoningTheme.teal)
        }
        .medicalCard()
    }

    private var continueButton: some View {
        Button(action: store.startConversation) {
            HStack {
                Text("Start clinical conversation")
                Spacer()
                Image(systemName: "arrow.right")
            }
            .font(.system(size: 16, weight: .semibold, design: .rounded))
            .foregroundStyle(.white)
            .padding(.horizontal, 20)
            .frame(height: 54)
            .background(
                store.contextIsUsable
                    ? MedicalReasoningTheme.ink
                    : Color.gray,
                in: Capsule()
            )
        }
        .buttonStyle(.plain)
        .disabled(!store.contextIsUsable)
        .padding(.horizontal, MedicalReasoningTheme.pagePadding)
        .padding(.top, 14)
        .padding(.bottom, 10)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) { Divider() }
    }
}

#Preview("Synthetic context") {
    ClinicalContextView(
        store: .preview(route: .clinicalContext, downloadState: .ready)
    )
    .background(MedicalReasoningTheme.canvas)
}
