import SwiftUI

struct ContentView: View {
    @ObservedObject var store: MedicalReasoningStore

    var body: some View {
        ZStack {
            MedicalReasoningTheme.canvas.ignoresSafeArea()

            switch store.route {
            case .modelSetup:
                ModelSetupView(store: store)
                    .transition(.opacity)
            case .clinicalContext:
                ClinicalContextView(store: store)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            case .conversation:
                MedicalConversationView(store: store)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.22), value: store.route)
        .alert(
            "Something went wrong",
            isPresented: Binding(
                get: { store.errorMessage != nil },
                set: { if !$0 { store.errorMessage = nil } }
            )
        ) {
            Button("OK", role: .cancel) { store.errorMessage = nil }
        } message: {
            if let errorMessage = store.errorMessage {
                Text(errorMessage)
            }
        }
    }
}

#Preview("Model setup") {
    ContentView(store: .preview(route: .modelSetup, downloadState: .missing))
}

#Preview("Clinical conversation") {
    ContentView(store: .previewConversation)
}
