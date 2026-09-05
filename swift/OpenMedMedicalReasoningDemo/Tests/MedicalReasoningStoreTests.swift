import OpenMedKit
import XCTest

@testable import OpenMedMedicalReasoningMac

@MainActor
final class MedicalReasoningStoreTests: XCTestCase {
    func testChatMarkdownPreservesParagraphsAndLists() {
        let result = MedicalConversationRow.renderedMarkdown("Follow-up:\n\n- **48 hours**\n- Cardiology if symptoms recur.")
        XCTAssertEqual(String(result.characters), "Follow-up:\n\n- 48 hours\n- Cardiology if symptoms recur.")
    }

    func testHistoryKeepsCompletePairsAndExcludesStoppedAndFailedTurns() {
        let history = MedicalReasoningStore.completedHistory(from: [
            .init(role: .user, content: "first"),
            .init(role: .assistant, content: "answer"),
            .init(role: .user, content: "failed question"),
            .init(role: .assistant, content: "partial", activity: .failed),
            .init(role: .user, content: "stopped question"),
            .init(role: .assistant, content: "partial", activity: .stopped),
        ])
        XCTAssertEqual(history.map(\.content), ["first", "answer"])
        XCTAssertEqual(history.map(\.role), [.user, .assistant])
    }

    func testHistoryLimitNeverSplitsAPair() {
        let messages = (0..<12).flatMap { index in
            [
                MedicalConversationMessage(role: .user, content: "q\(index)"),
                MedicalConversationMessage(role: .assistant, content: "a\(index)"),
            ]
        }
        let history = MedicalReasoningStore.completedHistory(from: messages)
        XCTAssertEqual(history.count, 16)
        XCTAssertEqual(history.first?.content, "q4")
        XCTAssertEqual(history.last?.content, "a11")
    }

    func testEmptyDraftDoesNotStartGeneration() async {
        let store = MedicalReasoningStore(downloadState: .ready)
        store.draft = "  \n"
        await store.sendMessage()
        XCTAssertFalse(store.isGenerating)
        XCTAssertTrue(store.messages.isEmpty)
    }

    /// Opt-in, real model test: validates the same downloader, store and runtime
    /// used by the SwiftUI app. Fixtures are synthetic; no user case is logged.
    func testLocalArtifactConversationStopAndRestart() async throws {
        guard ProcessInfo.processInfo.environment["OPENMED_LFM_MODEL_DIRECTORY"] != nil else {
            throw XCTSkip("Set OPENMED_LFM_MODEL_DIRECTORY to the pinned local artifact.")
        }
        let directory = try LFMModelDownloader.cachedDirectory()
        XCTAssertTrue(LFMModelDownloader.isCachedArtifactValid(at: directory))
        let prepared = try await LFMModelDownloader.shared.prepare { _, _, _ in }
        XCTAssertEqual(prepared, directory)
        let store = MedicalReasoningStore()
        XCTAssertTrue(store.modelIsReady)
        store.startConversation()
        store.draft = "What follow-up is documented? Answer in one sentence."
        await store.sendMessage()
        XCTAssertNil(store.errorMessage)
        XCTAssertEqual(store.messages.last?.activity, .complete)
        XCTAssertTrue(store.messages.last?.content.contains("48") == true)
        XCTAssertFalse(store.messages.last?.reasoning.isEmpty ?? true)
        let firstAnswer = store.messages.last?.content ?? ""
        print("SYNTHETIC first answer: \(firstAnswer)")

        store.draft = "thanks"
        await store.sendMessage()
        XCTAssertNil(store.errorMessage)
        XCTAssertEqual(store.messages.count, 4)
        XCTAssertEqual(store.messages.last?.activity, .complete)
        let thanks = store.messages.last?.content ?? ""
        XCTAssertFalse(thanks.isEmpty)
        XCTAssertLessThan(thanks.count, 500, "An acknowledgement should not restart the clinical analysis.")
        print("SYNTHETIC acknowledgement: \(thanks)")

        store.draft = "Explain every documented finding and each evidence gap in detail."
        let pending = Task { await store.sendMessage() }
        // Poll a visible state with a deadline, not a guessed generation delay.
        let deadline = Date().addingTimeInterval(60)
        while store.messages.count < 6 || store.messages.last?.reasoning.isEmpty == true,
            Date() < deadline
        {
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertTrue(store.isGenerating)
        store.stopGenerating()
        await pending.value
        XCTAssertEqual(store.messages.last?.activity, .stopped)
        XCTAssertNil(store.errorMessage)
        XCTAssertFalse(store.isGenerating)

        await store.releaseRuntime()
        store.draft = "What was the follow-up interval again? Answer in one sentence."
        await store.sendMessage()
        XCTAssertNil(store.errorMessage)
        XCTAssertEqual(store.messages.last?.activity, .complete)
        XCTAssertTrue(store.messages.last?.content.contains("48") == true)
        await store.releaseRuntime()
    }
}
