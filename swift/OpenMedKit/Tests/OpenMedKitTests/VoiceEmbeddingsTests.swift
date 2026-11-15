import XCTest

@testable import OpenMedKit

final class VoiceEmbeddingsTests: XCTestCase {
    private func refused<T>(_ code: VoiceEmbeddingError, _ action: () throws -> T) {
        XCTAssertThrowsError(try action()) { error in
            XCTAssertEqual(error as? VoiceEmbeddingError, code)
        }
    }

    func testLocalSimilarityAndOpaqueEvidence() throws {
        let owner = try VoiceEmbeddingSession()
        let a = try owner.add([1, 0])
        let b = try owner.add([0, 1])
        XCTAssertEqual(try a.similarity(to: b).score, 0)
        XCTAssertEqual(try a.similarity(to: a).score, 1)
        let large = try owner.add([1e308, 1e308])
        XCTAssertEqual(try large.similarity(to: large).score, 1, accuracy: 1e-12)
        XCTAssertTrue(try a.similarity(to: b).reviewerConfirmationRequired)
        XCTAssertEqual(try a.similarity(to: b).notice, "non_diagnostic_voice_similarity")
        XCTAssertEqual(a.evidence().keys.sorted(), ["handle_digest"])
        XCTAssertEqual(a.handleDigest.count, 64)
        XCTAssertNotEqual(a.handleDigest, try owner.add([1, 0]).handleDigest)
        XCTAssertEqual(Mirror(reflecting: a).children.count, 1)
        XCTAssertEqual(Mirror(reflecting: owner).children.count, 0)
        refused(.serializationRefused) { try a.serialize() }
        refused(.persistenceRefused) { try a.persist() }
        refused(.dimensionsMismatch) { try a.similarity(to: owner.add([1])) }
    }

    func testEveryBoundaryErasesRetainedAllocationsAndInvalidatesHandles() throws {
        for boundary in [VoiceEmbeddingBoundary.pause, .withdrawal, .cancellation, .finalization] {
            var allocations: [VoiceEmbeddingBuffer] = []
            let owner = try VoiceEmbeddingSession(allocator: { count in
                let buffer = try VoiceEmbeddingBuffer(count: count)
                allocations.append(buffer)
                return buffer
            })
            let handle = try owner.add([827.125, -934.375])
            let receipt = owner.destroy(boundary)
            XCTAssertEqual(receipt.reasonCode, boundary.rawValue)
            XCTAssertEqual(receipt.destroyedCount, 1)
            XCTAssertEqual(receipt.handleDigests, [handle.handleDigest])
            XCTAssertTrue(allocations.allSatisfy { $0.count == 0 })
            XCTAssertEqual(owner.diagnostics().handleCount, 0)
            refused(.destroyed) { try handle.similarity(to: handle) }
            refused(.sessionClosed) { try owner.add([1]) }
            XCTAssertEqual(owner.destroy(boundary).destroyedCount, 0)
            let evidence = String(decoding: try JSONEncoder().encode(receipt), as: UTF8.self)
            XCTAssertFalse((evidence + handle.description + owner.description).contains("827.125"))
            XCTAssertFalse((evidence + handle.description + owner.description).contains("934.375"))
        }
    }

    func testUnreachableAllocationsAndWeakOwnerWithRetainedCallbacks() throws {
        weak var buffer: VoiceEmbeddingBuffer?
        weak var weakOwner: VoiceEmbeddingSession?
        var callbacks: [() -> EmbeddingDestructionReceipt] = []
        var owner: VoiceEmbeddingSession? = try VoiceEmbeddingSession(
            allocator: { count in
                let result = try VoiceEmbeddingBuffer(count: count)
                buffer = result
                return result
            }, register: { _, erase in callbacks.append(erase) })
        weakOwner = owner
        let handle = try XCTUnwrap(owner).add([1, 2])
        owner = nil
        XCTAssertNil(weakOwner)
        XCTAssertNil(buffer)
        XCTAssertEqual(callbacks[0]().destroyedCount, 0)
        refused(.destroyed) { try handle.similarity(to: handle) }
    }

    func testCrossSessionRefusalSurvivesDestruction() throws {
        var left: VoiceEmbeddingSession? = try VoiceEmbeddingSession()
        var right: VoiceEmbeddingSession? = try VoiceEmbeddingSession()
        let a = try XCTUnwrap(left).add([1])
        let b = try XCTUnwrap(right).add([1])
        refused(.crossSessionRefused) { try a.similarity(to: b) }
        left = nil
        right = nil
        refused(.crossSessionRefused) { try a.similarity(to: b) }
    }

    func testRetentionHookAndFailedRegistration() throws {
        var callbacks: [() -> EmbeddingDestructionReceipt] = []
        let owner = try VoiceEmbeddingSession(register: { _, erase in callbacks.append(erase) })
        let handles = [try owner.add([1, 0]), try owner.add([0, 1])]
        XCTAssertEqual(callbacks.map { $0().destroyedCount }.reduce(0, +), 2)
        for handle in handles { refused(.destroyed) { try handle.similarity(to: handle) } }
        XCTAssertTrue(callbacks.allSatisfy { $0().destroyedCount == 0 })
        var retained: VoiceEmbeddingBuffer?
        let failing = try VoiceEmbeddingSession(
            allocator: { count in
                let buffer = try VoiceEmbeddingBuffer(count: count)
                retained = buffer
                return buffer
            }, register: { _, _ in throw NSError(domain: "synthetic-private-payload", code: 1) })
        refused(.registrationFailed) { try failing.add([123.5]) }
        XCTAssertEqual(retained?.count, 0)
        XCTAssertEqual(failing.diagnostics().handleCount, 0)
        let reentrant = try VoiceEmbeddingSession(register: { _, erase in _ = erase() })
        refused(.destroyed) { try reentrant.add([1]) }
    }

    func testNegativeVectorsBoundsAndAllocatorReuse() throws {
        let owner = try VoiceEmbeddingSession(maxHandles: 1, maxDimensions: 2)
        for values: [Double] in [[], [0, 0], [.nan], [.infinity], [1, 2, 3]] {
            refused(.vectorInvalid) { try owner.add(values) }
        }
        refused(.limitsInvalid) { try VoiceEmbeddingSession(maxHandles: 0) }
        _ = try owner.add([1])
        refused(.capacityExceeded) { try owner.add([1]) }
        let allocation = try VoiceEmbeddingBuffer(count: 1)
        let reused = try VoiceEmbeddingSession(allocator: { _ in allocation })
        let handle = try reused.add([1])
        refused(.allocatorFailed) { try reused.add([2]) }
        XCTAssertEqual(try handle.similarity(to: handle).score, 1)
        let wrongSize = try VoiceEmbeddingSession(allocator: { _ in try VoiceEmbeddingBuffer(count: 2) })
        refused(.allocatorFailed) { try wrongSize.add([1]) }
    }
}
