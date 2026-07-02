import Foundation

enum SafeLog {
    enum Level: String, Sendable {
        case debug
        case info
        case warning
        case error
    }

    enum Operation: String, Sendable {
        case analyzeText
        case extractPII
        case extractPIIChunked
    }

    enum Event: Sendable {
        case inferenceStarted(operation: Operation)
        case inferenceCompleted(operation: Operation, entityCount: Int)
        case inferenceFailed(operation: Operation, errorType: String)
    }

    struct SpanSummary: Equatable, Sendable {
        let label: String
        let start: Int
        let end: Int
        let confidence: Float
        let textHash: String
    }

    typealias Sink = (
        _ level: Level,
        _ event: String,
        _ metadata: [String: String]
    ) -> Void

    private static let lock = NSLock()
    private static var sink: Sink?

    static func withSink<T>(_ newSink: @escaping Sink, _ body: () throws -> T) rethrows -> T {
        lock.lock()
        let previousSink = sink
        sink = newSink
        lock.unlock()

        defer {
            lock.lock()
            sink = previousSink
            lock.unlock()
        }

        return try body()
    }

    static func log(_ event: Event, level: Level = .debug) {
        let (eventName, metadata) = event.payload
        let currentSink: Sink?

        lock.lock()
        currentSink = sink
        lock.unlock()

        currentSink?(level, eventName, metadata)
    }

    static func summarize(_ entity: EntityPrediction) -> SpanSummary {
        SpanSummary(
            label: entity.label,
            start: entity.start,
            end: entity.end,
            confidence: entity.confidence,
            textHash: entity.textHash
        )
    }
}

extension SafeLog.Event {
    fileprivate var payload: (String, [String: String]) {
        switch self {
        case .inferenceStarted(let operation):
            return (
                "openmed.inference.started",
                ["operation": operation.rawValue]
            )
        case .inferenceCompleted(let operation, let entityCount):
            return (
                "openmed.inference.completed",
                [
                    "operation": operation.rawValue,
                    "entity_count": String(entityCount),
                ]
            )
        case .inferenceFailed(let operation, let errorType):
            return (
                "openmed.inference.failed",
                [
                    "operation": operation.rawValue,
                    "error_type": errorType,
                ]
            )
        }
    }
}
