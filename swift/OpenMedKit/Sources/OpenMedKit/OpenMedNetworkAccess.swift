import Foundation

enum OpenMedNetworkOperation: String, Sendable {
    case modelArtifactDownload
    case tokenizerAssetDownload
}

enum OpenMedNetworkAccess {
    typealias Observer = (_ operation: OpenMedNetworkOperation, _ url: URL) -> Void

    private static let lock = NSLock()
    private static var observer: Observer?

    static func withObserver<T>(
        _ newObserver: @escaping Observer,
        _ body: () throws -> T
    ) rethrows -> T {
        lock.lock()
        let previousObserver = observer
        observer = newObserver
        lock.unlock()

        defer {
            lock.lock()
            observer = previousObserver
            lock.unlock()
        }

        return try body()
    }

    static func data(
        from url: URL,
        operation: OpenMedNetworkOperation
    ) async throws -> (Data, URLResponse) {
        notify(operation: operation, url: url)
        return try await URLSession.shared.data(from: url)
    }

    static func data(
        for request: URLRequest,
        operation: OpenMedNetworkOperation
    ) async throws -> (Data, URLResponse) {
        if let url = request.url {
            notify(operation: operation, url: url)
        }
        return try await URLSession.shared.data(for: request)
    }

    private static func notify(operation: OpenMedNetworkOperation, url: URL) {
        let currentObserver: Observer?

        lock.lock()
        currentObserver = observer
        lock.unlock()

        currentObserver?(operation, url)
    }
}
