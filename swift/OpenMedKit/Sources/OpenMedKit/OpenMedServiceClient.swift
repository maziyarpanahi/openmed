import Foundation

/// Health metadata returned by the OpenMed REST service.
public struct OpenMedServiceHealth: Codable, Equatable {
    public let status: String
    public let service: String
    public let version: String
    public let profile: String

    public init(status: String, service: String, version: String, profile: String) {
        self.status = status
        self.service = service
        self.version = version
        self.profile = profile
    }
}

/// Lightweight Swift client for the OpenMed REST service.
public final class OpenMedServiceClient {
    public let baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    public init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder = decoder

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        self.encoder = encoder
    }

    public func health() async throws -> OpenMedServiceHealth {
        let request = try makeRequest(path: "health", method: "GET")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(OpenMedServiceHealth.self, from: data)
    }

    public func extractPII(
        _ text: String,
        modelName: String,
        confidenceThreshold: Float = 0.5,
        lang: String = "en",
        normalizeAccents: Bool? = nil,
        useSmartMerging: Bool = true
    ) async throws -> [EntityPrediction] {
        let payload = PIIExtractRequest(
            text: text,
            modelName: modelName,
            confidenceThreshold: confidenceThreshold,
            useSmartMerging: useSmartMerging,
            lang: lang,
            normalizeAccents: normalizeAccents
        )
        var request = try makeRequest(path: "pii/extract", method: "POST")
        request.httpBody = try encoder.encode(payload)

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(PIIExtractResponse.self, from: data).entities
    }

    private func makeRequest(path: String, method: String) throws -> URLRequest {
        guard let url = URL(string: path, relativeTo: baseURL)?.absoluteURL else {
            throw OpenMedServiceClientError.invalidURL(path)
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return request
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw OpenMedServiceClientError.invalidResponse
        }

        guard (200..<300).contains(http.statusCode) else {
            if let envelope = try? decoder.decode(ServiceErrorEnvelope.self, from: data) {
                throw OpenMedServiceClientError.server(
                    statusCode: http.statusCode,
                    code: envelope.error.code,
                    message: envelope.error.message
                )
            }
            throw OpenMedServiceClientError.server(
                statusCode: http.statusCode,
                code: "http_\(http.statusCode)",
                message: String(data: data, encoding: .utf8) ?? "Request failed"
            )
        }
    }
}

public enum OpenMedServiceClientError: LocalizedError, Equatable {
    case invalidURL(String)
    case invalidResponse
    case server(statusCode: Int, code: String, message: String)

    public var errorDescription: String? {
        switch self {
        case .invalidURL(let path):
            return "Invalid OpenMed service URL for path \(path)."
        case .invalidResponse:
            return "OpenMed service returned an invalid response."
        case .server(_, _, let message):
            return message
        }
    }
}

private struct PIIExtractRequest: Encodable {
    let text: String
    let modelName: String
    let confidenceThreshold: Float
    let useSmartMerging: Bool
    let lang: String
    let normalizeAccents: Bool?
}

private struct PIIExtractResponse: Decodable {
    let entities: [EntityPrediction]
}

private struct ServiceErrorEnvelope: Decodable {
    let error: ServiceErrorPayload
}

private struct ServiceErrorPayload: Decodable {
    let code: String
    let message: String
}
