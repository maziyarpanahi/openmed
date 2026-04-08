import Foundation
import XCTest
@testable import OpenMedKit

final class RemoteServiceTests: XCTestCase {

    func testHealthDecodesServiceMetadata() async throws {
        let session = makeSession(statusCode: 200, body: """
        {
          "status": "ok",
          "service": "openmed-rest",
          "version": "1.0.0",
          "profile": "prod"
        }
        """)
        let client = OpenMedServiceClient(
            baseURL: URL(string: "http://127.0.0.1:8080/")!,
            session: session
        )

        let health = try await client.health()

        XCTAssertEqual(health.status, "ok")
        XCTAssertEqual(health.service, "openmed-rest")
        XCTAssertEqual(health.profile, "prod")
    }

    func testExtractPIIDecodesEntities() async throws {
        let session = makeSession(statusCode: 200, body: """
        {
          "text": "Patient John Doe",
          "model_name": "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
          "entities": [
            {
              "text": "John",
              "label": "first_name",
              "confidence": 0.97,
              "start": 8,
              "end": 12
            },
            {
              "text": "Doe",
              "label": "last_name",
              "confidence": 0.96,
              "start": 13,
              "end": 16
            }
          ]
        }
        """)
        let client = OpenMedServiceClient(
            baseURL: URL(string: "http://127.0.0.1:8080/")!,
            session: session
        )

        let entities = try await client.extractPII(
            "Patient John Doe",
            modelName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1"
        )

        XCTAssertEqual(entities.count, 2)
        XCTAssertEqual(entities[0].label, "first_name")
        XCTAssertEqual(entities[1].text, "Doe")
    }

    func testServerEnvelopeBecomesReadableError() async {
        let session = makeSession(statusCode: 503, body: """
        {
          "error": {
            "code": "internal_error",
            "message": "Backend not available"
          }
        }
        """)
        let client = OpenMedServiceClient(
            baseURL: URL(string: "http://127.0.0.1:8080/")!,
            session: session
        )

        do {
            _ = try await client.health()
            XCTFail("Expected server error")
        } catch let error as OpenMedServiceClientError {
            XCTAssertEqual(
                error,
                .server(statusCode: 503, code: "internal_error", message: "Backend not available")
            )
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    private func makeSession(statusCode: Int, body: String) -> URLSession {
        MockURLProtocol.handler = { request in
            let response = HTTPURLResponse(
                url: try XCTUnwrap(request.url),
                statusCode: statusCode,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, Data(body.utf8))
        }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return URLSession(configuration: configuration)
    }
}

private final class MockURLProtocol: URLProtocol {
    static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }

        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
