import Foundation
import XCTest

@testable import OpenMedKit

#if canImport(CoreGraphics) && canImport(ImageIO)
    import CoreGraphics
    import ImageIO
#endif
#if canImport(PDFKit)
    import PDFKit
#endif

private let syntheticDigest = String(repeating: "e", count: 64)

private final class FakeImage: DocumentIntakeImage, @unchecked Sendable {
    let pixelWidth: Int
    let pixelHeight: Int
    private let lock = NSLock()
    private var releases = 0

    init(width: Int, height: Int) {
        pixelWidth = width
        pixelHeight = height
    }

    var releaseCount: Int { lock.withLock { releases } }
    func release() { lock.withLock { releases += 1 } }
}

private enum FakePage {
    case page(DocumentPageGeometry, FakeImage)
    case fails(DocumentIntakeReason)
}

private final class FakeDocument: DocumentIntakeDocument {
    let sourceDigest = syntheticDigest
    let pages: [FakePage]
    var rendered: [Int] = []
    var closes = 0

    init(_ pages: [FakePage]) { self.pages = pages }

    var pageCount: Int { pages.count }

    func renderPage(at index: Int, limits: DocumentIntakeLimits) throws -> DocumentRenderedPage {
        rendered.append(index)
        switch pages[index] {
        case .page(let geometry, let image): return DocumentRenderedPage(geometry: geometry, image: image)
        case .fails(let reason): throw reason
        }
    }

    func close() { closes += 1 }
}

private struct FakeSource: DocumentIntakeSource {
    let document: FakeDocument?
    var failure: Error = DocumentIntakeReason.permissionDenied

    func open(limits: DocumentIntakeLimits) throws -> any DocumentIntakeDocument {
        guard let document else { throw failure }
        return document
    }
}

/// Returns one word covering the whole OCR image, or a configured behavior.
private struct FakeRecognizer: DocumentTextRecognizer {
    enum Behavior: Sendable { case word, fail, wrongSize, cancelOnSecondPage }
    let behavior: Behavior
    let calls = Counter()

    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func next() -> Int {
            lock.withLock {
                defer { value += 1 }
                return value
            }
        }
        var count: Int { lock.withLock { value } }
    }

    func recognize(_ image: any DocumentIntakeImage) async throws -> DocumentRecognition {
        let call = calls.next()
        switch behavior {
        case .fail: throw NSError(domain: "synthetic-private-diagnostic", code: 7)
        case .wrongSize: return DocumentRecognition(width: image.pixelWidth + 50, height: image.pixelHeight, text: "", regions: [])
        case .cancelOnSecondPage where call == 1:
            withUnsafeCurrentTask { $0?.cancel() }
            try await Task.sleep(nanoseconds: 1_000_000_000)
        default: break
        }
        let box = DocumentIntakeBox(left: 0, top: 0, right: Double(image.pixelWidth), bottom: Double(image.pixelHeight))
        return DocumentRecognition(
            width: image.pixelWidth, height: image.pixelHeight, text: "Word",
            regions: [DocumentRecognizedRegion(scalarStart: 0, scalarEnd: 4, box: box)])
    }
}

private func page(_ width: Int = 100, _ height: Int = 50) -> (FakePage, FakeImage) {
    let image = FakeImage(width: width, height: height)
    return (.page(DocumentPageGeometry(width: Double(width), height: Double(height)), image), image)
}

final class DocumentIntakeTests: XCTestCase {
    func testDocumentRefusalsAreValueFree() async throws {
        let intake = DocumentIntake(recognizer: FakeRecognizer(behavior: .word))
        for (failure, reason) in [
            (DocumentIntakeReason.permissionDenied, DocumentIntakeReason.permissionDenied),
            (.permissionLost, .permissionDenied), (.documentLocked, .documentLocked), (.sizeLimit, .sizeLimit),
            (.pageMissing, .unsupportedDocument),
        ] {
            let report = try await intake.run(FakeSource(document: nil, failure: failure))
            XCTAssertEqual(report.outcome, .refused)
            XCTAssertEqual(report.reasons, [reason])
        }
        let report = try await intake.run(FakeSource(document: nil, failure: NSError(domain: "synthetic-path", code: 1)))
        XCTAssertEqual(report.reasons, [.unsupportedDocument])
        XCTAssertFalse(String(describing: report).contains("synthetic-path"))
        XCTAssertEqual(DocumentIntakeReason.permissionLost.localizedDescription, "permission_lost")
    }

    func testMissingPagesAndPermissionLossArePartialAndStopReading() async throws {
        let (first, firstImage) = page()
        let (third, thirdImage) = page()
        let (last, lastImage) = page()
        let document = FakeDocument([first, .fails(.pageMissing), third, .fails(.permissionLost), last])
        let report = try await DocumentIntake(recognizer: FakeRecognizer(behavior: .word)).run(FakeSource(document: document))
        XCTAssertEqual(report.outcome, .partial)
        XCTAssertEqual(report.reasons, [.pageMissing, .permissionLost])
        XCTAssertEqual(report.pages.map(\.status), [.mapped, .held, .mapped, .held])
        XCTAssertEqual(report.text, "Word\n\nWord")
        XCTAssertEqual(document.rendered, [0, 1, 2, 3])
        XCTAssertEqual(document.closes, 1)
        XCTAssertEqual([firstImage, thirdImage].map(\.releaseCount), [1, 1])
        XCTAssertEqual(lastImage.releaseCount, 0)
        XCTAssertEqual(report.regions(overlapping: 6..<10).map(\.page), [2])
    }

    func testRecognitionFailuresAndSizeMismatchesHoldPagesAndReleaseImages() async throws {
        for (behavior, reason) in [(FakeRecognizer.Behavior.fail, DocumentIntakeReason.recognitionFailed), (.wrongSize, .ambiguousTransform)] {
            let (only, image) = page()
            let document = FakeDocument([only])
            let report = try await DocumentIntake(recognizer: FakeRecognizer(behavior: behavior)).run(FakeSource(document: document))
            XCTAssertEqual(report.outcome, .refused)
            XCTAssertEqual(report.reasons, [reason])
            XCTAssertEqual(image.releaseCount, 1)
            XCTAssertEqual(document.closes, 1)
            XCTAssertFalse(String(describing: report).contains("synthetic-private-diagnostic"))
        }
    }

    func testPageLimitStopsRenderingAndIsReported() async throws {
        let pages = (0..<4).map { _ in page() }
        let document = FakeDocument(pages.map(\.0))
        let intake = DocumentIntake(recognizer: FakeRecognizer(behavior: .word), limits: DocumentIntakeLimits(maxPages: 2))
        let report = try await intake.run(FakeSource(document: document))
        XCTAssertEqual(report.outcome, .partial)
        XCTAssertEqual(report.reasons, [.pageLimit])
        XCTAssertEqual(document.rendered, [0, 1])
        XCTAssertEqual(report.pages.count, 2)
        let empty = try await intake.run(FakeSource(document: FakeDocument([])))
        XCTAssertEqual(empty.reasons, [.emptyDocument])
    }

    func testCancellationClosesTheDocumentReleasesImagesAndWritesNothing() async throws {
        let temporary = FileManager.default.temporaryDirectory
        let before = try FileManager.default.contentsOfDirectory(atPath: temporary.path).sorted()
        let pages = (0..<3).map { _ in page() }
        let document = FakeDocument(pages.map(\.0))
        let recognizer = FakeRecognizer(behavior: .cancelOnSecondPage)
        let task = Task { try await DocumentIntake(recognizer: recognizer).run(FakeSource(document: document)) }
        do {
            _ = try await task.value
            XCTFail("cancelled intake returned a report")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertEqual(document.rendered, [0, 1])
        XCTAssertEqual(document.closes, 1)
        XCTAssertEqual(pages.map(\.1.releaseCount), [1, 1, 0])
        XCTAssertEqual(recognizer.calls.count, 2)
        let after = try FileManager.default.contentsOfDirectory(atPath: temporary.path).sorted()
        XCTAssertEqual(after, before)

        let early = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await DocumentIntake(recognizer: recognizer).run(FakeSource(document: FakeDocument([page().0])))
        }
        do {
            _ = try await early.value
            XCTFail("cancelled intake opened the source")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
    }

    func testVisualFactoryReproducesPythonVisualPackets() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/multimodal_contract_v1.json"))
        let corpus = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var checked = 0
        for row in try XCTUnwrap(corpus["cases"] as? [[String: Any]]) {
            let wire = Data(try XCTUnwrap(row["wire"] as? String).utf8)
            guard case .visual(let visual) = try MultimodalPacket.fromJSON(wire).projection else { continue }
            let rebuilt = try MultimodalPacket.visual(
                nodeRef: visual.nodeRef, sourceDigest: visual.sourceDigest, kind: visual.kind, page: visual.page,
                width: visual.width, height: visual.height, unit: visual.unit, sourceBox: visual.sourceBox,
                scalarSpan: visual.scalarSpan, rotation: visual.rotation)
            XCTAssertEqual(rebuilt.toJSON(), wire)
            checked += 1
        }
        XCTAssertGreaterThan(checked, 3)
        XCTAssertThrowsError(
            try MultimodalPacket.visual(
                nodeRef: String(repeating: "1", count: 32), sourceDigest: syntheticDigest, kind: .page, page: 0, width: 10,
                height: 10, unit: .pixel, sourceBox: nil, scalarSpan: nil, rotation: 45))
    }
}

#if canImport(CoreGraphics) && canImport(ImageIO)
    /// Treats the bounding box of dark pixels as one recognized word.
    private struct DarkPixelRecognizer: DocumentTextRecognizer {
        let seen = Seen()

        final class Seen: @unchecked Sendable {
            private let lock = NSLock()
            private var images: [CGDocumentIntakeImage] = []
            func add(_ image: CGDocumentIntakeImage) { lock.withLock { images.append(image) } }
            var all: [CGDocumentIntakeImage] { lock.withLock { images } }
        }

        func recognize(_ image: any DocumentIntakeImage) async throws -> DocumentRecognition {
            let source = try XCTUnwrap(image as? CGDocumentIntakeImage)
            seen.add(source)
            let cgImage = try XCTUnwrap(source.cgImage)
            let width = cgImage.width
            let height = cgImage.height
            var pixels = [UInt8](repeating: 0, count: width * height * 4)
            pixels.withUnsafeMutableBytes { buffer in
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
                context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            }
            var box: (Int, Int, Int, Int)?
            for y in 0..<height {
                for x in 0..<width where pixels[(y * width + x) * 4] < 128 {
                    let current = box ?? (x, y, x, y)
                    box = (min(current.0, x), min(current.1, y), max(current.2, x), max(current.3, y))
                }
            }
            guard let box else { return DocumentRecognition(width: width, height: height, text: "", regions: []) }
            let region = DocumentIntakeBox(left: Double(box.0), top: Double(box.1), right: Double(box.2 + 1), bottom: Double(box.3 + 1))
            return DocumentRecognition(
                width: width, height: height, text: "X", regions: [DocumentRecognizedRegion(scalarStart: 0, scalarEnd: 1, box: region)])
        }
    }

    private func assertBox(_ box: DocumentIntakeBox?, _ expected: [Double], accuracy: Double, file: StaticString = #filePath, line: UInt = #line) {
        guard let box else { return XCTFail("missing box", file: file, line: line) }
        for (actual, want) in zip([box.left, box.top, box.right, box.bottom], expected) {
            XCTAssertEqual(actual, want, accuracy: accuracy, file: file, line: line)
        }
    }

    /// A stored-pixel image, white with a black block at top-left `(30, 20)`-`(50, 40)`.
    private func syntheticImage() throws -> CGImage {
        let context = try XCTUnwrap(
            CGContext(
                data: nil, width: 120, height: 80, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 120, height: 80))
        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 30, y: 40, width: 20, height: 20))
        return try XCTUnwrap(context.makeImage())
    }

    private func tiff(orientations: [Int]) throws -> Data {
        let data = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(data, "public.tiff" as CFString, orientations.count, nil))
        let image = try syntheticImage()
        for orientation in orientations {
            CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation] as CFDictionary)
        }
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return data as Data
    }

    final class DocumentIntakeImageIOTests: XCTestCase {
        func testOrientedCroppedFramesMapBackToStoredPixels() async throws {
            let recognizer = DarkPixelRecognizer()
            let crop = DocumentIntakeBox(left: 10, top: 10, right: 110, bottom: 70)
            let source = ImageIODocumentIntakeSource(data: try tiff(orientations: [1, 6, 3, 8]), crop: crop)
            let report = try await DocumentIntake(recognizer: recognizer).run(source)
            XCTAssertEqual(report.outcome, .complete)
            XCTAssertEqual(report.text, "X\n\nX\n\nX\n\nX")
            XCTAssertEqual(report.pages.compactMap { $0.cropEvidence == nil ? nil : $0.page }, [0, 1, 2, 3])
            XCTAssertEqual(report.regions.map { $0.visual?.rotation }, [0, 90, 180, 270])
            for region in report.regions {
                assertBox(region.sourceBox, [30, 20, 50, 40], accuracy: 0)
                XCTAssertEqual(region.visual?.unit, .pixel)
            }
            XCTAssertEqual(recognizer.seen.all.map { [$0.pixelWidth, $0.pixelHeight] }, [[100, 60], [60, 100], [100, 60], [60, 100]])
            XCTAssertTrue(recognizer.seen.all.allSatisfy(\.isReleased))
        }

        func testMirroredOrientationIsAnAmbiguousTransform() async throws {
            let report = try await DocumentIntake(recognizer: DarkPixelRecognizer()).run(ImageIODocumentIntakeSource(data: try tiff(orientations: [2, 1])))
            XCTAssertEqual(report.outcome, .partial)
            XCTAssertEqual(report.reasons, [.ambiguousTransform])
            XCTAssertEqual(report.pages.map(\.status), [.held, .mapped])
        }

        func testUnreadableOversizedAndUnsupportedInputsAreRefused() async throws {
            let intake = DocumentIntake(recognizer: DarkPixelRecognizer(), limits: DocumentIntakeLimits(maxDocumentBytes: 64))
            let oversized = try await intake.run(ImageIODocumentIntakeSource(data: try tiff(orientations: [1])))
            XCTAssertEqual(oversized.reasons, [.sizeLimit])
            let unsupported = try await intake.run(ImageIODocumentIntakeSource(data: Data("synthetic".utf8)))
            XCTAssertEqual(unsupported.reasons, [.unsupportedDocument])
            let missing = FileManager.default.temporaryDirectory.appending(path: "openmed-missing-\(UUID().uuidString).tiff")
            let denied = try await intake.run(ImageIODocumentIntakeSource(url: missing))
            XCTAssertEqual(denied.reasons, [.permissionDenied])
            XCTAssertFalse(String(describing: denied).contains("openmed-missing"))
        }

        func testVisionLayoutOrdersLinesAndConvertsNormalizedBoxes() {
            let lines: [[DocumentTextLayout.Word]] = [
                [DocumentTextLayout.Word(text: "रवि", box: CGRect(x: 0.5, y: 0.1, width: 0.2, height: 0.1))],
                [
                    DocumentTextLayout.Word(text: "患者", box: CGRect(x: 0.1, y: 0.8, width: 0.2, height: 0.1)),
                    DocumentTextLayout.Word(text: "😀", box: CGRect(x: 0.35, y: 0.8, width: 0.1, height: 0.15)),
                ],
                [],
            ]
            let recognition = DocumentTextLayout.assemble(lines: lines, width: 200, height: 100)
            XCTAssertEqual(recognition.text, "患者 😀\nरवि")
            XCTAssertEqual(recognition.regions.map { [$0.scalarStart, $0.scalarEnd] }, [[0, 2], [3, 4], [5, 8]])
            assertBox(recognition.regions[0].box, [20, 10, 60, 20], accuracy: 1e-9)
            assertBox(DocumentTextLayout.pixels(CGRect(x: -0.1, y: 0.9, width: 1.3, height: 0.3), 200, 100), [0, 0, 200, 10], accuracy: 1e-9)
            XCTAssertFalse(String(describing: recognition).contains("患者"))
        }
    }
#endif

#if canImport(PDFKit)
    final class DocumentIntakePDFKitTests: XCTestCase {
        /// A one-page synthetic PDF with an offset media box and a black block whose
        /// top-left media-box coordinates are `(40, 50)`-`(80, 80)` points.
        private func pdf(rotation: Int, crop: Bool = true) throws -> Data {
            let data = NSMutableData()
            var media = CGRect(x: 10, y: 20, width: 200, height: 100)
            let consumer = try XCTUnwrap(CGDataConsumer(data: data))
            let context = try XCTUnwrap(CGContext(consumer: consumer, mediaBox: &media, nil))
            context.beginPDFPage(nil)
            context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            context.fill(CGRect(x: 50, y: 40, width: 40, height: 30))
            context.endPDFPage()
            context.closePDF()
            let document = try XCTUnwrap(PDFDocument(data: data as Data))
            let page = try XCTUnwrap(document.page(at: 0))
            if crop { page.setBounds(CGRect(x: 30, y: 30, width: 160, height: 80), for: .cropBox) }
            page.rotation = rotation
            return try XCTUnwrap(document.dataRepresentation())
        }

        func testRotatedCroppedPagesMapBackToMediaBoxPoints() async throws {
            for rotation in [0, 90, 180, 270] {
                let recognizer = DarkPixelRecognizer()
                let report = try await DocumentIntake(recognizer: recognizer).run(PDFKitDocumentIntakeSource(data: try pdf(rotation: rotation)))
                XCTAssertEqual(report.outcome, .complete, "\(rotation)")
                let page = try XCTUnwrap(report.pages.first)
                guard case .visual(let geometry)? = page.evidence?.projection, case .visual(let crop)? = page.cropEvidence?.projection else {
                    return XCTFail("missing page evidence")
                }
                XCTAssertEqual(geometry.unit, .point)
                XCTAssertEqual([geometry.width, geometry.height], [200, 100])
                XCTAssertEqual(geometry.rotation, Int64(rotation))
                assertBox(crop.sourceBox.map { DocumentIntakeBox(left: $0.left, top: $0.top, right: $0.right, bottom: $0.bottom) }, [20, 10, 180, 90], accuracy: 0)
                let region = try XCTUnwrap(report.regions.first)
                assertBox(region.sourceBox, [40, 50, 80, 80], accuracy: 0.5)
                let quarter = rotation == 90 || rotation == 270
                XCTAssertEqual(recognizer.seen.all.map { [$0.pixelWidth, $0.pixelHeight] }, [quarter ? [160, 320] : [320, 160]])
                XCTAssertTrue(recognizer.seen.all.allSatisfy(\.isReleased))
            }
        }

        func testUncroppedPageHasNoCropEvidence() async throws {
            let report = try await DocumentIntake(recognizer: DarkPixelRecognizer()).run(
                PDFKitDocumentIntakeSource(data: try pdf(rotation: 0, crop: false), renderScale: 1))
            XCTAssertNil(report.pages.first?.cropEvidence)
            assertBox(report.regions.first?.sourceBox, [40, 50, 80, 80], accuracy: 1)
        }

        func testPDFGeometryRejectsUnrepresentableBoxes() throws {
            let media = CGRect(x: 0, y: 0, width: 100, height: 100)
            XCTAssertThrowsError(try PDFKitIntakeDocument.geometry(mediaBox: media, cropBox: CGRect(x: 200, y: 200, width: 5, height: 5), rotation: 0))
            XCTAssertThrowsError(try PDFKitIntakeDocument.geometry(mediaBox: media, cropBox: media, rotation: 45))
            XCTAssertEqual(try PDFKitIntakeDocument.geometry(mediaBox: media, cropBox: media, rotation: -90).rotation, 270)
        }

        func testDocumentHandleIsReleasedOnCloseAndMalformedPDFIsRefused() async throws {
            let document = try XCTUnwrap(PDFKitDocumentIntakeSource(data: try pdf(rotation: 0)).open(limits: DocumentIntakeLimits()) as? PDFKitIntakeDocument)
            XCTAssertTrue(document.isOpen)
            document.close()
            XCTAssertFalse(document.isOpen)
            XCTAssertThrowsError(try document.renderPage(at: 0, limits: DocumentIntakeLimits())) { error in
                XCTAssertEqual(error as? DocumentIntakeReason, .permissionLost)
            }
            let report = try await DocumentIntake(recognizer: DarkPixelRecognizer()).run(PDFKitDocumentIntakeSource(data: Data("%PDF-synthetic".utf8)))
            XCTAssertEqual(report.reasons, [.unsupportedDocument])
        }
    }
#endif
