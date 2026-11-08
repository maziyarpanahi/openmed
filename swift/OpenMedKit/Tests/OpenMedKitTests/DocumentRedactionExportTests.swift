import Foundation
import XCTest

@testable import OpenMedKit

#if canImport(CoreGraphics) && canImport(ImageIO)
    import CoreGraphics
    import ImageIO
#endif
#if canImport(CoreText)
    import CoreText
#endif
#if canImport(PDFKit)
    import PDFKit
#endif

private let exportDigest = String(repeating: "a", count: 64)

/// One recognized word per region, as `DocumentIntakeMapper` input.
private func mappedReport(text: String, regions: [(Range<Int>, DocumentIntakeBox)], width: Int = 200, height: Int = 100) -> DocumentIntakeReport {
    let recognition = DocumentRecognition(
        width: width, height: height, text: text,
        regions: regions.map { DocumentRecognizedRegion(scalarStart: $0.0.lowerBound, scalarEnd: $0.0.upperBound, box: $0.1) })
    let geometry = DocumentPageGeometry(width: Double(width), height: Double(height))
    return DocumentIntakeMapper.map(sourceDigest: exportDigest, pages: [.rendered(geometry, recognition)])
}

final class DocumentRedactionContractTests: XCTestCase {
    private func box(_ left: Double, _ top: Double, _ right: Double, _ bottom: Double) -> DocumentIntakeBox {
        DocumentIntakeBox(left: left, top: top, right: right, bottom: bottom)
    }

    func testBindingIsOrderFreeAndSharedWithAndroid() {
        let report = mappedReport(text: "Jane Roe", regions: [(0..<4, box(10, 10, 50, 30)), (5..<8, box(60, 10, 90, 30))])
        let masks: [DocumentRedactionMask] = [.span(0..<4), .box(page: 0, box: box(1, 2, 3, 4.5))]
        let request = DocumentRedactionRequest(report: report, masks: masks)
        // The same vector is asserted by Android's DocumentRedactionExportTest.
        XCTAssertEqual(request.binding, "9e1f9f4ea6a29afb6aeaddae7371b52ea919ae6e800b19c320d8b5a688eb2d4d")
        XCTAssertEqual(DocumentRedactionRequest(report: report, masks: masks.reversed() + masks).binding, request.binding)
        XCTAssertNotEqual(DocumentRedactionRequest(report: report, masks: [.span(0..<4)]).binding, request.binding)
        XCTAssertNotEqual(DocumentRedactionRequest(report: report, masks: masks, policy: DocumentRedactionPolicy(padding: 3)).binding, request.binding)
        XCTAssertNotEqual(
            DocumentRedactionRequest(report: report, masks: masks, policy: DocumentRedactionPolicy(policyID: "hipaa_safe_harbor")).binding,
            request.binding)
        let other = mappedReport(text: "Jane Rob", regions: [(0..<4, box(10, 10, 50, 30)), (5..<8, box(60, 10, 90, 30))])
        XCTAssertNotEqual(DocumentRedactionRequest(report: other, masks: masks).binding, request.binding)
        XCTAssertFalse(String(describing: request).contains("Jane"))
    }

    func testDeidentificationResultsBecomeSpanMasksOnlyForTheSameText() throws {
        let report = mappedReport(text: "Jane Roe", regions: [(0..<4, box(10, 10, 50, 30)), (5..<8, box(60, 10, 90, 30))])
        let entities = [
            DeidentificationResult.PIIEntityRecord(text: "Jane", label: "NAME", start: 0, end: 4, confidence: 0.9, action: "mask"),
            DeidentificationResult.PIIEntityRecord(text: "Roe", label: "NAME", start: 5, end: 8, confidence: 0.9, action: "keep"),
        ]
        let result = DeidentificationResult(originalText: "Jane Roe", deidentifiedText: "[NAME] Roe", piiEntities: entities, method: "mask")
        XCTAssertEqual(try DocumentRedactionRequest(report: report, deidentification: result).masks, [.span(0..<4)])
        let stale = DeidentificationResult(originalText: "Jane Rob", deidentifiedText: "[NAME] Rob", piiEntities: entities, method: "mask")
        XCTAssertThrowsError(try DocumentRedactionRequest(report: report, deidentification: stale)) {
            XCTAssertEqual($0 as? DocumentRedactionExportError, .staleRequest)
        }
    }

    func testPlanPlacesSpansThroughRegionsAndRejectsUnplacedOrOutOfRangeMasks() throws {
        // "Roe" has no recognized region, so a mask over it cannot be placed.
        let report = mappedReport(text: "Jane Roe", regions: [(0..<4, box(10, 10, 50, 30))])
        let plan = try DocumentRedactionExportContract.plan(DocumentRedactionRequest(report: report, masks: [.span(0..<5), .box(page: 0, box: box(0, 0, 5, 5))]))
        XCTAssertEqual(plan.map(\.boxes), [[box(10, 10, 50, 30), box(0, 0, 5, 5)]])
        for (mask, error) in [
            (DocumentRedactionMask.span(5..<8), DocumentRedactionExportError.maskUnplaced),
            (.span(0..<9), .maskOutOfRange), (.span(3..<3), .maskOutOfRange),
            (.box(page: 1, box: box(0, 0, 5, 5)), .maskOutOfRange), (.box(page: 0, box: box(0, 0, 201, 5)), .maskOutOfRange),
            (.box(page: 0, box: box(5, 0, 5, 5)), .maskOutOfRange), (.box(page: 0, box: box(.nan, 0, 5, 5)), .maskOutOfRange),
        ] {
            XCTAssertThrowsError(try DocumentRedactionExportContract.plan(DocumentRedactionRequest(report: report, masks: [mask]))) {
                XCTAssertEqual($0 as? DocumentRedactionExportError, error, "\(mask)")
            }
        }
        let partial = DocumentIntakeMapper.map(sourceDigest: exportDigest, pages: [.unavailable(.pageMissing)])
        XCTAssertThrowsError(try DocumentRedactionExportContract.plan(DocumentRedactionRequest(report: partial, masks: []))) {
            XCTAssertEqual($0 as? DocumentRedactionExportError, .incompleteReport)
        }
    }

    func testForwardTransformReproducesSharedIntakeVectors() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/document_intake_v1.json"))
        let corpus = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var checked = 0
        for row in try XCTUnwrap(corpus["cases"] as? [[String: Any]]) {
            let exact = (row["tolerance"] as? Double ?? 0) == 0
            for page in row["pages"] as? [[String: Any]] ?? [] {
                guard let geometry = page["geometry"] as? [String: Any], let ocr = page["ocr"] as? [String: Any] else { continue }
                let crop = (geometry["crop"] as? [Double]).map { box($0[0], $0[1], $0[2], $0[3]) }
                let pageGeometry = DocumentPageGeometry(
                    width: try XCTUnwrap(geometry["width"] as? Double), height: try XCTUnwrap(geometry["height"] as? Double),
                    rotation: try XCTUnwrap(geometry["rotation"] as? Int), crop: crop)
                let width = try XCTUnwrap(ocr["width"] as? Int)
                let height = try XCTUnwrap(ocr["height"] as? Int)
                for region in ocr["regions"] as? [[String: Any]] ?? [] {
                    let visible = crop ?? box(0, 0, pageGeometry.width, pageGeometry.height)
                    guard let truth = region["truth"] as? [Double], let expected = region["box"] as? [Double],
                        truth[0] >= visible.left, truth[1] >= visible.top, truth[2] <= visible.right, truth[3] <= visible.bottom,
                        let mapped = DocumentRedactionExportContract.imageBox(
                            box(truth[0], truth[1], truth[2], truth[3]), geometry: pageGeometry, imageWidth: width, imageHeight: height)
                    else { continue }
                    var values = [mapped.left, mapped.top, mapped.right, mapped.bottom]
                    if !exact { values = [values[0].rounded(.down), values[1].rounded(.down), values[2].rounded(.up), values[3].rounded(.up)] }
                    let clamped = [
                        min(max(expected[0], 0), Double(width)), min(max(expected[1], 0), Double(height)),
                        min(max(expected[2], 0), Double(width)), min(max(expected[3], 0), Double(height)),
                    ]
                    XCTAssertTrue(values == expected || values == clamped, "\(row["id"] ?? "")")
                    checked += 1
                }
            }
        }
        XCTAssertGreaterThan(checked, 20)
        XCTAssertEqual(DocumentRedactionExportContract.pixelRect(box(10.2, 3.5, 20.1, 7), padding: 2, imageWidth: 21, imageHeight: 30), [8, 1, 21, 9])
        XCTAssertNil(DocumentRedactionExportContract.pixelRect(box(30, 3, 40, 7), padding: 2, imageWidth: 21, imageHeight: 30))
    }

    func testResidualSearchNormalizesTokensAndScriptsWithoutSpaces() {
        XCTAssertEqual(DocumentRedactionExportContract.tokens("Ｊａｎｅ  O'Neil, रवि 2026"), ["jane", "o", "neil", "रवि", "2026"])
        let contract = DocumentRedactionExportContract.self
        XCTAssertTrue(contract.containsResidual(masked: ["Jane Roe"], extracted: "Seen by JANE today", minimumScalars: 2))
        XCTAssertFalse(contract.containsResidual(masked: ["Jane Roe"], extracted: "Seen by Janet today", minimumScalars: 2))
        XCTAssertFalse(contract.containsResidual(masked: ["J. Roe"], extracted: "J was seen", minimumScalars: 2))
        XCTAssertTrue(contract.containsResidual(masked: ["王小明"], extracted: "患者王小明来访", minimumScalars: 2))
        XCTAssertFalse(contract.containsResidual(masked: ["王小明"], extracted: "患者王小来访", minimumScalars: 2))
        XCTAssertTrue(contract.containsResidual(masked: ["रवि"], extracted: "नाम रवि", minimumScalars: 2))
    }
}

#if canImport(CoreGraphics) && canImport(ImageIO) && canImport(PDFKit) && canImport(CoreText)
    private let palette: [(Double, Double, Double)] = [(1, 0, 0), (0, 0.6, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    private func deviceColor(_ index: Int) -> CGColor {
        let rgb = palette[index]
        return CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [rgb.0, rgb.1, rgb.2, 1])!
    }

    /// Recognizes each solid palette-colored block as one word, in palette order.
    /// Rendering may shift colors slightly, so each palette color matches within a margin.
    private struct BlockRecognizer: DocumentTextRecognizer {
        enum Behavior: Sendable { case read, fail, cancel }
        let words: [String]
        var behavior = Behavior.read

        func recognize(_ image: any DocumentIntakeImage) async throws -> DocumentRecognition {
            switch behavior {
            case .fail: throw NSError(domain: "synthetic-private-diagnostic", code: 3)
            case .cancel:
                withUnsafeCurrentTask { $0?.cancel() }
                try await Task.sleep(nanoseconds: 1_000_000_000)
            case .read: break
            }
            let cgImage = try XCTUnwrap((image as? CGDocumentIntakeImage)?.cgImage)
            let pixels = try XCTUnwrap(DocumentRedactionRaster.rgba(cgImage))
            let width = cgImage.width
            var boxes: [Int: [Int]] = [:]
            var counts: [Int: Int] = [:]
            for y in 0..<cgImage.height {
                for x in 0..<width {
                    let index = (y * width + x) * 4
                    let rgb = (0..<3).map { Double(pixels[index + $0]) }
                    guard
                        let color = palette.indices.first(where: { colorIndex in
                            let target = palette[colorIndex]
                            return abs(rgb[0] - target.0 * 255) + abs(rgb[1] - target.1 * 255) + abs(rgb[2] - target.2 * 255) <= 120
                        })
                    else { continue }
                    let current = boxes[color] ?? [x, y, x + 1, y + 1]
                    boxes[color] = [min(current[0], x), min(current[1], y), max(current[2], x + 1), max(current[3], y + 1)]
                    counts[color, default: 0] += 1
                }
            }
            var text = ""
            var regions: [DocumentRecognizedRegion] = []
            for color in boxes.keys.sorted() where counts[color, default: 0] >= 4 && color < words.count {
                if !text.isEmpty { text += " " }
                let start = text.unicodeScalars.count
                text += words[color]
                let found = boxes[color]!.map(Double.init)
                regions.append(
                    DocumentRecognizedRegion(
                        scalarStart: start, scalarEnd: text.unicodeScalars.count,
                        box: DocumentIntakeBox(left: found[0], top: found[1], right: found[2], bottom: found[3])))
            }
            return DocumentRecognition(width: width, height: cgImage.height, text: text, regions: regions)
        }
    }

    /// A colored word block at top-left media-box points.
    private struct Block {
        let color: Int
        let rect: CGRect
    }

    private struct SyntheticPage {
        var blocks: [Block]
        var rotation = 0
        var cropBox: CGRect?
    }

    /// Synthetic PDF with colored word blocks, an invisible text layer, a title and an annotation.
    private func syntheticPDF(_ pages: [SyntheticPage], hiddenText: String = "Jane Roe MRN 4417", annotate: Bool = true) throws -> Data {
        let data = NSMutableData()
        var media = CGRect(x: 0, y: 0, width: 200, height: 100)
        let consumer = try XCTUnwrap(CGDataConsumer(data: data))
        let info = [kCGPDFContextTitle: "Synthetic Jane Roe", kCGPDFContextAuthor: "Synthetic Author"] as CFDictionary
        let context = try XCTUnwrap(CGContext(consumer: consumer, mediaBox: &media, info))
        for page in pages {
            context.beginPDFPage(nil)
            for block in page.blocks {
                context.setFillColor(deviceColor(block.color))
                context.fill(CGRect(x: block.rect.minX, y: media.height - block.rect.maxY, width: block.rect.width, height: block.rect.height))
            }
            let font = CTFontCreateWithName("Helvetica" as CFString, 8, nil)
            let line = CTLineCreateWithAttributedString(
                NSAttributedString(string: hiddenText, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font]))
            context.setTextDrawingMode(.invisible)
            context.textPosition = CGPoint(x: 10, y: 10)
            CTLineDraw(line, context)
            context.endPDFPage()
        }
        context.closePDF()
        let document = try XCTUnwrap(PDFDocument(data: data as Data))
        for (index, spec) in pages.enumerated() {
            let page = try XCTUnwrap(document.page(at: index))
            page.rotation = spec.rotation
            if let crop = spec.cropBox { page.setBounds(crop, for: .cropBox) }
            if annotate {
                let note = PDFAnnotation(bounds: CGRect(x: 150, y: 70, width: 20, height: 20), forType: .text, withProperties: nil)
                note.contents = "Synthetic Jane Roe"
                page.addAnnotation(note)
            }
        }
        return try XCTUnwrap(document.dataRepresentation())
    }

    private func storedFiles(_ directory: URL) -> [String] {
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey])
        return (enumerator?.allObjects as? [URL] ?? [])
            .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true }
            .map(\.lastPathComponent)
    }

    final class DocumentRedactionExporterTests: XCTestCase {
        private var directory: URL!

        override func setUpWithError() throws {
            directory = FileManager.default.temporaryDirectory.appending(path: "openmed-redaction-tests-\(UUID().uuidString)")
        }

        override func tearDownWithError() throws {
            try? FileManager.default.removeItem(at: directory)
        }

        private let words = ["Jane", "Roe", "Visit", "Plan", "Jane"]

        /// Two pages: "Jane Roe", then "Visit Plan" on a rotated, cropped page.
        private func twoPagePDF() throws -> Data {
            try syntheticPDF([
                SyntheticPage(blocks: [Block(color: 0, rect: CGRect(x: 20, y: 20, width: 40, height: 20)), Block(color: 1, rect: CGRect(x: 80, y: 20, width: 30, height: 20))]),
                SyntheticPage(
                    blocks: [Block(color: 2, rect: CGRect(x: 30, y: 30, width: 40, height: 20)), Block(color: 3, rect: CGRect(x: 100, y: 30, width: 40, height: 20))],
                    rotation: 90, cropBox: CGRect(x: 10, y: 10, width: 180, height: 80)),
            ])
        }

        func testExportWritesAVerifiedImageOnlyPDFAndSharesItOnlyOnRequest() async throws {
            let pdf = try twoPagePDF()
            let source = PDFKitDocumentIntakeSource(data: pdf)
            let recognizer = BlockRecognizer(words: words)
            let report = try await DocumentIntake(recognizer: recognizer).run(source)
            XCTAssertEqual(report.outcome, .complete)
            XCTAssertEqual(report.text, "Jane Roe\n\nVisit Plan")
            let original = try XCTUnwrap(PDFDocument(data: pdf))
            XCTAssertTrue(original.string?.contains("Jane") == true, "the source carries a hidden text layer")
            XCTAssertNotNil(original.documentAttributes?[PDFDocumentAttribute.titleAttribute])
            XCTAssertFalse(original.page(at: 0)?.annotations.isEmpty ?? true)

            let request = DocumentRedactionRequest(report: report, masks: [.span(0..<8), .span(10..<15)])
            let exporter = try DocumentRedactionExporter(recognizer: recognizer, directory: directory)
            let artifact = try await exporter.export(request, from: source)
            let receipt = artifact.receipt
            XCTAssertEqual(receipt.profile, "openmed.redaction-export.image-pdf.v1")
            XCTAssertEqual(receipt.binding, request.binding)
            XCTAssertEqual(receipt.sourceDigest, report.sourceDigest)
            XCTAssertEqual([receipt.pageCount, receipt.maskCount, receipt.paintedRegionCount], [2, 2, 3])
            XCTAssertEqual(receipt.checks, DocumentRedactionCheck.allCases)
            XCTAssertFalse(String(describing: artifact).contains("Jane"))
            let artifactCount = await exporter.artifactCount
            XCTAssertEqual(artifactCount, 1)

            let url = try await exporter.shareURL(for: artifact, matching: request)
            XCTAssertTrue(url.path.hasPrefix(directory.path))
            XCTAssertEqual(url.lastPathComponent, "\(artifact.id).pdf")
            let exported = try Data(contentsOf: url)
            XCTAssertNotEqual(exported, pdf)
            XCTAssertEqual(DocumentRedactionExportContract.sha256(exported), receipt.outputDigest)
            XCTAssertEqual(exported.count, receipt.byteCount)
            XCTAssertEqual(try DocumentRedactionPDF.inspect(exported, maxPixels: 40_000_000).map { [$0.pixelWidth, $0.pixelHeight] }, [[400, 200], [160, 360]])
            let document = try XCTUnwrap(PDFDocument(data: exported))
            XCTAssertEqual(document.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "", "")
            XCTAssertTrue((0..<2).allSatisfy { document.page(at: $0)?.annotations.isEmpty == true })
            XCTAssertNil(document.documentAttributes?[PDFDocumentAttribute.titleAttribute])
            for marker in ["Jane", "Synthetic", "/Font", "/Annots", "/Title"] {
                XCTAssertNil(exported.range(of: Data(marker.utf8)), marker)
            }
            let reread = try await DocumentIntake(recognizer: recognizer).run(PDFKitDocumentIntakeSource(data: exported))
            XCTAssertEqual(reread.text, "\n\nPlan")
            XCTAssertEqual(reread.pages.map(\.status), [.mapped, .mapped])

            await exporter.complete(artifact)
            XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
            await assertExportError(.artifactUnavailable) { _ = try await exporter.shareURL(for: artifact, matching: request) }
            XCTAssertEqual(storedFiles(directory), [])
        }

        func testRotatedCroppedImageFramesExportOnlyTheExaminedUprightCrop() async throws {
            let context = try XCTUnwrap(
                CGContext(
                    data: nil, width: 120, height: 80, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: 120, height: 80))
            for (color, x) in [(0, 30), (1, 70)] {
                context.setFillColor(deviceColor(color))
                context.fill(CGRect(x: x, y: 40, width: 20, height: 20))
            }
            let tiff = NSMutableData()
            let destination = try XCTUnwrap(CGImageDestinationCreateWithData(tiff, "public.tiff" as CFString, 1, nil))
            let properties: [CFString: Any] = [
                kCGImagePropertyOrientation: 6, kCGImagePropertyTIFFDictionary: [kCGImagePropertyTIFFArtist: "Synthetic Jane"],
            ]
            CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), properties as CFDictionary)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
            let source = ImageIODocumentIntakeSource(data: tiff as Data, crop: DocumentIntakeBox(left: 10, top: 10, right: 110, bottom: 70))
            let recognizer = BlockRecognizer(words: words)
            let report = try await DocumentIntake(recognizer: recognizer).run(source)
            XCTAssertEqual(report.text, "Jane Roe")
            XCTAssertNotNil(report.pages.first?.cropEvidence)

            let request = DocumentRedactionRequest(report: report, masks: [.span(0..<4)])
            let exporter = try DocumentRedactionExporter(recognizer: recognizer, directory: directory)
            let artifact = try await exporter.export(request, from: source)
            let exported = try Data(contentsOf: try await exporter.shareURL(for: artifact, matching: request))
            let page = try XCTUnwrap(try DocumentRedactionPDF.inspect(exported, maxPixels: 40_000_000).first)
            XCTAssertEqual([page.pixelWidth, page.pixelHeight], [60, 100])
            XCTAssertEqual([page.mediaWidth, page.mediaHeight], ["60", "100"])
            XCTAssertNil(exported.range(of: Data("Synthetic".utf8)))
            let reread = try await DocumentIntake(recognizer: recognizer).run(PDFKitDocumentIntakeSource(data: exported, renderScale: 1))
            XCTAssertEqual(reread.text, "Roe")
            XCTAssertEqual(artifact.receipt.paintedRegionCount, 1)
        }

        func testResidualIdentifierElsewhereRefusesTheExportAndLeavesNoFile() async throws {
            // Negative control: "Jane" also appears as a second, unmasked block.
            let pdf = try syntheticPDF([
                SyntheticPage(blocks: [
                    Block(color: 0, rect: CGRect(x: 20, y: 20, width: 40, height: 20)), Block(color: 1, rect: CGRect(x: 80, y: 20, width: 30, height: 20)),
                    Block(color: 4, rect: CGRect(x: 20, y: 60, width: 40, height: 20)),
                ])
            ])
            let source = PDFKitDocumentIntakeSource(data: pdf)
            let recognizer = BlockRecognizer(words: words)
            let report = try await DocumentIntake(recognizer: recognizer).run(source)
            XCTAssertEqual(report.text, "Jane Roe Jane")
            let exporter = try DocumentRedactionExporter(recognizer: recognizer, directory: directory)
            await assertExportError(.residualText) { _ = try await exporter.export(DocumentRedactionRequest(report: report, masks: [.span(0..<4)]), from: source) }
            XCTAssertEqual(storedFiles(directory), [])
            let artifact = try await exporter.export(DocumentRedactionRequest(report: report, masks: [.span(0..<4), .span(9..<13)]), from: source)
            XCTAssertEqual(artifact.receipt.paintedRegionCount, 2)
        }

        func testUnverifiableOrChangedSourcesNeverProduceArtifacts() async throws {
            let pdf = try twoPagePDF()
            let source = PDFKitDocumentIntakeSource(data: pdf)
            let report = try await DocumentIntake(recognizer: BlockRecognizer(words: words)).run(source)
            let request = DocumentRedactionRequest(report: report, masks: [.span(0..<4)])
            let failing = try DocumentRedactionExporter(recognizer: BlockRecognizer(words: words, behavior: .fail), directory: directory)
            await assertExportError(.verificationIncomplete) { _ = try await failing.export(request, from: source) }
            let exporter = try DocumentRedactionExporter(recognizer: BlockRecognizer(words: words), directory: directory)
            let changed = PDFKitDocumentIntakeSource(data: try syntheticPDF([SyntheticPage(blocks: [])], annotate: false))
            await assertExportError(.sourceChanged) { _ = try await exporter.export(request, from: changed) }
            await assertExportError(.sourceChanged) { _ = try await exporter.export(request, from: PDFKitDocumentIntakeSource(data: Data("%PDF-x".utf8))) }
            let partial = try await DocumentIntake(recognizer: BlockRecognizer(words: words), limits: DocumentIntakeLimits(maxPages: 1)).run(source)
            XCTAssertEqual(partial.outcome, .partial)
            await assertExportError(.incompleteReport) { _ = try await exporter.export(DocumentRedactionRequest(report: partial, masks: []), from: source) }
            let failingCount = await failing.artifactCount
            let exporterCount = await exporter.artifactCount
            XCTAssertEqual([failingCount, exporterCount], [0, 0])
            XCTAssertEqual(storedFiles(directory), [])
        }

        func testCancelledExportRemovesPartialOutput() async throws {
            let pdf = try twoPagePDF()
            let source = PDFKitDocumentIntakeSource(data: pdf)
            let report = try await DocumentIntake(recognizer: BlockRecognizer(words: words)).run(source)
            let exporter = try DocumentRedactionExporter(recognizer: BlockRecognizer(words: words, behavior: .cancel), directory: directory)
            let task = Task { try await exporter.export(DocumentRedactionRequest(report: report, masks: [.span(0..<4)]), from: source) }
            do {
                _ = try await task.value
                XCTFail("cancelled export returned an artifact")
            } catch {
                XCTAssertTrue(error is CancellationError)
            }
            let early = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await exporter.export(DocumentRedactionRequest(report: report, masks: []), from: source)
            }
            do {
                _ = try await early.value
                XCTFail("cancelled export returned an artifact")
            } catch {
                XCTAssertTrue(error is CancellationError)
            }
            let count = await exporter.artifactCount
            XCTAssertEqual(count, 0)
            XCTAssertEqual(storedFiles(directory), [])
        }

        func testChangedMasksPolicyOrBytesInvalidateTheArtifact() async throws {
            let pdf = try twoPagePDF()
            let source = PDFKitDocumentIntakeSource(data: pdf)
            let recognizer = BlockRecognizer(words: words)
            let report = try await DocumentIntake(recognizer: recognizer).run(source)
            let request = DocumentRedactionRequest(report: report, masks: [.span(0..<4)])
            let exporter = try DocumentRedactionExporter(recognizer: recognizer, directory: directory)

            let first = try await exporter.export(request, from: source)
            let widened = DocumentRedactionRequest(report: report, masks: [.span(0..<8)])
            await assertExportError(.staleRequest) { _ = try await exporter.shareURL(for: first, matching: widened) }
            await assertExportError(.artifactUnavailable) { _ = try await exporter.shareURL(for: first, matching: request) }

            let second = try await exporter.export(request, from: source)
            let third = try await exporter.export(widened, from: source)
            await exporter.invalidate(keeping: widened)
            await assertExportError(.artifactUnavailable) { _ = try await exporter.shareURL(for: second, matching: request) }
            let url = try await exporter.shareURL(for: third, matching: widened)
            let stricter = DocumentRedactionRequest(report: report, masks: [.span(0..<8)], policy: DocumentRedactionPolicy(padding: 4))
            XCTAssertNotEqual(stricter.binding, widened.binding)

            var bytes = try Data(contentsOf: url)
            bytes[bytes.count - 2] ^= 0x01
            try bytes.write(to: url)
            await assertExportError(.artifactChanged) { _ = try await exporter.shareURL(for: third, matching: widened) }
            XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))

            _ = try await exporter.export(request, from: source)
            await exporter.invalidateAll()
            let count = await exporter.artifactCount
            XCTAssertEqual(count, 0)
            XCTAssertEqual(storedFiles(directory), [])
        }

        func testRestartSweepRemovesLeftoversButKeepsLiveSessions() async throws {
            let leftover = directory.appending(path: "0123456789abcdef0123456789abcdef")
            try FileManager.default.createDirectory(at: leftover, withIntermediateDirectories: true)
            try Data("synthetic".utf8).write(to: leftover.appending(path: "pending-1.pdf"))
            try Data("synthetic".utf8).write(to: directory.appending(path: "stray.pdf"))

            let pdf = try twoPagePDF()
            let source = PDFKitDocumentIntakeSource(data: pdf)
            let recognizer = BlockRecognizer(words: words)
            let report = try await DocumentIntake(recognizer: recognizer).run(source)
            let request = DocumentRedactionRequest(report: report, masks: [.span(0..<4)])
            var live: DocumentRedactionExporter? = try DocumentRedactionExporter(recognizer: recognizer, directory: directory)
            XCTAssertFalse(FileManager.default.fileExists(atPath: leftover.path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appending(path: "stray.pdf").path))

            let artifact = try await live!.export(request, from: source)
            let second = try DocumentRedactionExporter(recognizer: recognizer, directory: directory)
            let url = try await live!.shareURL(for: artifact, matching: request)
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
            await assertExportError(.artifactUnavailable) { _ = try await second.shareURL(for: artifact, matching: request) }
            live = nil
            XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
            XCTAssertEqual(storedFiles(directory), [])
        }

        private func assertExportError(
            _ expected: DocumentRedactionExportError, file: StaticString = #filePath, line: UInt = #line, _ body: () async throws -> Void
        ) async {
            do {
                try await body()
                XCTFail("expected \(expected.rawValue)", file: file, line: line)
            } catch {
                XCTAssertEqual(error as? DocumentRedactionExportError, expected, file: file, line: line)
                XCTAssertEqual(error.localizedDescription, expected.rawValue, file: file, line: line)
            }
        }
    }

    final class DocumentRedactionProfileTests: XCTestCase {
        private func profilePDF() throws -> Data {
            let rgb = Data((0..<(4 * 3)).flatMap { _ in [UInt8(255), 0, 0] })
            let pages = [
                try DocumentRedactionPDF.page(rgb: rgb, pixelWidth: 4, pixelHeight: 3, mediaWidth: 2, mediaHeight: 1.5),
                try DocumentRedactionPDF.page(rgb: rgb, pixelWidth: 3, pixelHeight: 4, mediaWidth: 612.25, mediaHeight: 792),
            ]
            let data = DocumentRedactionPDF.write(pages)
            XCTAssertEqual(try DocumentRedactionPDF.inspect(data, maxPixels: 100), pages)
            return data
        }

        func testProfileRoundTripsAndRendersThroughPDFKit() throws {
            let data = try profilePDF()
            let pages = try DocumentRedactionPDF.inspect(data, maxPixels: 100)
            XCTAssertEqual(pages.map { [$0.mediaWidth, $0.mediaHeight] }, [["2", "1.5"], ["612.25", "792"]])
            try DocumentRedactionExporter.inspectWithPDFKit(data, pageCount: 2)
            let document = try XCTUnwrap(PDFDocument(data: data))
            let reference = try XCTUnwrap(document.page(at: 0)?.pageRef)
            let context = try XCTUnwrap(
                CGContext(
                    data: nil, width: 8, height: 6, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            context.scaleBy(x: 4, y: 4)
            context.drawPDFPage(reference)
            let image = try XCTUnwrap(context.makeImage())
            let pixels = try XCTUnwrap(DocumentRedactionRaster.rgba(image))
            let center = ((image.height / 2) * image.width + image.width / 2) * 4
            XCTAssertGreaterThan(pixels[center], 200)
            XCTAssertLessThan(pixels[center + 1], 60)
            XCTAssertEqual(DocumentRedactionPDF.number(14_399.99999), "14400")
            XCTAssertThrowsError(try DocumentRedactionPDF.inspect(data, maxPixels: 11))
        }

        func testTamperedOrForeignPDFsAreRejectedWithTheirCategory() throws {
            let data = try profilePDF()
            var flipped = data
            flipped[20] = UInt8(ascii: "X")
            for (bytes, expected) in [
                (flipped, DocumentRedactionExportError.profileViolation), (data + Data("\n".utf8), .profileViolation),
                (Data(data.prefix(data.count - 7)), .profileViolation),
            ] {
                XCTAssertThrowsError(try DocumentRedactionPDF.inspect(bytes, maxPixels: 100)) {
                    XCTAssertEqual($0 as? DocumentRedactionExportError, expected)
                }
            }
            let text = try syntheticPDF([SyntheticPage(blocks: [])], annotate: false)
            XCTAssertThrowsError(try DocumentRedactionPDF.inspect(text, maxPixels: 100)) { XCTAssertEqual($0 as? DocumentRedactionExportError, .hiddenText) }
            XCTAssertThrowsError(try DocumentRedactionExporter.inspectWithPDFKit(text, pageCount: 1)) {
                XCTAssertEqual($0 as? DocumentRedactionExportError, .hiddenText)
            }

            let annotated = try XCTUnwrap(PDFDocument(data: data))
            annotated.page(at: 0)?.addAnnotation(PDFAnnotation(bounds: CGRect(x: 0, y: 0, width: 1, height: 1), forType: .text, withProperties: nil))
            let annotatedData = try XCTUnwrap(annotated.dataRepresentation())
            XCTAssertThrowsError(try DocumentRedactionPDF.inspect(annotatedData, maxPixels: 100)) {
                XCTAssertEqual($0 as? DocumentRedactionExportError, .attachmentsPresent)
            }
            XCTAssertThrowsError(try DocumentRedactionExporter.inspectWithPDFKit(annotatedData, pageCount: 2)) {
                XCTAssertEqual($0 as? DocumentRedactionExportError, .attachmentsPresent)
            }

            let titled = try XCTUnwrap(PDFDocument(data: data))
            titled.documentAttributes = [PDFDocumentAttribute.titleAttribute: "Synthetic Jane"]
            let titledData = try XCTUnwrap(titled.dataRepresentation())
            XCTAssertThrowsError(try DocumentRedactionPDF.inspect(titledData, maxPixels: 100)) {
                XCTAssertEqual($0 as? DocumentRedactionExportError, .metadataPresent)
            }
            XCTAssertThrowsError(try DocumentRedactionExporter.inspectWithPDFKit(titledData, pageCount: 2)) {
                XCTAssertEqual($0 as? DocumentRedactionExportError, .metadataPresent)
            }
        }

        func testUnpaintedMaskIsNotOpaque() throws {
            let context = try XCTUnwrap(
                CGContext(
                    data: nil, width: 10, height: 10, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: 10, height: 10))
            context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            context.fill(CGRect(x: 0, y: 5, width: 5, height: 5))
            let image = try XCTUnwrap(context.makeImage())
            XCTAssertTrue(DocumentRedactionRaster.isOpaque(image, rects: [DocumentIntakeBox(left: 0, top: 0, right: 0.5, bottom: 0.5)]))
            XCTAssertFalse(DocumentRedactionRaster.isOpaque(image, rects: [DocumentIntakeBox(left: 0.5, top: 0, right: 1, bottom: 0.5)]))
        }
    }
#endif
