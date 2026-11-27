import CryptoKit
import Foundation

/// Document-level intake result shared with Python and Android.
public enum DocumentIntakeOutcome: String, Sendable, CaseIterable {
    case complete, partial, refused
}

/// Closed, value-free intake reasons in report order. Sources throw these directly.
public enum DocumentIntakeReason: String, Error, LocalizedError, Sendable, CaseIterable {
    case unsupportedDocument = "unsupported_document"
    case documentLocked = "document_locked"
    case permissionDenied = "permission_denied"
    case sizeLimit = "size_limit"
    case emptyDocument = "empty_document"
    case pageLimit = "page_limit"
    case pageMissing = "page_missing"
    case permissionLost = "permission_lost"
    case recognitionFailed = "recognition_failed"
    case resourceLimit = "resource_limit"
    case ambiguousTransform = "ambiguous_transform"
    case regionOutOfBounds = "region_out_of_bounds"
    case invalidRecognition = "invalid_recognition"

    public var errorDescription: String? { rawValue }

    fileprivate var order: Int { Self.allCases.firstIndex(of: self) ?? 0 }
}

/// Whether a page contributed text and evidence, or was held with a reason.
public enum DocumentIntakePageStatus: String, Sendable, CaseIterable {
    case mapped, held
}

/// Bounds applied before any page contributes text.
public struct DocumentIntakeLimits: Sendable, Equatable {
    public var maxPages: Int
    public var maxRegionsPerPage: Int
    public var maxPageScalars: Int
    public var maxDocumentBytes: Int
    public var maxRenderPixels: Int

    public init(
        maxPages: Int = 2048, maxRegionsPerPage: Int = 4096, maxPageScalars: Int = 65536,
        maxDocumentBytes: Int = 64 * 1024 * 1024, maxRenderPixels: Int = 40_000_000
    ) {
        self.maxPages = max(1, maxPages)
        self.maxRegionsPerPage = max(1, maxRegionsPerPage)
        self.maxPageScalars = max(1, maxPageScalars)
        self.maxDocumentBytes = max(1, maxDocumentBytes)
        self.maxRenderPixels = max(1, maxRenderPixels)
    }
}

/// Half-open, top-left-origin rectangle in pixels or PDF points.
public struct DocumentIntakeBox: Sendable, Equatable {
    public var left: Double
    public var top: Double
    public var right: Double
    public var bottom: Double

    public init(left: Double, top: Double, right: Double, bottom: Double) {
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
    }
}

/// Original page extent plus the crop and clockwise rotation handed to OCR.
///
/// The OCR image is the crop, turned upright by `rotation`, at any render scale.
/// Inconsistent declarations hold the page as `ambiguous_transform` when mapped.
public struct DocumentPageGeometry: Sendable, Equatable {
    public var width: Double
    public var height: Double
    public var unit: MultimodalSpatialUnit
    public var rotation: Int
    /// Original-page region handed to OCR; `nil` means the whole page.
    public var crop: DocumentIntakeBox?

    public init(width: Double, height: Double, unit: MultimodalSpatialUnit = .pixel, rotation: Int = 0, crop: DocumentIntakeBox? = nil) {
        self.width = width
        self.height = height
        self.unit = unit
        self.rotation = rotation
        self.crop = crop
    }
}

/// One OCR region: Unicode scalar offsets into the page text and an OCR-image box.
public struct DocumentRecognizedRegion: Sendable, Equatable {
    public var scalarStart: Int
    public var scalarEnd: Int
    public var box: DocumentIntakeBox

    public init(scalarStart: Int, scalarEnd: Int, box: DocumentIntakeBox) {
        self.scalarStart = scalarStart
        self.scalarEnd = scalarEnd
        self.box = box
    }
}

/// OCR output for one upright page image. Descriptions never include the text.
public struct DocumentRecognition: Sendable, Equatable, CustomStringConvertible, CustomDebugStringConvertible {
    public var width: Int
    public var height: Int
    public var text: String
    public var regions: [DocumentRecognizedRegion]

    public init(width: Int, height: Int, text: String, regions: [DocumentRecognizedRegion]) {
        self.width = width
        self.height = height
        self.text = text
        self.regions = regions
    }

    public var description: String { "DocumentRecognition(width: \(width), height: \(height), regions: \(regions.count))" }
    public var debugDescription: String { description }
}

/// What a native source produced for one page, in page order.
public enum DocumentPageInput: Sendable, Equatable {
    case rendered(DocumentPageGeometry, DocumentRecognition)
    /// `permissionLost` ends the document; `pageLimit` marks a source that stopped at the limit.
    case unavailable(DocumentIntakeReason)
}

/// Page record: mapped with a text span and evidence, or held with a reason.
public struct DocumentIntakePage: Sendable, Equatable {
    public let page: Int
    public let status: DocumentIntakePageStatus
    public let reason: DocumentIntakeReason?
    /// Unicode scalar offsets into the report text.
    public let textSpan: Range<Int>?
    /// `page_geometry` visual packet carrying the original extent and rotation.
    public let evidence: MultimodalPacket?
    /// `pixels` visual packet for a declared crop.
    public let cropEvidence: MultimodalPacket?

    public var nodeRef: String? {
        guard case .visual(let visual)? = evidence?.projection else { return nil }
        return visual.nodeRef
    }
}

/// Recognized region mapped back to the original page.
public struct DocumentIntakeRegion: Sendable, Equatable {
    public let page: Int
    /// Unicode scalar offsets into the report text.
    public let scalarSpan: Range<Int>
    /// Original-page box with a top-left origin, in the page unit.
    public let sourceBox: DocumentIntakeBox
    public let nodeRef: String
    /// Validated `ocr_span` visual packet.
    public let evidence: MultimodalPacket

    public var visual: MultimodalVisual? {
        guard case .visual(let visual) = evidence.projection else { return nil }
        return visual
    }
}

/// Bounded intake result. Every region span indexes `text`; held pages add no text.
public struct DocumentIntakeReport: Sendable, Equatable, CustomStringConvertible, CustomDebugStringConvertible {
    public let outcome: DocumentIntakeOutcome
    public let reasons: [DocumentIntakeReason]
    /// Extracted protected text; never included in descriptions.
    public let text: String
    public let pages: [DocumentIntakePage]
    public let regions: [DocumentIntakeRegion]

    /// Returns regions whose scalar span overlaps `span`; out-of-range spans match nothing.
    public func regions(overlapping span: Range<Int>) -> [DocumentIntakeRegion] {
        regions.filter { $0.scalarSpan.lowerBound < span.upperBound && span.lowerBound < $0.scalarSpan.upperBound }
    }

    /// Returns the report text for a scalar span, or `nil` when it is out of range.
    public func text(scalarSpan span: Range<Int>) -> String? {
        let scalars = Array(text.unicodeScalars)
        guard span.lowerBound >= 0, span.upperBound <= scalars.count else { return nil }
        return String(String.UnicodeScalarView(scalars[span]))
    }

    public var description: String {
        let codes = reasons.map(\.rawValue).joined(separator: ",")
        return "DocumentIntakeReport(outcome: \(outcome.rawValue), reasons: [\(codes)], pages: \(pages.count), regions: \(regions.count))"
    }

    public var debugDescription: String { description }
}

/// Reference mapping shared with Python and Android through
/// `tests/fixtures/parity/document_intake_v1.json`.
///
/// OCR-image boxes are mapped back by undoing the render scale, the quarter-turn
/// rotation and the crop, in that order. The arithmetic order is part of the
/// contract, so all three implementations produce identical binary64 values.
public enum DocumentIntakeMapper {
    public static let schemaVersion = "openmed.multimodal.document-intake.v1"
    public static let pageSeparator = "\n\n"
    /// Overshoot past the OCR image edge, as a fraction of its size, that is clamped.
    public static let boxTolerance = 0.01
    static let maxCoordinate = 2_147_483_647.0

    /// Opaque 128-bit reference for `"page"`, `"crop"` or `"region:<index>"`.
    public static func nodeRef(sourceDigest: String, page: Int, label: String) -> String {
        let identity = "\(schemaVersion)|\(sourceDigest)|\(page)|\(label)"
        return SHA256.hash(data: Data(identity.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    /// Report for a document that could not be opened.
    public static func refused(_ reason: DocumentIntakeReason) -> DocumentIntakeReport {
        DocumentIntakeReport(outcome: .refused, reasons: [reason], text: "", pages: [], regions: [])
    }

    /// Maps one OCR-image box to original-page coordinates or throws the hold reason.
    public static func mapBox(_ box: DocumentIntakeBox, geometry: DocumentPageGeometry, imageWidth: Int, imageHeight: Int) throws -> DocumentIntakeBox {
        let crop = try checkedCrop(geometry, imageWidth, imageHeight)
        let cropWidth = crop.right - crop.left
        let cropHeight = crop.bottom - crop.top
        let quarter = geometry.rotation == 90 || geometry.rotation == 270
        let uprightWidth = quarter ? cropHeight : cropWidth
        let uprightHeight = quarter ? cropWidth : cropHeight
        let width = Double(imageWidth)
        let height = Double(imageHeight)
        guard [box.left, box.top, box.right, box.bottom].allSatisfy(\.isFinite) else {
            throw DocumentIntakeReason.regionOutOfBounds
        }
        let toleranceX = width * boxTolerance
        let toleranceY = height * boxTolerance
        guard box.left >= -toleranceX, box.top >= -toleranceY, box.right <= width + toleranceX, box.bottom <= height + toleranceY else {
            throw DocumentIntakeReason.regionOutOfBounds
        }
        let x0 = clamp(box.left, 0, width)
        let x1 = clamp(box.right, 0, width)
        let y0 = clamp(box.top, 0, height)
        let y1 = clamp(box.bottom, 0, height)
        guard x0 < x1, y0 < y1 else { throw DocumentIntakeReason.regionOutOfBounds }
        let u0 = clamp(x0 * uprightWidth / width, 0, uprightWidth)
        let u1 = clamp(x1 * uprightWidth / width, 0, uprightWidth)
        let v0 = clamp(y0 * uprightHeight / height, 0, uprightHeight)
        let v1 = clamp(y1 * uprightHeight / height, 0, uprightHeight)
        let local: [Double]
        switch geometry.rotation {
        case 0: local = [u0, v0, u1, v1]
        case 90: local = [v0, cropHeight - u1, v1, cropHeight - u0]
        case 180: local = [cropWidth - u1, cropHeight - v1, cropWidth - u0, cropHeight - v0]
        default: local = [cropWidth - v1, u0, cropWidth - v0, u1]
        }
        let mapped = DocumentIntakeBox(
            left: clamp(crop.left + local[0], crop.left, crop.right) + 0.0,
            top: clamp(crop.top + local[1], crop.top, crop.bottom) + 0.0,
            right: clamp(crop.left + local[2], crop.left, crop.right) + 0.0,
            bottom: clamp(crop.top + local[3], crop.top, crop.bottom) + 0.0)
        guard mapped.left < mapped.right, mapped.top < mapped.bottom else { throw DocumentIntakeReason.ambiguousTransform }
        return mapped
    }

    /// Maps native page inputs to page, crop and region evidence.
    ///
    /// The page index is the position in `pages`. `partial` means at least one page
    /// was mapped and a reason applies; `refused` means no page was mapped. A
    /// malformed source digest refuses the document as unsupported.
    public static func map(
        sourceDigest: String, pages: [DocumentPageInput], limits: DocumentIntakeLimits = DocumentIntakeLimits(),
        pageSeparator: String = pageSeparator
    ) -> DocumentIntakeReport {
        guard sourceDigest.utf8.count == 64, sourceDigest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            return refused(.unsupportedDocument)
        }
        guard !pages.isEmpty else { return refused(.emptyDocument) }
        var reasons = Set<DocumentIntakeReason>()
        if pages.count > limits.maxPages { reasons.insert(.pageLimit) }
        var text = ""
        var length = 0
        var records: [DocumentIntakePage] = []
        var regions: [DocumentIntakeRegion] = []
        pageLoop: for (index, item) in pages.prefix(limits.maxPages).enumerated() {
            let reason: DocumentIntakeReason
            switch item {
            case .unavailable(.pageLimit):
                reasons.insert(.pageLimit)
                break pageLoop
            case .unavailable(let unavailable):
                reason = DocumentIntake.pageReason(unavailable)
            case .rendered(let geometry, let recognition):
                let separator = records.contains { $0.status == .mapped } ? pageSeparator : ""
                let offset = length + separator.unicodeScalars.count
                do {
                    let (page, mapped) = try mapPage(sourceDigest, index, geometry, recognition, limits, offset)
                    text += separator + recognition.text
                    length = offset + recognition.text.unicodeScalars.count
                    records.append(page)
                    regions += mapped
                    continue
                } catch {
                    reason = error as? DocumentIntakeReason ?? .ambiguousTransform
                }
            }
            reasons.insert(reason)
            records.append(DocumentIntakePage(page: index, status: .held, reason: reason, textSpan: nil, evidence: nil, cropEvidence: nil))
            if reason == .permissionLost { break }
        }
        let outcome: DocumentIntakeOutcome
        if !records.contains(where: { $0.status == .mapped }) {
            outcome = .refused
        } else {
            outcome = reasons.isEmpty ? .complete : .partial
        }
        return DocumentIntakeReport(
            outcome: outcome, reasons: reasons.sorted { $0.order < $1.order }, text: text, pages: records, regions: regions)
    }

    private static func clamp(_ value: Double, _ low: Double, _ high: Double) -> Double {
        if value < low { return low }
        if value > high { return high }
        return value
    }

    private static func bounded(_ value: Double) -> Bool {
        value.isFinite && value >= 0 && value <= maxCoordinate
    }

    private static func checkedCrop(_ geometry: DocumentPageGeometry, _ width: Int, _ height: Int) throws -> DocumentIntakeBox {
        guard bounded(geometry.width), bounded(geometry.height), geometry.width > 0, geometry.height > 0,
            [0, 90, 180, 270].contains(geometry.rotation),
            (1...Int(maxCoordinate)).contains(width), (1...Int(maxCoordinate)).contains(height)
        else { throw DocumentIntakeReason.ambiguousTransform }
        let crop = geometry.crop ?? DocumentIntakeBox(left: 0, top: 0, right: geometry.width, bottom: geometry.height)
        guard [crop.left, crop.top, crop.right, crop.bottom].allSatisfy(bounded),
            crop.left < crop.right, crop.right <= geometry.width, crop.top < crop.bottom, crop.bottom <= geometry.height
        else { throw DocumentIntakeReason.ambiguousTransform }
        let cropWidth = crop.right - crop.left
        let cropHeight = crop.bottom - crop.top
        let quarter = geometry.rotation == 90 || geometry.rotation == 270
        let uprightWidth = quarter ? cropHeight : cropWidth
        let uprightHeight = quarter ? cropWidth : cropHeight
        // Rendering rounds each OCR dimension to a whole pixel, so the image may
        // differ from the upright crop's aspect by at most half a pixel per side.
        let skew = abs(Double(width) * uprightHeight - Double(height) * uprightWidth)
        guard skew <= (uprightWidth + uprightHeight) * 0.5 else { throw DocumentIntakeReason.ambiguousTransform }
        return crop
    }

    private static func mapPage(
        _ digest: String, _ index: Int, _ geometry: DocumentPageGeometry, _ recognition: DocumentRecognition,
        _ limits: DocumentIntakeLimits, _ offset: Int
    ) throws -> (DocumentIntakePage, [DocumentIntakeRegion]) {
        let crop = try checkedCrop(geometry, recognition.width, recognition.height)
        let length = recognition.text.unicodeScalars.count
        guard recognition.regions.count <= limits.maxRegionsPerPage, length <= limits.maxPageScalars else {
            throw DocumentIntakeReason.resourceLimit
        }
        var mapped: [(Range<Int>, DocumentIntakeBox)] = []
        for region in recognition.regions {
            guard 0 <= region.scalarStart, region.scalarStart < region.scalarEnd, region.scalarEnd <= length else {
                throw DocumentIntakeReason.invalidRecognition
            }
            let box = try mapBox(region.box, geometry: geometry, imageWidth: recognition.width, imageHeight: recognition.height)
            mapped.append(((offset + region.scalarStart)..<(offset + region.scalarEnd), box))
        }
        func visual(_ label: String, _ kind: MultimodalVisualKind, _ box: DocumentIntakeBox?, _ span: Range<Int>?) throws -> MultimodalPacket {
            do {
                return try MultimodalPacket.visual(
                    nodeRef: nodeRef(sourceDigest: digest, page: index, label: label), sourceDigest: digest, kind: kind,
                    page: Int64(index), width: geometry.width, height: geometry.height, unit: geometry.unit,
                    sourceBox: box.map { MultimodalRectangle(left: $0.left, top: $0.top, right: $0.right, bottom: $0.bottom) },
                    scalarSpan: span.map { MultimodalInterval(start: Int64($0.lowerBound), end: Int64($0.upperBound)) },
                    rotation: Int64(geometry.rotation))
            } catch {
                throw DocumentIntakeReason.ambiguousTransform
            }
        }
        let page = try visual("page", .page, nil, nil)
        let cropPacket = geometry.crop == nil ? nil : try visual("crop", .pixels, crop, nil)
        var regions: [DocumentIntakeRegion] = []
        for (ordinal, (span, box)) in mapped.enumerated() {
            let label = "region:\(ordinal)"
            regions.append(
                DocumentIntakeRegion(
                    page: index, scalarSpan: span, sourceBox: box,
                    nodeRef: nodeRef(sourceDigest: digest, page: index, label: label),
                    evidence: try visual(label, .ocr, box, span)))
        }
        let record = DocumentIntakePage(
            page: index, status: .mapped, reason: nil, textSpan: offset..<(offset + length), evidence: page, cropEvidence: cropPacket)
        return (record, regions)
    }
}

/// Transient OCR input. The intake calls `release()` once recognition ends or is cancelled.
public protocol DocumentIntakeImage: AnyObject, Sendable {
    var pixelWidth: Int { get }
    var pixelHeight: Int { get }
    func release()
}

/// One rendered page: original geometry plus the upright crop for OCR.
public struct DocumentRenderedPage {
    public let geometry: DocumentPageGeometry
    public let image: any DocumentIntakeImage

    public init(geometry: DocumentPageGeometry, image: any DocumentIntakeImage) {
        self.geometry = geometry
        self.image = image
    }
}

/// An opened document. The intake owns it and calls `close()` exactly once.
///
/// `renderPage` throws a `DocumentIntakeReason` for an unreadable page; a source
/// that loses access throws `permissionLost`, after which no page is requested.
public protocol DocumentIntakeDocument: AnyObject {
    /// Lowercase SHA-256 of the source bytes.
    var sourceDigest: String { get }
    var pageCount: Int { get }
    func renderPage(at index: Int, limits: DocumentIntakeLimits) throws -> DocumentRenderedPage
    func close()
}

/// Opens a document from caller-owned storage without copying it elsewhere.
public protocol DocumentIntakeSource {
    /// Throws `permissionDenied`, `documentLocked`, `sizeLimit` or `unsupportedDocument`.
    func open(limits: DocumentIntakeLimits) throws -> any DocumentIntakeDocument
}

/// On-device text recognition over one upright page image, in OCR-image pixels.
public protocol DocumentTextRecognizer: Sendable {
    func recognize(_ image: any DocumentIntakeImage) async throws -> DocumentRecognition
}

/// Library-level PDF and image intake that emits shared evidence coordinates.
///
/// Pages are rendered and recognized one at a time; each transient image is
/// released as soon as its recognition ends, and the document is closed on every
/// exit, including cancellation. Nothing is written to storage.
public struct DocumentIntake: Sendable {
    public let limits: DocumentIntakeLimits
    public let pageSeparator: String
    private let recognizer: any DocumentTextRecognizer

    public init(
        recognizer: any DocumentTextRecognizer, limits: DocumentIntakeLimits = DocumentIntakeLimits(),
        pageSeparator: String = DocumentIntakeMapper.pageSeparator
    ) {
        self.recognizer = recognizer
        self.limits = limits
        self.pageSeparator = pageSeparator
    }

    /// Runs the intake. Throws only `CancellationError`; every other failure is an
    /// explicit `partial` or `refused` outcome with value-free reasons.
    public func run(_ source: any DocumentIntakeSource) async throws -> DocumentIntakeReport {
        try Task.checkCancellation()
        let document: any DocumentIntakeDocument
        do {
            document = try source.open(limits: limits)
        } catch {
            if error is CancellationError { throw error }
            return DocumentIntakeMapper.refused(Self.documentReason(error))
        }
        defer { document.close() }
        var inputs: [DocumentPageInput] = []
        for index in 0..<max(0, document.pageCount) {
            if index == limits.maxPages {
                inputs.append(.unavailable(.pageLimit))
                break
            }
            try Task.checkCancellation()
            let page: DocumentRenderedPage
            do {
                page = try document.renderPage(at: index, limits: limits)
            } catch {
                let reason = Self.pageReason(error)
                inputs.append(.unavailable(reason))
                if reason == .permissionLost { break }
                continue
            }
            inputs.append(try await recognize(page))
        }
        try Task.checkCancellation()
        return DocumentIntakeMapper.map(sourceDigest: document.sourceDigest, pages: inputs, limits: limits, pageSeparator: pageSeparator)
    }

    private func recognize(_ page: DocumentRenderedPage) async throws -> DocumentPageInput {
        defer { page.image.release() }
        let recognition: DocumentRecognition
        do {
            recognition = try await recognizer.recognize(page.image)
        } catch {
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            return .unavailable(.recognitionFailed)
        }
        try Task.checkCancellation()
        guard recognition.width == page.image.pixelWidth, recognition.height == page.image.pixelHeight else {
            return .unavailable(.ambiguousTransform)
        }
        return .rendered(page.geometry, recognition)
    }

    static func documentReason(_ error: Error) -> DocumentIntakeReason {
        switch error as? DocumentIntakeReason {
        case .permissionDenied?, .permissionLost?: return .permissionDenied
        case .documentLocked?: return .documentLocked
        case .sizeLimit?: return .sizeLimit
        default: return .unsupportedDocument
        }
    }

    static func pageReason(_ error: Error) -> DocumentIntakeReason {
        switch error as? DocumentIntakeReason {
        case .permissionLost?, .permissionDenied?: return .permissionLost
        case .ambiguousTransform?: return .ambiguousTransform
        case .resourceLimit?, .sizeLimit?: return .resourceLimit
        case .recognitionFailed?: return .recognitionFailed
        default: return .pageMissing
        }
    }
}
