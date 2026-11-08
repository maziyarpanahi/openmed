import CryptoKit
import Foundation

/// Closed, value-free export failures. Messages are the raw codes only.
public enum DocumentRedactionExportError: String, Error, LocalizedError, Sendable, CaseIterable {
    /// The intake report is not `complete`, so some page was never examined.
    case incompleteReport = "incomplete_report"
    /// A mask span or box lies outside the report text or its page.
    case maskOutOfRange = "mask_out_of_range"
    /// A mask span covers text that no recognized region places on a page.
    case maskUnplaced = "mask_unplaced"
    /// The source no longer has the digest, page count or geometry of the report.
    case sourceChanged = "source_changed"
    /// A page could not be rendered into pixels.
    case renderFailed = "render_failed"
    /// The sanitized artifact exceeds the byte or render-pixel limit.
    case resourceLimit = "resource_limit"
    /// Private export storage could not be created, written or read.
    case storageUnavailable = "storage_unavailable"
    /// Re-extraction of the artifact did not complete on every page.
    case verificationIncomplete = "verification_incomplete"
    /// The artifact does not have one page per source page.
    case pageCountMismatch = "page_count_mismatch"
    /// A painted mask is not opaque in the rendered artifact.
    case maskNotOpaque = "mask_not_opaque"
    /// The artifact carries a text layer or font resources.
    case hiddenText = "hidden_text"
    /// The artifact carries annotations, attachments, forms or actions.
    case attachmentsPresent = "attachments_present"
    /// The artifact carries document metadata.
    case metadataPresent = "metadata_present"
    /// The artifact is not exactly the image-only PDF profile.
    case profileViolation = "profile_violation"
    /// Text extracted from the artifact still contains a masked identifier.
    case residualText = "residual_text"
    /// The source, masks or policy changed since the artifact was exported.
    case staleRequest = "stale_request"
    /// The artifact was completed, invalidated or never exported here.
    case artifactUnavailable = "artifact_unavailable"
    /// The stored artifact bytes no longer match the verified digest.
    case artifactChanged = "artifact_changed"

    public var errorDescription: String? { rawValue }
}

/// Verification checks every exported artifact passed, in receipt order.
public enum DocumentRedactionCheck: String, Sendable, CaseIterable {
    case profile
    case hiddenText = "hidden_text"
    case attachments
    case metadata
    case pageCount = "page_count"
    case visiblePixels = "visible_pixels"
    case residualText = "residual_text"
}

/// A region to remove from the exported pages.
public enum DocumentRedactionMask: Sendable, Equatable {
    /// Unicode scalar offsets into the report text. Every recognized region the
    /// span overlaps is painted over, and every non-whitespace scalar in it must
    /// belong to a recognized region.
    case span(Range<Int>)
    /// A box on a mapped page, in original-page coordinates with a top-left
    /// origin and the page unit, for content OCR cannot see, such as a signature.
    case box(page: Int, box: DocumentIntakeBox)
}

/// Export settings bound into every artifact; changing them invalidates it.
public struct DocumentRedactionPolicy: Sendable, Equatable {
    public static let defaultPolicyID = "openmed.redaction-export.default"

    /// Caller policy identity, such as a de-identification profile name. Only
    /// its digest enters the binding; it never appears in a receipt.
    public var policyID: String
    /// OCR-image pixels added on every side of a painted mask.
    public var padding: Double
    /// Masked tokens shorter than this many scalars are not searched for in the
    /// re-extracted text.
    public var minimumResidualTokenScalars: Int

    public init(policyID: String = defaultPolicyID, padding: Double = 2, minimumResidualTokenScalars: Int = 2) {
        self.policyID = policyID
        self.padding = padding.isFinite ? max(0, padding) : 0
        self.minimumResidualTokenScalars = max(1, minimumResidualTokenScalars)
    }
}

/// A verified redaction result: the intake report, the applied masks and the policy.
///
/// `binding` is the SHA-256 that ties an artifact to exactly this source, text,
/// mask set and policy. Mask order and duplicates do not change it.
public struct DocumentRedactionRequest: Sendable, Equatable, CustomStringConvertible {
    public let report: DocumentIntakeReport
    public let masks: [DocumentRedactionMask]
    public let policy: DocumentRedactionPolicy
    public let binding: String

    public init(report: DocumentIntakeReport, masks: [DocumentRedactionMask], policy: DocumentRedactionPolicy = DocumentRedactionPolicy()) {
        self.report = report
        self.masks = masks
        self.policy = policy
        self.binding = DocumentRedactionExportContract.binding(report: report, masks: masks, policy: policy)
    }

    /// Masks every de-identified entity that was not kept.
    ///
    /// - Throws: `staleRequest` when the result was computed on different text.
    public init(
        report: DocumentIntakeReport, deidentification: DeidentificationResult,
        policy: DocumentRedactionPolicy = DocumentRedactionPolicy(), additionalMasks: [DocumentRedactionMask] = []
    ) throws {
        guard deidentification.originalText == report.text else { throw DocumentRedactionExportError.staleRequest }
        let spans = deidentification.piiEntities
            .filter { $0.action != PolicyAction.keep.rawValue }
            .map { DocumentRedactionMask.span($0.start..<max($0.start, $0.end)) }
        self.init(report: report, masks: spans + additionalMasks, policy: policy)
    }

    public var description: String { "DocumentRedactionRequest(masks: \(masks.count), binding: \(binding))" }
}

/// Value-free evidence of one verified export.
public struct DocumentRedactionReceipt: Sendable, Equatable, CustomStringConvertible {
    public let schemaVersion: String
    public let profile: String
    public let binding: String
    public let sourceDigest: String
    public let outputDigest: String
    public let byteCount: Int
    public let pageCount: Int
    public let maskCount: Int
    /// Rectangles painted across all pages after mapping masks to pixels.
    public let paintedRegionCount: Int
    public let checks: [DocumentRedactionCheck]

    public var description: String {
        "DocumentRedactionReceipt(profile: \(profile), pages: \(pageCount), masks: \(maskCount), "
            + "painted: \(paintedRegionCount), output: \(outputDigest))"
    }
}

/// A verified artifact in private storage. It is shared only through an explicit
/// exporter action, never automatically, and it exposes no file location itself.
public struct DocumentRedactionArtifact: Sendable, Equatable, CustomStringConvertible {
    /// Random 128-bit reference, also the artifact file name.
    public let id: String
    public let receipt: DocumentRedactionReceipt

    public var description: String { "DocumentRedactionArtifact(id: \(id), \(receipt))" }
}

extension DocumentIntakeReport {
    /// SHA-256 of the source bytes, or `nil` for a refused report.
    public var sourceDigest: String? {
        for page in pages {
            if case .visual(let visual)? = page.evidence?.projection { return visual.sourceDigest }
        }
        return nil
    }
}

/// One mapped page and the original-page boxes to paint on it.
struct DocumentRedactionPagePlan: Equatable {
    let page: Int
    let geometry: DocumentPageGeometry
    let boxes: [DocumentIntakeBox]
}

/// Mask placement, pixel mapping, binding and residual search shared with Android.
public enum DocumentRedactionExportContract {
    public static let schemaVersion = "openmed.redaction-export.v1"
    public static let profile = "openmed.redaction-export.image-pdf.v1"

    /// SHA-256 over the source digest, text digest, page count, policy and the
    /// sorted, de-duplicated masks.
    public static func binding(report: DocumentIntakeReport, masks: [DocumentRedactionMask], policy: DocumentRedactionPolicy) -> String {
        var lines = Set<String>()
        for mask in masks {
            switch mask {
            case .span(let span): lines.insert("span:\(span.lowerBound):\(span.upperBound)")
            case .box(let page, let box):
                lines.insert("box:\(page):" + [box.left, box.top, box.right, box.bottom].map(bits).joined(separator: ":"))
            }
        }
        let header = [
            schemaVersion,
            "source:\(report.sourceDigest ?? "")",
            "text:\(sha256(Data(report.text.utf8)))",
            "pages:\(report.pages.count)",
            "policy:\(sha256(Data(policy.policyID.utf8))):\(bits(policy.padding)):\(policy.minimumResidualTokenScalars)",
        ]
        return sha256(Data((header + lines.sorted()).joined(separator: "\n").utf8))
    }

    /// Maps an original-page box into OCR-image pixels: the crop, then the
    /// quarter-turn rotation, then the render scale. This is the forward
    /// transform that `DocumentIntakeMapper.mapBox` inverts. Returns `nil` when
    /// the box lies outside the crop, which is not part of the exported image.
    public static func imageBox(_ box: DocumentIntakeBox, geometry: DocumentPageGeometry, imageWidth: Int, imageHeight: Int) -> DocumentIntakeBox? {
        let crop = geometry.crop ?? DocumentIntakeBox(left: 0, top: 0, right: geometry.width, bottom: geometry.height)
        let left = max(box.left, crop.left) - crop.left
        let top = max(box.top, crop.top) - crop.top
        let right = min(box.right, crop.right) - crop.left
        let bottom = min(box.bottom, crop.bottom) - crop.top
        guard left < right, top < bottom, imageWidth > 0, imageHeight > 0 else { return nil }
        let cropWidth = crop.right - crop.left
        let cropHeight = crop.bottom - crop.top
        let quarter = geometry.rotation == 90 || geometry.rotation == 270
        let upright: [Double]
        switch geometry.rotation {
        case 0: upright = [left, top, right, bottom]
        case 90: upright = [cropHeight - bottom, left, cropHeight - top, right]
        case 180: upright = [cropWidth - right, cropHeight - bottom, cropWidth - left, cropHeight - top]
        case 270: upright = [top, cropWidth - right, bottom, cropWidth - left]
        default: return nil
        }
        let scaleX = Double(imageWidth) / (quarter ? cropHeight : cropWidth)
        let scaleY = Double(imageHeight) / (quarter ? cropWidth : cropHeight)
        return DocumentIntakeBox(left: upright[0] * scaleX, top: upright[1] * scaleY, right: upright[2] * scaleX, bottom: upright[3] * scaleY)
    }

    /// Whole-pixel rectangle covering `box` plus `padding`, rounded outward and
    /// clamped to the image, as `[left, top, right, bottom]`.
    public static func pixelRect(_ box: DocumentIntakeBox, padding: Double, imageWidth: Int, imageHeight: Int) -> [Int]? {
        let values = [box.left - padding, box.top - padding, box.right + padding, box.bottom + padding]
        guard values.allSatisfy(\.isFinite) else { return nil }
        let left = Int(max(0, min(Double(imageWidth), values[0].rounded(.down))))
        let top = Int(max(0, min(Double(imageHeight), values[1].rounded(.down))))
        let right = Int(max(0, min(Double(imageWidth), values[2].rounded(.up))))
        let bottom = Int(max(0, min(Double(imageHeight), values[3].rounded(.up))))
        guard left < right, top < bottom else { return nil }
        return [left, top, right, bottom]
    }

    /// Normalized search tokens: NFKC, lowercased, maximal runs of letters, marks and numbers.
    public static func tokens(_ text: String) -> [String] {
        var result: [String] = []
        var current = String.UnicodeScalarView()
        for scalar in text.precomposedStringWithCompatibilityMapping.lowercased().unicodeScalars {
            if isTokenScalar(scalar) {
                current.append(scalar)
            } else if !current.isEmpty {
                result.append(String(current))
                current = String.UnicodeScalarView()
            }
        }
        if !current.isEmpty { result.append(String(current)) }
        return result
    }

    /// True when a masked token of at least `minimumScalars` scalars is an
    /// extracted token, or, for scripts written without spaces, occurs inside one.
    public static func containsResidual(masked: [String], extracted: String, minimumScalars: Int) -> Bool {
        let found = tokens(extracted)
        let exact = Set(found)
        for text in masked {
            for token in tokens(text) where token.unicodeScalars.count >= max(1, minimumScalars) {
                if exact.contains(token) { return true }
                let needle = Array(token.unicodeScalars)
                if needle.contains(where: isUnspacedScript), found.contains(where: { contains(Array($0.unicodeScalars), needle) }) {
                    return true
                }
            }
        }
        return false
    }

    static func plan(_ request: DocumentRedactionRequest) throws -> [DocumentRedactionPagePlan] {
        let report = request.report
        guard report.outcome == .complete, report.reasons.isEmpty, !report.pages.isEmpty, report.sourceDigest != nil,
            report.pages.enumerated().allSatisfy({ $0.offset == $0.element.page && $0.element.status == .mapped })
        else { throw DocumentRedactionExportError.incompleteReport }
        let geometries = try report.pages.map(geometry)
        var boxes = Array(repeating: [DocumentIntakeBox](), count: geometries.count)
        func add(_ page: Int, _ box: DocumentIntakeBox) {
            if !boxes[page].contains(box) { boxes[page].append(box) }
        }
        let scalars = Array(report.text.unicodeScalars)
        for mask in request.masks {
            switch mask {
            case .span(let span):
                guard span.lowerBound >= 0, span.lowerBound < span.upperBound, span.upperBound <= scalars.count else {
                    throw DocumentRedactionExportError.maskOutOfRange
                }
                let regions = report.regions(overlapping: span)
                for index in span where !scalars[index].properties.isWhitespace {
                    guard regions.contains(where: { $0.scalarSpan.contains(index) }) else {
                        throw DocumentRedactionExportError.maskUnplaced
                    }
                }
                for region in regions { add(region.page, region.sourceBox) }
            case .box(let page, let box):
                guard geometries.indices.contains(page), [box.left, box.top, box.right, box.bottom].allSatisfy(\.isFinite),
                    box.left >= 0, box.top >= 0, box.left < box.right, box.top < box.bottom,
                    box.right <= geometries[page].width, box.bottom <= geometries[page].height
                else { throw DocumentRedactionExportError.maskOutOfRange }
                add(page, box)
            }
        }
        return geometries.enumerated().map { DocumentRedactionPagePlan(page: $0.offset, geometry: $0.element, boxes: boxes[$0.offset]) }
    }

    /// Masked text searched for in the re-extracted artifact.
    static func maskedTexts(_ request: DocumentRedactionRequest) -> [String] {
        request.masks.compactMap {
            guard case .span(let span) = $0 else { return nil }
            return request.report.text(scalarSpan: span)
        }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func randomReference() -> String {
        var generator = SystemRandomNumberGenerator()
        return (0..<16).map { _ in String(format: "%02x", UInt8.random(in: 0...255, using: &generator)) }.joined()
    }

    private static func geometry(_ page: DocumentIntakePage) throws -> DocumentPageGeometry {
        guard case .visual(let visual)? = page.evidence?.projection else { throw DocumentRedactionExportError.incompleteReport }
        var crop: DocumentIntakeBox?
        if let packet = page.cropEvidence {
            guard case .visual(let cropVisual) = packet.projection, let box = cropVisual.sourceBox else {
                throw DocumentRedactionExportError.incompleteReport
            }
            crop = DocumentIntakeBox(left: box.left, top: box.top, right: box.right, bottom: box.bottom)
        }
        return DocumentPageGeometry(width: visual.width, height: visual.height, unit: visual.unit, rotation: Int(visual.rotation), crop: crop)
    }

    private static func bits(_ value: Double) -> String { String(format: "%016llx", (value + 0.0).bitPattern) }

    private static func isTokenScalar(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter, .nonspacingMark, .spacingMark,
            .enclosingMark, .decimalNumber, .letterNumber, .otherNumber:
            return true
        default:
            return false
        }
    }

    private static func isUnspacedScript(_ scalar: Unicode.Scalar) -> Bool {
        let value = scalar.value
        return scalar.properties.isIdeographic || (0x3040...0x30FF).contains(value) || (0x0E00...0x0EFF).contains(value)
            || (0x1000...0x109F).contains(value) || (0x1780...0x17FF).contains(value)
    }

    private static func contains(_ haystack: [Unicode.Scalar], _ needle: [Unicode.Scalar]) -> Bool {
        guard !needle.isEmpty, haystack.count >= needle.count else { return false }
        return (0...(haystack.count - needle.count)).contains { start in
            needle.indices.allSatisfy { haystack[start + $0] == needle[$0] }
        }
    }
}

/// Writer and strict reader for the image-only PDF export profile.
///
/// Every page is one Flate-compressed 8-bit RGB image drawn over the whole
/// media box by a fixed content stream. There is no text, font, annotation,
/// attachment, form, action, outline, document information or XMP metadata.
/// The reader accepts exactly the bytes this writer produces and nothing else.
enum DocumentRedactionPDF {
    struct Page: Equatable {
        let pixelWidth: Int
        let pixelHeight: Int
        let mediaWidth: String
        let mediaHeight: String
        let stream: Data
    }

    private static let header = Data("%PDF-1.4\n%".utf8) + Data([0xE2, 0xE3, 0xCF, 0xD3, 0x0A])
    /// Largest PDF page side, in default user-space units.
    private static let maxMediaSide = 14_400.0

    /// Compresses one upright RGB page. `mediaWidth` and `mediaHeight` are the
    /// physical page size in points.
    static func page(rgb: Data, pixelWidth: Int, pixelHeight: Int, mediaWidth: Double, mediaHeight: Double) throws -> Page {
        guard pixelWidth > 0, pixelHeight > 0, rgb.count == pixelWidth * pixelHeight * 3, mediaWidth.isFinite, mediaHeight.isFinite,
            mediaWidth > 0, mediaHeight > 0
        else { throw DocumentRedactionExportError.renderFailed }
        let factor = min(1, maxMediaSide / max(mediaWidth, mediaHeight))
        return Page(
            pixelWidth: pixelWidth, pixelHeight: pixelHeight, mediaWidth: number(mediaWidth * factor),
            mediaHeight: number(mediaHeight * factor), stream: try zlib(rgb))
    }

    static func write(_ pages: [Page]) -> Data {
        var data = header
        var offsets: [Int] = [0]
        func object(_ body: String, stream: Data? = nil) {
            offsets.append(data.count)
            data.append(Data("\(offsets.count - 1) 0 obj\n\(body)\n".utf8))
            if let stream {
                data.append(Data("stream\n".utf8))
                data.append(stream)
                data.append(Data("\nendstream\n".utf8))
            }
            data.append(Data("endobj\n".utf8))
        }
        object("<< /Type /Catalog /Pages 2 0 R >>")
        let kids = pages.indices.map { "\(3 + 3 * $0) 0 R" }.joined(separator: " ")
        object("<< /Type /Pages /Kids [\(kids)] /Count \(pages.count) >>")
        for (index, page) in pages.enumerated() {
            let base = 3 + 3 * index
            let content = Data(contentStream(page).utf8)
            object(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 \(page.mediaWidth) \(page.mediaHeight)] "
                    + "/Resources << /XObject << /Im0 \(base + 1) 0 R >> >> /Contents \(base + 2) 0 R >>")
            object(
                "<< /Type /XObject /Subtype /Image /Width \(page.pixelWidth) /Height \(page.pixelHeight) "
                    + "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length \(page.stream.count) >>",
                stream: page.stream)
            object("<< /Length \(content.count) >>", stream: content)
        }
        let xref = data.count
        var table = "xref\n0 \(offsets.count)\n0000000000 65535 f \n"
        for offset in offsets.dropFirst() { table += String(format: "%010d 00000 n \n", offset) }
        table += "trailer\n<< /Size \(offsets.count) /Root 1 0 R >>\nstartxref\n\(xref)\n%%EOF\n"
        data.append(Data(table.utf8))
        return data
    }

    /// Re-reads an artifact, accepting only the exact profile layout.
    ///
    /// - Throws: `hiddenText`, `attachmentsPresent` or `metadataPresent` when a
    ///   rejected file carries those structures, otherwise `profileViolation`.
    static func inspect(_ data: Data, maxPixels: Int) throws -> [Page] {
        do {
            return try read(data, maxPixels: maxPixels)
        } catch {
            throw classify(data)
        }
    }

    private struct Malformed: Error {}

    private static func read(_ data: Data, maxPixels: Int) throws -> [Page] {
        var reader = Reader(bytes: [UInt8](data))
        var offsets: [Int] = [0]
        try reader.expect(header)
        offsets.append(reader.position)
        try reader.expect("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        offsets.append(reader.position)
        try reader.expect("2 0 obj\n<< /Type /Pages /Kids [")
        var kids: [Int] = []
        repeat {
            if !kids.isEmpty { try reader.expect(" ") }
            kids.append(try reader.integer())
            try reader.expect(" 0 R")
        } while reader.peek() == UInt8(ascii: " ")
        try reader.expect("] /Count ")
        let count = try reader.integer()
        try reader.expect(" >>\nendobj\n")
        guard count >= 1, kids == (0..<count).map({ 3 + 3 * $0 }) else { throw Malformed() }
        var pages: [Page] = []
        for index in 0..<count {
            let base = 3 + 3 * index
            offsets.append(reader.position)
            try reader.expect("\(base) 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ")
            let mediaWidth = try reader.number()
            try reader.expect(" ")
            let mediaHeight = try reader.number()
            try reader.expect("] /Resources << /XObject << /Im0 \(base + 1) 0 R >> >> /Contents \(base + 2) 0 R >>\nendobj\n")
            offsets.append(reader.position)
            try reader.expect("\(base + 1) 0 obj\n<< /Type /XObject /Subtype /Image /Width ")
            let width = try reader.integer()
            try reader.expect(" /Height ")
            let height = try reader.integer()
            try reader.expect(" /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length ")
            let length = try reader.integer()
            let stream = try reader.stream(length: length)
            guard width >= 1, height >= 1, width <= maxPixels / height else { throw Malformed() }
            let page = Page(pixelWidth: width, pixelHeight: height, mediaWidth: mediaWidth, mediaHeight: mediaHeight, stream: Data(stream))
            offsets.append(reader.position)
            try reader.expect("\(base + 2) 0 obj\n<< /Length ")
            let contentLength = try reader.integer()
            let content = try reader.stream(length: contentLength)
            guard content.elementsEqual(contentStream(page).utf8) else { throw Malformed() }
            pages.append(page)
        }
        let xref = reader.position
        try reader.expect("xref\n0 \(offsets.count)\n0000000000 65535 f \n")
        for offset in offsets.dropFirst() { try reader.expect(String(format: "%010d 00000 n \n", offset)) }
        try reader.expect("trailer\n<< /Size \(offsets.count) /Root 1 0 R >>\nstartxref\n\(xref)\n%%EOF\n")
        guard reader.atEnd else { throw Malformed() }
        return pages
    }

    private static func classify(_ data: Data) -> DocumentRedactionExportError {
        let markers: [(DocumentRedactionExportError, [String])] = [
            (.hiddenText, ["/Font", "/ToUnicode", "/ActualText"]),
            (
                .attachmentsPresent,
                ["/Annots", "/EmbeddedFile", "/FileAttachment", "/AcroForm", "/JavaScript", "/OpenAction", "/AA ", "/Launch", "/URI"]
            ),
            (.metadataPresent, ["/Info", "/Metadata", "/Title", "/Author", "/Producer", "/Creator", "/CreationDate", "/ModDate"]),
        ]
        for (error, needles) in markers where needles.contains(where: { data.range(of: Data($0.utf8)) != nil }) {
            return error
        }
        return .profileViolation
    }

    private static func contentStream(_ page: Page) -> String {
        "q \(page.mediaWidth) 0 0 \(page.mediaHeight) 0 0 cm /Im0 Do Q"
    }

    /// A positive decimal with at most four fractional digits and no exponent.
    static func number(_ value: Double) -> String {
        var text = String(format: "%.4f", value)
        while text.hasSuffix("0") { text.removeLast() }
        if text.hasSuffix(".") { text.removeLast() }
        return text
    }

    /// RFC 1950 zlib stream around Foundation's raw DEFLATE output.
    static func zlib(_ data: Data) throws -> Data {
        let deflated: Data
        do { deflated = try (data as NSData).compressed(using: .zlib) as Data } catch { throw DocumentRedactionExportError.renderFailed }
        var a: UInt32 = 1
        var b: UInt32 = 0
        data.withUnsafeBytes { buffer in
            var index = 0
            while index < buffer.count {
                let end = min(index + 5552, buffer.count)
                while index < end {
                    a += UInt32(buffer[index])
                    b += a
                    index += 1
                }
                a %= 65521
                b %= 65521
            }
        }
        let adler = (b << 16) | a
        return Data([0x78, 0x9C]) + deflated + Data([UInt8(adler >> 24), UInt8((adler >> 16) & 0xFF), UInt8((adler >> 8) & 0xFF), UInt8(adler & 0xFF)])
    }

    private struct Reader {
        let bytes: [UInt8]
        var position = 0

        var atEnd: Bool { position == bytes.count }

        func peek() -> UInt8? { position < bytes.count ? bytes[position] : nil }

        mutating func expect(_ text: String) throws { try expect(Data(text.utf8)) }

        mutating func expect(_ data: Data) throws {
            guard position + data.count <= bytes.count, bytes[position..<(position + data.count)].elementsEqual(data) else { throw Malformed() }
            position += data.count
        }

        /// A non-negative decimal integer of at most nine digits, without leading zeros.
        mutating func integer() throws -> Int {
            let start = position
            while let byte = peek(), (48...57).contains(byte), position - start < 9 { position += 1 }
            guard position > start, bytes[start] != 48 || position == start + 1, let byte = peek(), !(48...57).contains(byte) else {
                throw Malformed()
            }
            return Int(String(decoding: bytes[start..<position], as: UTF8.self)) ?? 0
        }

        /// A positive decimal with at most four fractional digits, as written.
        mutating func number() throws -> String {
            let start = position
            _ = try integer()
            if peek() == UInt8(ascii: ".") {
                position += 1
                let fraction = position
                while let byte = peek(), (48...57).contains(byte), position - fraction < 4 { position += 1 }
                guard position > fraction, bytes[position - 1] != 48 else { throw Malformed() }
            }
            let text = String(decoding: bytes[start..<position], as: UTF8.self)
            guard let value = Double(text), value > 0 else { throw Malformed() }
            return text
        }

        mutating func stream(length: Int) throws -> ArraySlice<UInt8> {
            try expect(" >>\nstream\n")
            guard length <= bytes.count - position else { throw Malformed() }
            let body = bytes[position..<(position + length)]
            position += length
            try expect("\nendstream\nendobj\n")
            return body
        }
    }
}
