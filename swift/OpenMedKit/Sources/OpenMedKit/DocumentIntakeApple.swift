#if canImport(CoreGraphics) && canImport(ImageIO)
    import CoreGraphics
    import CryptoKit
    import Foundation
    import ImageIO

    /// A transient, in-memory upright page image. `release()` drops the pixels.
    public final class CGDocumentIntakeImage: DocumentIntakeImage, @unchecked Sendable {
        private let lock = NSLock()
        private var storage: CGImage?
        public let pixelWidth: Int
        public let pixelHeight: Int

        public init(_ image: CGImage) {
            storage = image
            pixelWidth = image.width
            pixelHeight = image.height
        }

        /// The pixels, or `nil` after release.
        public var cgImage: CGImage? { lock.withLock { storage } }

        public var isReleased: Bool { lock.withLock { storage == nil } }

        public func release() { lock.withLock { storage = nil } }
    }

    enum DocumentIntakeRendering {
        /// Reads caller-owned bytes once, in memory or mapped, with value-free failures.
        static func read(_ url: URL, limits: DocumentIntakeLimits) throws -> Data {
            guard FileManager.default.isReadableFile(atPath: url.path) else { throw DocumentIntakeReason.permissionDenied }
            if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size > limits.maxDocumentBytes {
                throw DocumentIntakeReason.sizeLimit
            }
            let data: Data
            do { data = try Data(contentsOf: url, options: .mappedIfSafe) } catch { throw DocumentIntakeReason.permissionDenied }
            return try bounded(data, limits: limits)
        }

        static func bounded(_ data: Data, limits: DocumentIntakeLimits) throws -> Data {
            guard data.count <= limits.maxDocumentBytes else { throw DocumentIntakeReason.sizeLimit }
            guard !data.isEmpty else { throw DocumentIntakeReason.unsupportedDocument }
            return data
        }

        static func digest(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }

        /// Renders the crop of an original page upright, at `scale` pixels per unit.
        ///
        /// `sourceFromUser` maps drawing user space to top-left original-page
        /// coordinates. The pixel grid is exactly the one the shared mapper inverts.
        static func renderUpright(
            _ geometry: DocumentPageGeometry, scale: Double, sourceFromUser: CGAffineTransform, limits: DocumentIntakeLimits,
            draw: (CGContext) -> Void
        ) throws -> CGDocumentIntakeImage {
            let crop = geometry.crop ?? DocumentIntakeBox(left: 0, top: 0, right: geometry.width, bottom: geometry.height)
            guard [0, 90, 180, 270].contains(geometry.rotation), scale.isFinite, scale > 0,
                crop.left >= 0, crop.top >= 0, crop.left < crop.right, crop.top < crop.bottom,
                crop.right <= geometry.width, crop.bottom <= geometry.height
            else { throw DocumentIntakeReason.ambiguousTransform }
            let cropWidth = crop.right - crop.left
            let cropHeight = crop.bottom - crop.top
            let quarter = geometry.rotation == 90 || geometry.rotation == 270
            let uprightWidth = quarter ? cropHeight : cropWidth
            let uprightHeight = quarter ? cropWidth : cropHeight
            let scaledWidth = (uprightWidth * scale).rounded()
            let scaledHeight = (uprightHeight * scale).rounded()
            guard scaledWidth.isFinite, scaledHeight.isFinite, scaledWidth * scaledHeight <= Double(limits.maxRenderPixels) else {
                throw DocumentIntakeReason.resourceLimit
            }
            let width = max(1, Int(scaledWidth))
            let height = max(1, Int(scaledHeight))
            guard
                let context = CGContext(
                    data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { throw DocumentIntakeReason.resourceLimit }
            context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            let local = CGAffineTransform(translationX: -crop.left, y: -crop.top)
            let upright: CGAffineTransform
            switch geometry.rotation {
            case 90: upright = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: cropHeight, ty: 0)
            case 180: upright = CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: cropWidth, ty: cropHeight)
            case 270: upright = CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: cropWidth)
            default: upright = .identity
            }
            let pixels = CGAffineTransform(scaleX: Double(width) / uprightWidth, y: Double(height) / uprightHeight)
            let device = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: Double(height))
            context.concatenate(
                sourceFromUser.concatenating(local).concatenating(upright).concatenating(pixels).concatenating(device))
            context.interpolationQuality = .high
            draw(context)
            guard let image = context.makeImage() else { throw DocumentIntakeReason.resourceLimit }
            return CGDocumentIntakeImage(image)
        }
    }

    /// Image intake through ImageIO. Each frame is a page in stored-pixel coordinates.
    ///
    /// EXIF quarter turns become the page rotation; mirrored orientations cannot
    /// be expressed as a quarter turn and hold the page as `ambiguous_transform`.
    public struct ImageIODocumentIntakeSource: DocumentIntakeSource {
        private let url: URL?
        private let data: Data?
        private let crop: DocumentIntakeBox?

        /// - Parameter crop: Stored-pixel region handed to OCR on every frame.
        public init(url: URL, crop: DocumentIntakeBox? = nil) {
            self.url = url
            self.data = nil
            self.crop = crop
        }

        public init(data: Data, crop: DocumentIntakeBox? = nil) {
            self.url = nil
            self.data = data
            self.crop = crop
        }

        public func open(limits: DocumentIntakeLimits) throws -> any DocumentIntakeDocument {
            let scoped = url?.startAccessingSecurityScopedResource() ?? false
            do {
                let bytes =
                    try url.map { try DocumentIntakeRendering.read($0, limits: limits) }
                    ?? DocumentIntakeRendering.bounded(data ?? Data(), limits: limits)
                guard let source = CGImageSourceCreateWithData(bytes as CFData, nil), CGImageSourceGetType(source) != nil,
                    CGImageSourceGetCount(source) > 0
                else { throw DocumentIntakeReason.unsupportedDocument }
                return ImageIOIntakeDocument(
                    source: source, digest: DocumentIntakeRendering.digest(bytes), crop: crop, scopedURL: scoped ? url : nil)
            } catch {
                if scoped { url?.stopAccessingSecurityScopedResource() }
                throw error
            }
        }
    }

    final class ImageIOIntakeDocument: DocumentIntakeDocument {
        private var source: CGImageSource?
        private var scopedURL: URL?
        private let crop: DocumentIntakeBox?
        let sourceDigest: String
        let pageCount: Int

        init(source: CGImageSource, digest: String, crop: DocumentIntakeBox?, scopedURL: URL?) {
            self.source = source
            self.sourceDigest = digest
            self.crop = crop
            self.scopedURL = scopedURL
            self.pageCount = CGImageSourceGetCount(source)
        }

        var isOpen: Bool { source != nil }

        static func rotation(exifOrientation: UInt32) throws -> Int {
            switch exifOrientation {
            case 1: return 0
            case 3: return 180
            case 6: return 90
            case 8: return 270
            default: throw DocumentIntakeReason.ambiguousTransform
            }
        }

        func renderPage(at index: Int, limits: DocumentIntakeLimits) throws -> DocumentRenderedPage {
            guard let source else { throw DocumentIntakeReason.permissionLost }
            guard let image = CGImageSourceCreateImageAtIndex(source, index, nil) else { throw DocumentIntakeReason.pageMissing }
            let properties = CGImageSourceCopyPropertiesAtIndex(source, index, nil) as? [CFString: Any]
            let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
            let geometry = DocumentPageGeometry(
                width: Double(image.width), height: Double(image.height), unit: .pixel,
                rotation: try Self.rotation(exifOrientation: orientation), crop: crop)
            let flip = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: Double(image.height))
            let rendered = try DocumentIntakeRendering.renderUpright(geometry, scale: 1, sourceFromUser: flip, limits: limits) {
                $0.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            }
            return DocumentRenderedPage(geometry: geometry, image: rendered)
        }

        func close() {
            source = nil
            scopedURL?.stopAccessingSecurityScopedResource()
            scopedURL = nil
        }
    }

    /// Vision lines in normalized, lower-left-origin image space, assembled into text.
    enum DocumentTextLayout {
        struct Word: Equatable {
            let text: String
            /// Normalized `[0, 1]` rectangle with a lower-left origin, as Vision reports it.
            let box: CGRect
        }

        /// Orders lines top to bottom, then left to right, joining words with a space
        /// and lines with a newline. Boxes become top-left OCR-image pixels.
        static func assemble(lines: [[Word]], width: Int, height: Int) -> DocumentRecognition {
            let ordered = lines.enumerated().compactMap { index, line -> (Double, Double, Int, [Word])? in
                let words = line.filter { !$0.text.unicodeScalars.isEmpty }
                guard let first = words.first else { return nil }
                let top = 1 - (words.map { Double($0.box.maxY) }.max() ?? 0)
                return (top, Double(first.box.minX), index, words)
            }
            .sorted { ($0.0, $0.1, $0.2) < ($1.0, $1.1, $1.2) }
            var text = ""
            var length = 0
            var regions: [DocumentRecognizedRegion] = []
            for (lineIndex, line) in ordered.enumerated() {
                if lineIndex > 0 {
                    text += "\n"
                    length += 1
                }
                for (wordIndex, word) in line.3.enumerated() {
                    if wordIndex > 0 {
                        text += " "
                        length += 1
                    }
                    let start = length
                    text += word.text
                    length += word.text.unicodeScalars.count
                    regions.append(DocumentRecognizedRegion(scalarStart: start, scalarEnd: length, box: pixels(word.box, width, height)))
                }
            }
            return DocumentRecognition(width: width, height: height, text: text, regions: regions)
        }

        static func pixels(_ box: CGRect, _ width: Int, _ height: Int) -> DocumentIntakeBox {
            func unit(_ value: CGFloat) -> Double { min(max(Double(value), 0), 1) }
            return DocumentIntakeBox(
                left: unit(box.minX) * Double(width), top: (1 - unit(box.maxY)) * Double(height),
                right: unit(box.maxX) * Double(width), bottom: (1 - unit(box.minY)) * Double(height))
        }
    }
#endif

#if canImport(PDFKit)
    import PDFKit

    /// PDF intake through PDFKit. Pages keep their media box, crop box and `/Rotate`.
    ///
    /// Coordinates are top-left PDF points on the media box; the crop box becomes
    /// the declared crop and `/Rotate` the page rotation. Each page is drawn into
    /// an in-memory bitmap at `renderScale` pixels per point.
    public struct PDFKitDocumentIntakeSource: DocumentIntakeSource {
        private let url: URL?
        private let data: Data?
        private let renderScale: Double

        public init(url: URL, renderScale: Double = 2) {
            self.url = url
            self.data = nil
            self.renderScale = renderScale
        }

        public init(data: Data, renderScale: Double = 2) {
            self.url = nil
            self.data = data
            self.renderScale = renderScale
        }

        public func open(limits: DocumentIntakeLimits) throws -> any DocumentIntakeDocument {
            let scoped = url?.startAccessingSecurityScopedResource() ?? false
            do {
                let bytes =
                    try url.map { try DocumentIntakeRendering.read($0, limits: limits) }
                    ?? DocumentIntakeRendering.bounded(data ?? Data(), limits: limits)
                guard let document = PDFDocument(data: bytes) else { throw DocumentIntakeReason.unsupportedDocument }
                guard !document.isLocked else { throw DocumentIntakeReason.documentLocked }
                return PDFKitIntakeDocument(
                    document: document, digest: DocumentIntakeRendering.digest(bytes), scale: renderScale,
                    scopedURL: scoped ? url : nil)
            } catch {
                if scoped { url?.stopAccessingSecurityScopedResource() }
                throw error
            }
        }
    }

    final class PDFKitIntakeDocument: DocumentIntakeDocument {
        private var document: PDFDocument?
        private var scopedURL: URL?
        private let scale: Double
        let sourceDigest: String
        let pageCount: Int

        init(document: PDFDocument, digest: String, scale: Double, scopedURL: URL?) {
            self.document = document
            self.sourceDigest = digest
            self.scale = scale
            self.scopedURL = scopedURL
            self.pageCount = document.pageCount
        }

        var isOpen: Bool { document != nil }

        /// Top-left geometry from PDF boxes, which use a lower-left origin.
        static func geometry(mediaBox: CGRect, cropBox: CGRect, rotation: Int) throws -> DocumentPageGeometry {
            let media = mediaBox.standardized
            let visible = cropBox.standardized.intersection(media)
            let turn = ((rotation % 360) + 360) % 360
            guard media.width > 0, media.height > 0, !visible.isNull, visible.width > 0, visible.height > 0, turn % 90 == 0 else {
                throw DocumentIntakeReason.ambiguousTransform
            }
            let crop = DocumentIntakeBox(
                left: Double(visible.minX - media.minX), top: Double(media.maxY - visible.maxY),
                right: Double(visible.maxX - media.minX), bottom: Double(media.maxY - visible.minY))
            let whole = DocumentIntakeBox(left: 0, top: 0, right: Double(media.width), bottom: Double(media.height))
            return DocumentPageGeometry(
                width: Double(media.width), height: Double(media.height), unit: .point, rotation: turn,
                crop: crop == whole ? nil : crop)
        }

        func renderPage(at index: Int, limits: DocumentIntakeLimits) throws -> DocumentRenderedPage {
            guard let document else { throw DocumentIntakeReason.permissionLost }
            guard let page = document.page(at: index), let reference = page.pageRef else { throw DocumentIntakeReason.pageMissing }
            let media = page.bounds(for: .mediaBox).standardized
            let geometry = try Self.geometry(mediaBox: media, cropBox: page.bounds(for: .cropBox), rotation: page.rotation)
            let flip = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: -media.minX, ty: media.maxY)
            let image = try DocumentIntakeRendering.renderUpright(geometry, scale: scale, sourceFromUser: flip, limits: limits) {
                $0.drawPDFPage(reference)
            }
            return DocumentRenderedPage(geometry: geometry, image: image)
        }

        func close() {
            document = nil
            scopedURL?.stopAccessingSecurityScopedResource()
            scopedURL = nil
        }
    }
#endif

#if canImport(Vision) && !os(watchOS)
    import Vision

    private final class UncheckedReference<Value>: @unchecked Sendable {
        let value: Value
        init(_ value: Value) { self.value = value }
    }

    /// On-device Vision text recognition with word boxes in OCR-image pixels.
    ///
    /// Language correction is off by default so recognized characters are not
    /// rewritten. Recognition runs locally; Vision makes no network request.
    public struct VisionDocumentTextRecognizer: DocumentTextRecognizer {
        public var recognitionLanguages: [String]
        public var usesLanguageCorrection: Bool

        public init(recognitionLanguages: [String] = [], usesLanguageCorrection: Bool = false) {
            self.recognitionLanguages = recognitionLanguages
            self.usesLanguageCorrection = usesLanguageCorrection
        }

        public func recognize(_ image: any DocumentIntakeImage) async throws -> DocumentRecognition {
            guard let cgImage = (image as? CGDocumentIntakeImage)?.cgImage else { throw DocumentIntakeReason.recognitionFailed }
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = usesLanguageCorrection
            if !recognitionLanguages.isEmpty { request.recognitionLanguages = recognitionLanguages }
            let work = UncheckedReference((VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:]), request))
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    DispatchQueue.global(qos: .userInitiated).async {
                        do {
                            try work.value.0.perform([work.value.1])
                            continuation.resume()
                        } catch {
                            continuation.resume(throwing: DocumentIntakeReason.recognitionFailed)
                        }
                    }
                }
            } onCancel: {
                work.value.1.cancel()
            }
            try Task.checkCancellation()
            let lines = (request.results ?? []).map { observation -> [DocumentTextLayout.Word] in
                guard let candidate = observation.topCandidates(1).first else { return [] }
                let string = candidate.string
                return string.split(whereSeparator: \.isWhitespace).map { word in
                    let range = word.startIndex..<word.endIndex
                    let box = (try? candidate.boundingBox(for: range))?.boundingBox ?? observation.boundingBox
                    return DocumentTextLayout.Word(text: String(word), box: box)
                }
            }
            return DocumentTextLayout.assemble(lines: lines, width: image.pixelWidth, height: image.pixelHeight)
        }
    }
#endif
