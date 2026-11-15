#if canImport(CoreGraphics) && canImport(ImageIO) && canImport(PDFKit)
    import CoreGraphics
    import Foundation
    import PDFKit

    /// Process-wide live export sessions. A new exporter sweeps every other
    /// session directory, which after an app restart is every leftover one.
    final class DocumentRedactionSessionRegistry: @unchecked Sendable {
        static let shared = DocumentRedactionSessionRegistry()
        private let lock = NSLock()
        private var live = Set<String>()

        func register(_ session: String, sweeping root: URL) {
            lock.withLock {
                live.insert(session)
                let entries = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
                for entry in entries where !live.contains(entry.lastPathComponent) {
                    try? FileManager.default.removeItem(at: entry)
                }
            }
        }

        func remove(_ session: String) { _ = lock.withLock { live.remove(session) } }
    }

    /// RGBA8 page rasters with a top-left origin, painted and converted in memory.
    enum DocumentRedactionRaster {
        static func rgba(_ image: CGImage) -> [UInt8]? {
            let width = image.width
            let height = image.height
            var pixels = [UInt8](repeating: 255, count: width * height * 4)
            let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
                guard
                    let context = CGContext(
                        data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
                else { return false }
                context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
                return true
            }
            return drawn ? pixels : nil
        }

        /// Paints `[left, top, right, bottom]` pixel rectangles opaque black.
        static func paint(_ pixels: inout [UInt8], width: Int, rects: [[Int]]) {
            for rect in rects {
                for y in rect[1]..<rect[3] {
                    for x in rect[0]..<rect[2] {
                        let index = (y * width + x) * 4
                        pixels[index] = 0
                        pixels[index + 1] = 0
                        pixels[index + 2] = 0
                        pixels[index + 3] = 255
                    }
                }
            }
        }

        static func rgb(_ pixels: [UInt8]) -> Data {
            let count = pixels.count / 4
            var output = Data(count: count * 3)
            output.withUnsafeMutableBytes { (target: UnsafeMutableRawBufferPointer) in
                pixels.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
                    for pixel in 0..<count {
                        target[pixel * 3] = source[pixel * 4]
                        target[pixel * 3 + 1] = source[pixel * 4 + 1]
                        target[pixel * 3 + 2] = source[pixel * 4 + 2]
                    }
                }
            }
            return output
        }

        /// True when every normalized rectangle, inset by one pixel, is dark.
        static func isOpaque(_ image: CGImage, rects: [DocumentIntakeBox]) -> Bool {
            guard !rects.isEmpty else { return true }
            guard let pixels = rgba(image) else { return false }
            let width = Double(image.width)
            let height = Double(image.height)
            for rect in rects {
                let left = Int((rect.left * width).rounded(.up)) + 1
                let top = Int((rect.top * height).rounded(.up)) + 1
                let right = Int((rect.right * width).rounded(.down)) - 1
                let bottom = Int((rect.bottom * height).rounded(.down)) - 1
                guard left < right, top < bottom, left >= 0, top >= 0, right <= image.width, bottom <= image.height else { continue }
                for y in top..<bottom {
                    for x in left..<right {
                        let index = (y * image.width + x) * 4
                        if pixels[index] > 48 || pixels[index + 1] > 48 || pixels[index + 2] > 48 { return false }
                    }
                }
            }
            return true
        }
    }

    /// Checks painted masks on each re-rendered page, then defers to the caller's recognizer.
    private final class VerifyingRecognizer: DocumentTextRecognizer, @unchecked Sendable {
        private let inner: any DocumentTextRecognizer
        private let expectations: [[DocumentIntakeBox]]
        private let lock = NSLock()
        private var seen = 0
        private var opaque = true

        init(inner: any DocumentTextRecognizer, expectations: [[DocumentIntakeBox]]) {
            self.inner = inner
            self.expectations = expectations
        }

        var allOpaque: Bool { lock.withLock { opaque && seen == expectations.count } }

        func recognize(_ image: any DocumentIntakeImage) async throws -> DocumentRecognition {
            let index = lock.withLock {
                defer { seen += 1 }
                return seen
            }
            let painted =
                expectations.indices.contains(index)
                && (image as? CGDocumentIntakeImage)?.cgImage.map { DocumentRedactionRaster.isOpaque($0, rects: expectations[index]) } == true
            if !painted { lock.withLock { opaque = false } }
            return try await inner.recognize(image)
        }
    }

    /// Exports verified redactions as new image-only PDFs in private temporary storage.
    ///
    /// `export` renders every source page through the same `DocumentIntakeSource`
    /// used for intake, paints each mask opaque, and writes a new PDF whose pages
    /// are only those pixels. Nothing from the original file is copied: no text
    /// layer, annotation, attachment or metadata. The stored file is then read
    /// back and verified before it can be offered: it must match the profile
    /// exactly, PDFKit must find no text, annotations or document attributes,
    /// re-intake through `DocumentIntake` with the caller's recognizer must be
    /// complete with every painted mask opaque, and the re-extracted text must not
    /// contain any masked identifier.
    ///
    /// The exporter never shares anything itself. `shareURL(for:matching:)` is
    /// the explicit caller action that hands the verified file to
    /// `UIActivityViewController`, `ShareLink` or a document exporter; call
    /// `complete(_:)` when that flow ends. Artifacts are deleted when the export
    /// is cancelled or fails, when they are completed or invalidated, when the
    /// exporter is released, and by the next exporter created after a restart.
    public actor DocumentRedactionExporter {
        public static let directoryName = "openmed-redaction-exports"

        public nonisolated let limits: DocumentIntakeLimits
        private let recognizer: any DocumentTextRecognizer
        private let session: URL
        private let sessionName: String
        private var entries: [String: Entry] = [:]

        private struct Entry {
            let url: URL
            let receipt: DocumentRedactionReceipt
        }

        /// - Parameters:
        ///   - recognizer: On-device recognizer used to re-extract the artifact.
        ///   - directory: Private parent directory; defaults to a folder in the
        ///     app's temporary directory. Other sessions in it are removed.
        /// - Throws: `storageUnavailable` when the directory cannot be created.
        public init(recognizer: any DocumentTextRecognizer, limits: DocumentIntakeLimits = DocumentIntakeLimits(), directory: URL? = nil) throws {
            let root = directory ?? FileManager.default.temporaryDirectory.appending(path: Self.directoryName, directoryHint: .isDirectory)
            let name = DocumentRedactionExportContract.randomReference()
            let session = root.appending(path: name, directoryHint: .isDirectory)
            let attributes: [FileAttributeKey: Any] = [.posixPermissions: 0o700]
            do {
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: attributes)
                DocumentRedactionSessionRegistry.shared.register(name, sweeping: root)
                try FileManager.default.createDirectory(at: session, withIntermediateDirectories: false, attributes: attributes)
            } catch {
                DocumentRedactionSessionRegistry.shared.remove(name)
                throw DocumentRedactionExportError.storageUnavailable
            }
            self.recognizer = recognizer
            self.limits = limits
            self.session = session
            self.sessionName = name
        }

        deinit {
            try? FileManager.default.removeItem(at: session)
            DocumentRedactionSessionRegistry.shared.remove(sessionName)
        }

        /// Number of verified artifacts that can still be shared.
        public var artifactCount: Int { entries.count }

        /// Renders, writes and verifies a sanitized artifact for `request`.
        ///
        /// - Parameter source: The source the report was produced from. Its
        ///   digest, page count and page geometry must still match the report.
        /// - Throws: `CancellationError` after removing partial output, or a
        ///   `DocumentRedactionExportError`. A failed export leaves no file.
        public func export(_ request: DocumentRedactionRequest, from source: any DocumentIntakeSource) async throws -> DocumentRedactionArtifact {
            try Task.checkCancellation()
            let plan = try DocumentRedactionExportContract.plan(request)
            let id = DocumentRedactionExportContract.randomReference()
            let pending = session.appending(path: "pending-\(id).pdf")
            let final = session.appending(path: "\(id).pdf")
            do {
                let rendered = try render(plan, request: request, source: source)
                let data = DocumentRedactionPDF.write(rendered.pages)
                guard data.count <= limits.maxDocumentBytes else { throw DocumentRedactionExportError.resourceLimit }
                do {
                    try data.write(to: pending, options: [.withoutOverwriting, .completeFileProtection])
                } catch {
                    throw DocumentRedactionExportError.storageUnavailable
                }
                let stored: Data
                do { stored = try Data(contentsOf: pending) } catch { throw DocumentRedactionExportError.storageUnavailable }
                let receipt = try await verify(stored, request: request, rendered: rendered)
                try Task.checkCancellation()
                do { try FileManager.default.moveItem(at: pending, to: final) } catch { throw DocumentRedactionExportError.storageUnavailable }
                entries[id] = Entry(url: final, receipt: receipt)
                return DocumentRedactionArtifact(id: id, receipt: receipt)
            } catch {
                try? FileManager.default.removeItem(at: pending)
                try? FileManager.default.removeItem(at: final)
                if error is CancellationError || Task.isCancelled { throw CancellationError() }
                throw error as? DocumentRedactionExportError ?? .renderFailed
            }
        }

        /// The explicit share or save action: returns the verified file for the
        /// caller to present. Nothing is shared until the caller presents it.
        ///
        /// - Parameter request: The caller's current redaction state. When its
        ///   binding differs from the artifact's, the artifact is deleted.
        /// - Throws: `artifactUnavailable`, `staleRequest` or `artifactChanged`;
        ///   the original document is never returned in place of the artifact.
        public func shareURL(for artifact: DocumentRedactionArtifact, matching request: DocumentRedactionRequest) throws -> URL {
            guard let entry = entries[artifact.id], entry.receipt == artifact.receipt else {
                throw DocumentRedactionExportError.artifactUnavailable
            }
            guard request.binding == entry.receipt.binding else {
                remove(artifact.id)
                throw DocumentRedactionExportError.staleRequest
            }
            guard let data = try? Data(contentsOf: entry.url), DocumentRedactionExportContract.sha256(data) == entry.receipt.outputDigest else {
                remove(artifact.id)
                throw DocumentRedactionExportError.artifactChanged
            }
            return entry.url
        }

        /// Deletes an artifact after its share or save flow completes or is dismissed.
        public func complete(_ artifact: DocumentRedactionArtifact) { remove(artifact.id) }

        /// Deletes every artifact whose binding differs from `request`, for
        /// example after the source, masks or policy changed.
        public func invalidate(keeping request: DocumentRedactionRequest) {
            for (id, entry) in entries where entry.receipt.binding != request.binding { remove(id) }
        }

        /// Deletes every artifact.
        public func invalidateAll() {
            for id in Array(entries.keys) { remove(id) }
        }

        private func remove(_ id: String) {
            guard let entry = entries.removeValue(forKey: id) else { return }
            try? FileManager.default.removeItem(at: entry.url)
        }

        private struct Rendered {
            var pages: [DocumentRedactionPDF.Page] = []
            /// Painted rectangles per page, normalized to the page image size.
            var expectations: [[DocumentIntakeBox]] = []
            var painted = 0
        }

        private func render(_ plan: [DocumentRedactionPagePlan], request: DocumentRedactionRequest, source: any DocumentIntakeSource) throws -> Rendered {
            let document: any DocumentIntakeDocument
            do { document = try source.open(limits: limits) } catch { throw DocumentRedactionExportError.sourceChanged }
            defer { document.close() }
            guard document.sourceDigest == request.report.sourceDigest, document.pageCount == plan.count else {
                throw DocumentRedactionExportError.sourceChanged
            }
            var rendered = Rendered()
            for page in plan {
                try Task.checkCancellation()
                let input: DocumentRenderedPage
                do {
                    input = try document.renderPage(at: page.page, limits: limits)
                } catch {
                    throw (error as? DocumentIntakeReason) == .resourceLimit
                        ? DocumentRedactionExportError.resourceLimit : DocumentRedactionExportError.renderFailed
                }
                defer { input.image.release() }
                guard input.geometry == page.geometry else { throw DocumentRedactionExportError.sourceChanged }
                guard let image = (input.image as? CGDocumentIntakeImage)?.cgImage, var pixels = DocumentRedactionRaster.rgba(image) else {
                    throw DocumentRedactionExportError.renderFailed
                }
                let width = image.width
                let height = image.height
                let rects = page.boxes.compactMap { box -> [Int]? in
                    guard let mapped = DocumentRedactionExportContract.imageBox(box, geometry: page.geometry, imageWidth: width, imageHeight: height)
                    else { return nil }
                    return DocumentRedactionExportContract.pixelRect(mapped, padding: request.policy.padding, imageWidth: width, imageHeight: height)
                }
                DocumentRedactionRaster.paint(&pixels, width: width, rects: rects)
                let crop = page.geometry.crop ?? DocumentIntakeBox(left: 0, top: 0, right: page.geometry.width, bottom: page.geometry.height)
                let quarter = page.geometry.rotation == 90 || page.geometry.rotation == 270
                let cropWidth = crop.right - crop.left
                let cropHeight = crop.bottom - crop.top
                rendered.pages.append(
                    try DocumentRedactionPDF.page(
                        rgb: DocumentRedactionRaster.rgb(pixels), pixelWidth: width, pixelHeight: height,
                        mediaWidth: quarter ? cropHeight : cropWidth, mediaHeight: quarter ? cropWidth : cropHeight))
                rendered.expectations.append(
                    rects.map {
                        DocumentIntakeBox(
                            left: Double($0[0]) / Double(width), top: Double($0[1]) / Double(height),
                            right: Double($0[2]) / Double(width), bottom: Double($0[3]) / Double(height))
                    })
                rendered.painted += rects.count
            }
            return rendered
        }

        private func verify(_ data: Data, request: DocumentRedactionRequest, rendered: Rendered) async throws -> DocumentRedactionReceipt {
            let pages = try DocumentRedactionPDF.inspect(data, maxPixels: limits.maxRenderPixels)
            guard pages.count == rendered.pages.count else { throw DocumentRedactionExportError.pageCountMismatch }
            guard pages == rendered.pages else { throw DocumentRedactionExportError.profileViolation }
            try Self.inspectWithPDFKit(data, pageCount: pages.count)
            guard let first = pages.first, let media = Double(first.mediaWidth) else { throw DocumentRedactionExportError.profileViolation }
            let verifier = VerifyingRecognizer(inner: recognizer, expectations: rendered.expectations)
            let report = try await DocumentIntake(recognizer: verifier, limits: limits)
                .run(PDFKitDocumentIntakeSource(data: data, renderScale: Double(first.pixelWidth) / media))
            guard report.outcome == .complete else { throw DocumentRedactionExportError.verificationIncomplete }
            guard report.pages.count == pages.count else { throw DocumentRedactionExportError.pageCountMismatch }
            guard verifier.allOpaque else { throw DocumentRedactionExportError.maskNotOpaque }
            guard
                !DocumentRedactionExportContract.containsResidual(
                    masked: DocumentRedactionExportContract.maskedTexts(request), extracted: report.text,
                    minimumScalars: request.policy.minimumResidualTokenScalars)
            else { throw DocumentRedactionExportError.residualText }
            return DocumentRedactionReceipt(
                schemaVersion: DocumentRedactionExportContract.schemaVersion, profile: DocumentRedactionExportContract.profile,
                binding: request.binding, sourceDigest: request.report.sourceDigest ?? "",
                outputDigest: DocumentRedactionExportContract.sha256(data), byteCount: data.count, pageCount: pages.count,
                maskCount: request.masks.count, paintedRegionCount: rendered.painted, checks: DocumentRedactionCheck.allCases)
        }

        /// Independent platform reading: no text layer, annotation, outline or document attributes.
        static func inspectWithPDFKit(_ data: Data, pageCount: Int) throws {
            guard let document = PDFDocument(data: data), !document.isEncrypted, !document.isLocked else {
                throw DocumentRedactionExportError.profileViolation
            }
            guard document.pageCount == pageCount else { throw DocumentRedactionExportError.pageCountMismatch }
            for index in 0..<pageCount {
                guard let page = document.page(at: index) else { throw DocumentRedactionExportError.pageCountMismatch }
                if !(page.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    throw DocumentRedactionExportError.hiddenText
                }
                if !page.annotations.isEmpty { throw DocumentRedactionExportError.attachmentsPresent }
            }
            if document.outlineRoot != nil { throw DocumentRedactionExportError.attachmentsPresent }
            let attributes = document.documentAttributes ?? [:]
            let keys: [PDFDocumentAttribute] = [
                .titleAttribute, .authorAttribute, .subjectAttribute, .creatorAttribute, .producerAttribute, .keywordsAttribute,
            ]
            if keys.contains(where: { attributes[$0] != nil }) { throw DocumentRedactionExportError.metadataPresent }
        }
    }
#endif
