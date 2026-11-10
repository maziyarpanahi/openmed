#if DEBUG
    import CoreGraphics
    import CoreText
    import Foundation
    import SwiftUI

    /// Synthetic preview data only; never invoked by production review loading.
    @MainActor
    enum MultimodalReviewFixtures {
        struct Row: Decodable {
            let id: String
            let kind: String
            let status: String
            let revision: String
            let uncertainty: [String]
            let wire: String
            let text: String
            let imageText: String?
            let alternatives: [Alternative]
            enum CodingKeys: String, CodingKey {
                case id, kind, status, revision, uncertainty, wire, text, alternatives
                case imageText = "image_text"
            }
        }

        struct Alternative: Decodable {
            let reference: String
            let value: String
        }
        private struct Corpus: Decodable {
            let version: Int
            let syntheticOnly: Bool
            let cases: [Row]
            enum CodingKeys: String, CodingKey {
                case version, cases
                case syntheticOnly = "synthetic_only"
            }
        }

        final class Source: LocalMultimodalReviewSource {
            var localOnly = true
            var revision: String?
            var content: LocalMultimodalReviewContent?
            init(revision: String?, content: LocalMultimodalReviewContent?) {
                self.revision = revision
                self.content = content
            }
            func currentRevision(_ handle: LocalMultimodalReviewHandle) throws -> String? { revision }
            func resolve(_ handle: LocalMultimodalReviewHandle) throws -> LocalMultimodalReviewContent? { content }
        }

        static func rows(data supplied: Data? = nil) throws -> [Row] {
            var root = URL(fileURLWithPath: #filePath)
            for _ in 0..<5 { root.deleteLastPathComponent() }
            let data = try supplied ?? Data(contentsOf: root.appending(path: "tests/fixtures/multimodal/local_review_v1.json"))
            let corpus = try JSONDecoder().decode(Corpus.self, from: data)
            guard corpus.version == 1, corpus.syntheticOnly, corpus.cases.count == 4 else { throw MultimodalReviewError.invalid }
            return corpus.cases
        }

        static func image(_ text: String) throws -> CGImage {
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                let context = CGContext(
                    data: nil, width: 240, height: 96, bitsPerComponent: 8, bytesPerRow: 240 * 4,
                    space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            else { throw MultimodalReviewError.invalid }
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: 240, height: 96))
            let line = CTLineCreateWithAttributedString(
                NSAttributedString(
                    string: text,
                    attributes: [
                        NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica" as CFString, 42, nil),
                        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0, alpha: 1),
                    ]))
            context.textPosition = CGPoint(x: 18, y: 28)
            CTLineDraw(line, context)
            guard let image = context.makeImage() else { throw MultimodalReviewError.invalid }
            return image
        }

        static func make(_ id: String, data: Data? = nil) throws -> (MultimodalReviewCase, Source) {
            guard let row = try rows(data: data).first(where: { $0.id == id }), let kind = MultimodalReviewKind(rawValue: row.kind) else {
                throw MultimodalReviewError.invalid
            }
            let review = try MultimodalReviewCase(
                handle: LocalMultimodalReviewHandle(), evidence: MultimodalPacket.fromJSON(Data(row.wire.utf8)),
                revision: row.revision, kind: kind, alternatives: row.alternatives.map(\.reference),
                uncertainty: row.uncertainty.map { raw in
                    guard let value = MultimodalReviewUncertainty(rawValue: raw) else { throw MultimodalReviewError.invalid }
                    return value
                })
            let content = try LocalMultimodalReviewContent(
                revision: row.revision, sourceDigest: review.sourceDigest, text: row.text,
                image: try row.imageText.map(image),
                alternatives: row.alternatives.map { try LocalReviewAlternative(reference: $0.reference, value: $0.value) })
            return (
                review,
                Source(
                    revision: row.status == "stale" ? String(repeating: "f", count: 64) : row.revision,
                    content: row.status == "missing" ? nil : content)
            )
        }

        static func view(_ id: String) throws -> some View {
            let (review, source) = try make(id)
            return MultimodalReviewView(review: review, source: source) { _ in }
                .frame(width: 420, height: 760)
        }
    }

    #Preview("Ambiguous visual value") { try! MultimodalReviewFixtures.view("ambiguous_visual") }
    #Preview("Stale evidence") { try! MultimodalReviewFixtures.view("stale_evidence") }
    #Preview("Missing content") { try! MultimodalReviewFixtures.view("missing_content") }
    #Preview("Uncertain speaker role") { try! MultimodalReviewFixtures.view("uncertain_speaker") }
#endif
