import CryptoKit
import Foundation

/// Controlled aECG rejection codes shared with the Python reader.
public struct OpenMedAECGError: Error, CustomStringConvertible, Equatable {
    public let reasonCode: String
    public var description: String { reasonCode }
    fileprivate init(_ code: String) { reasonCode = code }
}

/// Resource bounds applied before complete waveform decoding.
public struct OpenMedAECGLimits: Sendable {
    public let maxBytes: Int
    public let maxElements: Int
    public let maxDepth: Int
    public let maxSeries: Int
    public let maxLeadsPerSeries: Int
    public let maxSamplesPerLead: Int
    public let maxTotalSamples: Int

    public init(maxBytes: Int = 16 * 1024 * 1024, maxElements: Int = 100_000, maxDepth: Int = 64, maxSeries: Int = 16, maxLeadsPerSeries: Int = 64, maxSamplesPerLead: Int = 1_000_000, maxTotalSamples: Int = 2_000_000) {
        self.maxBytes = maxBytes
        self.maxElements = maxElements
        self.maxDepth = maxDepth
        self.maxSeries = maxSeries
        self.maxLeadsPerSeries = maxLeadsPerSeries
        self.maxSamplesPerLead = maxSamplesPerLead
        self.maxTotalSamples = maxTotalSamples
    }
}

/// An original controlled MDC code with decoded samples in the source unit.
public struct OpenMedAECGLead: Sendable {
    public let code: String
    public let unit: String
    public let samples: [Double]
}

/// Rhythm or derived representative-beat signals with relative timing only.
public struct OpenMedAECGSeries: Sendable {
    public let kind: String
    public let sampleIntervalSeconds: Double
    public let leads: [OpenMedAECGLead]
}

/// Reader-local signals without demographics, annotation text or absolute times.
public struct OpenMedAECGRecord: Sendable {
    public let rhythm: [OpenMedAECGSeries]
    public let derived: [OpenMedAECGSeries]
    public let sourceDigest: String
    public let demographicElementCount: Int
    public let annotationCount: Int
    public var demographicsPresent: Bool { demographicElementCount > 0 }
    public var annotationsPresent: Bool { annotationCount > 0 }
    public var notice: String { OpenMedAECG.notice }

    /// Require explicit reviewer confirmation before consequential use.
    public func confirmReview(reviewerConfirmed: Bool) throws -> Self {
        guard reviewerConfirmed else { throw OpenMedAECGError("review_required") }
        return self
    }

    /// Return controlled metadata without samples, identifiers or absolute times.
    public func report() -> [String: Any] {
        [
            "source_digest": sourceDigest,
            "demographics_present": demographicsPresent,
            "demographic_element_count": demographicElementCount,
            "annotations_present": annotationsPresent,
            "annotation_count": annotationCount,
            "notice": notice,
            "review_required": true,
            "series": (rhythm + derived).map {
                ["kind": $0.kind, "lead_count": $0.leads.count, "sample_count": $0.leads[0].samples.count] as [String: Any]
            },
        ]
    }
}

/// Offline, entity-safe reading of a deliberately bounded HL7 aECG subset.
public enum OpenMedAECG {
    public static let notice = "Non-diagnostic waveform for review only. Explicit reviewer confirmation is required before consequential use. No autonomous clinical decisions."
    private static let hl7 = "{urn:hl7-org:v3}"
    private static let xsiType = "{http://www.w3.org/2001/XMLSchema-instance}type"
    private static let actSystem = "2.16.840.1.113883.5.4"
    private static let leadSystem = "2.16.840.1.113883.6.24"
    private static let leadCodes = Set(["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"].map { "MDC_ECG_LEAD_" + $0 })
    private static let demographics = Set(["subject", "trialSubject", "subjectAssignment", "subjectDemographicPerson"].map { hl7 + $0 })

    /// Read caller-supplied UTF-8 XML data without file, network or model access.
    /// - Parameters:
    ///   - data: In-memory XML; paths, archives and other encodings are unsupported.
    ///   - limits: Positive resource bounds, with defaults matching Python.
    /// - Returns: Separate rhythm and derived signals with controlled reports.
    /// - Throws: `OpenMedAECGError` with a payload-free reason code.
    public static func read(_ data: Data, limits: OpenMedAECGLimits = .init()) throws -> OpenMedAECGRecord {
        guard [limits.maxBytes, limits.maxElements, limits.maxDepth, limits.maxSeries, limits.maxLeadsPerSeries, limits.maxSamplesPerLead, limits.maxTotalSamples].allSatisfy({ $0 > 0 }) else { throw OpenMedAECGError("invalid_limits") }
        guard data.count <= limits.maxBytes else { throw OpenMedAECGError("byte_limit") }
        guard let text = String(data: data, encoding: .utf8), !text.contains("\0") else { throw OpenMedAECGError("unsupported_encoding") }
        if let declaration = text.range(of: "^\\uFEFF?<\\?xml\\s+[^?]*\\bencoding\\s*=\\s*['\"][^'\"]+['\"]", options: [.regularExpression, .caseInsensitive]) {
            let header = String(text[declaration])
            guard header.range(of: "encoding\\s*=\\s*['\"](?:utf-8|utf8)['\"]", options: [.regularExpression, .caseInsensitive]) != nil else { throw OpenMedAECGError("unsupported_encoding") }
        }
        guard text.range(of: "<!\\s*(?:DOCTYPE|ENTITY)\\b", options: [.regularExpression, .caseInsensitive]) == nil else { throw OpenMedAECGError("unsafe_xml") }
        let builder = Tree(limits)
        let parser = XMLParser(data: data)
        parser.shouldResolveExternalEntities = false
        parser.externalEntityResolvingPolicy = .never
        parser.delegate = builder
        let parsed = parser.parse()
        if let failure = builder.failure { throw failure }
        guard parsed, let root = builder.root else { throw OpenMedAECGError("malformed_xml") }
        guard root.name == hl7 + "AnnotatedECG" else { throw OpenMedAECGError("unsupported_layout") }
        let rhythmNodes = root.path(["component", "series"])
        let derivedNodes = rhythmNodes.flatMap { $0.path(["derivation", "derivedSeries"]) }
        let allNodes = rhythmNodes + derivedNodes
        let elements = builder.elements
        guard !rhythmNodes.isEmpty, elements.filter({ $0.name == hl7 + "series" || $0.name == hl7 + "derivedSeries" }).count == allNodes.count else { throw OpenMedAECGError("unsupported_layout") }
        let sets = allNodes.flatMap { $0.path(["component", "sequenceSet"]) }
        let sequences = sets.flatMap { $0.path(["component", "sequence"]) }
        guard elements.filter({ $0.name == hl7 + "sequenceSet" }).count == sets.count, elements.filter({ $0.name == hl7 + "sequence" }).count == sequences.count else { throw OpenMedAECGError("unsupported_layout") }
        var rhythm: [OpenMedAECGSeries] = []
        var derived: [OpenMedAECGSeries] = []
        var remaining = limits.maxTotalSamples
        for node in allNodes {
            let isRhythm = node.name == hl7 + "series"
            let code = try node.one("code")
            guard code.attributes["codeSystem"] == actSystem, code.attributes["code"] == (isRhythm ? "RHYTHM" : "REPRESENTATIVE_BEAT") else { throw OpenMedAECGError("unsupported_layout") }
            let result = try decode(node, kind: isRhythm ? "rhythm" : "derived_representative_beat", limits: limits, remaining: remaining)
            remaining -= result.leads.reduce(0) { $0 + $1.samples.count }
            if isRhythm { rhythm.append(result) } else { derived.append(result) }
        }
        return OpenMedAECGRecord(rhythm: rhythm, derived: derived, sourceDigest: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(), demographicElementCount: elements.filter { demographics.contains($0.name) }.count, annotationCount: elements.filter { $0.name == hl7 + "annotation" || $0.name == hl7 + "annotationSet" }.count)
    }

    private static func number(_ text: String?) throws -> Double {
        guard let text, text.utf8.count <= 64, text.range(of: "^[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?$", options: .regularExpression) != nil, let value = Double(text), value.isFinite else { throw OpenMedAECGError("invalid_numeric") }
        return value
    }

    private static func leaf(_ node: Node, attributes: Set<String>) throws {
        guard node.children.isEmpty, Set(node.attributes.keys).isSubset(of: attributes) else { throw OpenMedAECGError("unsupported_encoding") }
    }

    private static func decode(_ node: Node, kind: String, limits: OpenMedAECGLimits, remaining: Int) throws -> OpenMedAECGSeries {
        let sets = node.path(["component", "sequenceSet"])
        guard sets.count == 1 else { throw OpenMedAECGError("unsupported_layout") }
        guard sets[0].children.allSatisfy({ $0.name == hl7 + "component" && $0.children.count == 1 && $0.children[0].name == hl7 + "sequence" }) else { throw OpenMedAECGError("unsupported_layout") }
        let sequences = sets[0].path(["component", "sequence"])
        guard sequences.count >= 2 else { throw OpenMedAECGError("unsupported_layout") }
        guard sequences.count - 1 <= limits.maxLeadsPerSeries else { throw OpenMedAECGError("lead_limit") }
        var leads: [OpenMedAECGLead] = []
        var interval: Double?
        var used = 0
        for sequence in sequences {
            guard sequence.children.count == 2 else { throw OpenMedAECGError("unsupported_layout") }
            let code = try sequence.one("code")
            let value = try sequence.one("value")
            guard Set(value.attributes.keys) == Set([xsiType]) else { throw OpenMedAECGError("unsupported_encoding") }
            let name = code.attributes["code"] ?? ""
            if name == "TIME_ABSOLUTE" || name == "TIME_RELATIVE" {
                guard interval == nil, code.attributes["codeSystem"] == actSystem else { throw OpenMedAECGError("unsupported_layout") }
                guard value.attributes[xsiType] == (name == "TIME_ABSOLUTE" ? "GLIST_TS" : "GLIST_PQ") else { throw OpenMedAECGError("unsupported_encoding") }
                guard Set(value.children.map(\.name)) == Set([hl7 + "head", hl7 + "increment"]) else { throw OpenMedAECGError("unsupported_layout") }
                let head = try value.one("head")
                try leaf(head, attributes: ["value", "unit"])
                guard head.attributes["value"] != nil else { throw OpenMedAECGError("unsupported_layout") }
                // Withhold the absolute acquisition timestamp.
                let increment = try value.one("increment")
                try leaf(increment, attributes: ["value", "unit"])
                guard increment.attributes["unit"] == "s" else { throw OpenMedAECGError("unsupported_encoding") }
                let amount = try number(increment.attributes["value"])
                guard amount > 0 else { throw OpenMedAECGError("invalid_numeric") }
                interval = amount
                continue
            }
            guard leadCodes.contains(name), code.attributes["codeSystem"] == leadSystem else { throw OpenMedAECGError("unsupported_lead") }
            guard !leads.contains(where: { $0.code == name }) else { throw OpenMedAECGError("unsupported_layout") }
            guard value.attributes[xsiType] == "SLIST_PQ" else { throw OpenMedAECGError("unsupported_encoding") }
            guard Set(value.children.map(\.name)) == Set([hl7 + "origin", hl7 + "scale", hl7 + "digits"]) else { throw OpenMedAECGError("unsupported_layout") }
            let originNode = try value.one("origin")
            let scaleNode = try value.one("scale")
            try leaf(originNode, attributes: ["value", "unit"])
            try leaf(scaleNode, attributes: ["value", "unit"])
            guard let unit = originNode.attributes["unit"], ["uV", "mV", "V"].contains(unit), scaleNode.attributes["unit"] == unit else { throw OpenMedAECGError("unsupported_encoding") }
            let origin = try number(originNode.attributes["value"])
            let scale = try number(scaleNode.attributes["value"])
            guard scale != 0 else { throw OpenMedAECGError("invalid_numeric") }
            let digits = try value.one("digits")
            try leaf(digits, attributes: [])
            var samples: [Double] = []
            // Lazy token traversal avoids allocating all digits before enforcing bounds.
            var tokenStart: String.Index?
            let whitespace: Set<Character> = [" ", "\t", "\r", "\n", "\u{000B}", "\u{000C}"]
            func appendToken(_ end: String.Index) throws {
                guard let start = tokenStart else { return }
                guard samples.count < limits.maxSamplesPerLead, used < remaining else { throw OpenMedAECGError("sample_limit") }
                let token = String(digits.text[start..<end])
                guard token.utf8.count <= 11, token.range(of: "^[+-]?[0-9]{1,10}$", options: .regularExpression) != nil, let digit = Int32(token) else { throw OpenMedAECGError("invalid_numeric") }
                let sample = origin + scale * Double(digit)
                guard sample.isFinite else { throw OpenMedAECGError("invalid_numeric") }
                samples.append(sample)
                used += 1
                tokenStart = nil
            }
            for index in digits.text.indices {
                if whitespace.contains(digits.text[index]) { try appendToken(index) } else if tokenStart == nil { tokenStart = index }
            }
            try appendToken(digits.text.endIndex)
            guard !samples.isEmpty else { throw OpenMedAECGError("unsupported_layout") }
            guard leads.isEmpty || samples.count == leads[0].samples.count else { throw OpenMedAECGError("inconsistent_lengths") }
            leads.append(OpenMedAECGLead(code: name, unit: unit, samples: samples))
        }
        guard let interval, !leads.isEmpty else { throw OpenMedAECGError("unsupported_layout") }
        return OpenMedAECGSeries(kind: kind, sampleIntervalSeconds: interval, leads: leads)
    }

    private final class Node {
        let name: String
        let attributes: [String: String]
        var children: [Node] = []
        var text = ""
        init(_ name: String, _ attributes: [String: String]) {
            self.name = name
            self.attributes = attributes
        }
        func path(_ names: [String]) -> [Node] {
            names.reduce([self]) { nodes, name in nodes.flatMap { $0.children.filter { $0.name == hl7 + name } } }
        }
        func one(_ name: String) throws -> Node {
            let matches = path([name])
            guard matches.count == 1 else { throw OpenMedAECGError("unsupported_layout") }
            return matches[0]
        }
    }

    private final class Tree: NSObject, XMLParserDelegate {
        let limits: OpenMedAECGLimits
        var elements: [Node] = []
        var stack: [Node] = []
        var namespaces: [[String: String]] = []
        var seriesCount = 0
        var root: Node?
        var failure: OpenMedAECGError?
        init(_ limits: OpenMedAECGLimits) { self.limits = limits }
        func reject(_ parser: XMLParser, _ code: String) {
            if failure == nil { failure = OpenMedAECGError(code) }
            parser.abortParsing()
        }
        func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?, qualifiedName: String?, attributes attributeDict: [String: String]) {
            guard failure == nil else { return }
            guard elements.count < limits.maxElements else {
                reject(parser, "element_limit")
                return
            }
            guard stack.count < limits.maxDepth else {
                reject(parser, "depth_limit")
                return
            }
            var scope = namespaces.last ?? ["xml": "http://www.w3.org/XML/1998/namespace"]
            for (key, value) in attributeDict {
                if key == "xmlns" { scope[""] = value } else if key.hasPrefix("xmlns:") { scope[String(key.dropFirst(6))] = value }
            }
            func resolve(_ name: String, attribute: Bool = false) -> String {
                let parts = name.split(separator: ":", maxSplits: 1).map(String.init)
                let prefix = parts.count == 2 ? parts[0] : ""
                let uri = attribute && parts.count == 1 ? "" : (scope[prefix] ?? "")
                return uri.isEmpty ? name : "{" + uri + "}" + (parts.last ?? name)
            }
            let name = resolve(elementName)
            if name == hl7 + "series" || name == hl7 + "derivedSeries" {
                seriesCount += 1
                guard seriesCount <= limits.maxSeries else {
                    reject(parser, "series_limit")
                    return
                }
            }
            guard name != hl7 + "reference", name != "{http://www.w3.org/2001/XInclude}include" else {
                reject(parser, "external_reference")
                return
            }
            var attributes: [String: String] = [:]
            for (key, value) in attributeDict where key != "xmlns" && !key.hasPrefix("xmlns:") {
                let resolved = resolve(key, attribute: true)
                guard attributes[resolved] == nil else {
                    reject(parser, "malformed_xml")
                    return
                }
                attributes[resolved] = value
            }
            let node = Node(name, attributes)
            if let parent = stack.last { parent.children.append(node) } else { root = node }
            elements.append(node)
            stack.append(node)
            namespaces.append(scope)
        }
        func parser(_ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?, qualifiedName: String?) {
            if !stack.isEmpty {
                stack.removeLast()
                namespaces.removeLast()
            }
        }
        func parser(_ parser: XMLParser, foundCharacters string: String) { stack.last?.text += string }
        func parser(_ parser: XMLParser, foundCDATA cdataBlock: Data) {
            guard let text = String(data: cdataBlock, encoding: .utf8) else {
                reject(parser, "malformed_xml")
                return
            }
            stack.last?.text += text
        }
        func parser(_ parser: XMLParser, resolveExternalEntityName name: String, systemID: String?) -> Data? {
            reject(parser, "external_reference")
            return nil
        }
    }
}
