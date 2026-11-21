import Foundation
import SwiftUI

/// Display information supplied by a bundled or host-reviewed offline resource table.
public struct OpenMedCrisisResource: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let label: String
    public let url: URL
    public let sourceURL: URL
    public let verifiedOn: String

    enum CodingKeys: String, CodingKey {
        case id, label, url
        case sourceURL = "source_url"
        case verifiedOn = "verified_on"
    }

    public init(from decoder: Decoder) throws {
        let row = try decoder.container(keyedBy: CodingKeys.self)
        id = try row.decode(String.self, forKey: .id)
        label = try row.decode(String.self, forKey: .label)
        verifiedOn = try row.decode(String.self, forKey: .verifiedOn)
        func link(_ key: CodingKeys) throws -> URL {
            let raw = try row.decode(String.self, forKey: key)
            guard raw.utf8.count <= 2048, raw.utf8.allSatisfy({ (33...126).contains($0) && $0 != 92 }),
                let parts = URLComponents(string: raw), parts.scheme?.lowercased() == "https",
                let host = parts.host, !host.isEmpty, parts.user == nil, parts.password == nil,
                (1...65535).contains(parts.port ?? 443), let value = parts.url
            else {
                throw OpenMedCrisisRoutingError.invalidTable
            }
            return value
        }
        url = try link(.url)
        sourceURL = try link(.sourceURL)
        let date = DateFormatter()
        date.locale = Locale(identifier: "en_US_POSIX")
        date.timeZone = TimeZone(secondsFromGMT: 0)
        date.dateFormat = "yyyy-MM-dd"
        guard id.range(of: #"\A[a-z][a-z0-9_]{0,63}\z"#, options: .regularExpression) != nil,
            !label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, label.count <= 160,
            label.rangeOfCharacter(from: .controlCharacters) == nil,
            let parsed = date.date(from: verifiedOn), date.string(from: parsed) == verifiedOn
        else {
            throw OpenMedCrisisRoutingError.invalidTable
        }
    }
}

public enum OpenMedCrisisRoutingError: Error, Equatable {
    case invalidTable
    case noticeRequired
}

public struct OpenMedCrisisNotice: Sendable {
    public let text = "If someone is in immediate danger, contact local emergency services now."
    public let resources: [OpenMedCrisisResource]
    public let matchedLocale: String?
}

/// Resolve an explicit signal locally. An override replaces only its declared locales.
public struct OpenMedCrisisRouting: Sendable {
    private let entries: [String: [OpenMedCrisisResource]]

    private struct Table: Decodable {
        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case locales
        }
        let schemaVersion: Int
        let locales: [String: [OpenMedCrisisResource]]
    }

    public init(overrideData: Data? = nil) throws {
        guard let url = Bundle.module.url(forResource: "crisis_resources", withExtension: "json") else {
            throw OpenMedCrisisRoutingError.invalidTable
        }
        do {
            var entries = try Self.table(Data(contentsOf: url))
            if let overrideData {
                entries.merge(try Self.table(overrideData)) { _, new in new }
            }
            self.entries = entries
        } catch {
            throw OpenMedCrisisRoutingError.invalidTable
        }
    }

    private static func table(_ data: Data) throws -> [String: [OpenMedCrisisResource]] {
        let table = try JSONDecoder().decode(Table.self, from: data)
        guard table.schemaVersion == 1 else { throw OpenMedCrisisRoutingError.invalidTable }
        var entries = [String: [OpenMedCrisisResource]]()
        for (locale, rows) in table.locales {
            guard let key = keys(locale).first, entries[key] == nil, !rows.isEmpty,
                Set(rows.map(\.id)).count == rows.count
            else { throw OpenMedCrisisRoutingError.invalidTable }
            entries[key] = rows
        }
        return entries
    }

    private static func keys(_ locale: String) -> [String] {
        let tag = locale.replacingOccurrences(of: "_", with: "-").lowercased()
        guard tag.range(of: #"\A[a-z]{2,3}(?:-[a-z]{4})?(?:-[a-z]{2}|-[0-9]{3})?\z"#, options: .regularExpression) != nil else { return [] }
        let parts = tag.components(separatedBy: "-")
        let language = parts[0]
        var script: String?
        var region: String?
        for part in parts.dropFirst() {
            if part.count == 4 { script = part.capitalized } else { region = part.uppercased() }
        }
        var result = [[language, script, region].compactMap { $0 }.joined(separator: "-")]
        if let script, let region {
            result += [language + "-" + region, language + "-" + script]
        }
        if result.last != language { result.append(language) }
        return result
    }

    /// Required and unknown capabilities reject suppression. Missing locales still yield guidance.
    public func route(capability: String, locale: String, signal: Bool, suppressNotice: Bool = false) throws -> OpenMedCrisisNotice? {
        guard signal else { return nil }
        if suppressNotice {
            guard let declaration = try? OpenMedIntendedUse.lookup(capability),
                !declaration.regulatoryTier.obligations.contains("crisis_routing_required")
            else {
                throw OpenMedCrisisRoutingError.noticeRequired
            }
            return nil
        }
        for key in Self.keys(locale) {
            if let rows = entries[key] { return OpenMedCrisisNotice(resources: rows, matchedLocale: key) }
        }
        return OpenMedCrisisNotice(resources: [], matchedLocale: nil)
    }
}

/// Present the notice before results. Resource links open only after a user taps them.
public struct OpenMedCrisisBanner: View {
    public let notice: OpenMedCrisisNotice
    public init(notice: OpenMedCrisisNotice) { self.notice = notice }
    public var body: some View {
        VStack(alignment: .leading) {
            Text(verbatim: notice.text)
            ForEach(notice.resources) { resource in
                Link(destination: resource.url) { Text(verbatim: resource.label) }
            }
        }
        .accessibilityElement(children: .contain)
    }
}
