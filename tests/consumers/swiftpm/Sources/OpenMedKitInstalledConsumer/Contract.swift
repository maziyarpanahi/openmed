import Foundation

/// Synthetic contract shared by every installed-package consumer.
///
/// The verifier copies `installed_consumer_contract_v1.json` into the staged
/// consumer as `consumer_contract.json`; the consumer never reads repository
/// paths.
struct ConsumerContract: Decodable {
    struct OffsetCase: Decodable {
        let id: String
        let text: String
        let start: Int
        let end: Int
        let utf16Start: Int
        let utf16End: Int
        let scalarLength: Int
        let utf16Length: Int
        let replacement: String
        let expectedRedacted: String

        enum CodingKeys: String, CodingKey {
            case id
            case text
            case start
            case end
            case utf16Start = "utf16_start"
            case utf16End = "utf16_end"
            case scalarLength = "scalar_length"
            case utf16Length = "utf16_length"
            case replacement
            case expectedRedacted = "expected_redacted"
        }
    }

    struct PolicyCase: Decodable {
        let profile: String
        let labelActions: [String: String]

        enum CodingKeys: String, CodingKey {
            case profile
            case labelActions = "label_actions"
        }
    }

    struct Entity: Decodable {
        let label: String
        let start: Int
        let end: Int
        let confidence: Float
    }

    struct RedactionCase: Decodable {
        let id: String
        let profile: String
        let text: String
        let entities: [Entity]
        let expectedRedacted: String

        enum CodingKeys: String, CodingKey {
            case id
            case profile
            case text
            case entities
            case expectedRedacted = "expected_redacted"
        }
    }

    struct FixtureFile: Decodable {
        let path: String
        let content: String
    }

    struct ModelFixture: Decodable {
        let repoID: String
        let revision: String
        let mlxLegacyFiles: [FixtureFile]
        let unsafeManifestConfigPath: String

        enum CodingKeys: String, CodingKey {
            case repoID = "repo_id"
            case revision
            case mlxLegacyFiles = "mlx_legacy_files"
            case unsafeManifestConfigPath = "unsafe_manifest_config_path"
        }
    }

    let schemaVersion: Int
    let synthetic: Bool
    let offsetUnit: String
    let offsetCases: [OffsetCase]
    let policyCases: [PolicyCase]
    let profileAliases: [String: String]
    let unknownProfile: String
    let redactionCases: [RedactionCase]
    let modelFixture: ModelFixture

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case synthetic
        case offsetUnit = "offset_unit"
        case offsetCases = "offset_cases"
        case policyCases = "policy_cases"
        case profileAliases = "profile_aliases"
        case unknownProfile = "unknown_profile"
        case redactionCases = "redaction_cases"
        case modelFixture = "model_fixture"
    }

    static func load() throws -> ConsumerContract {
        let path =
            ProcessInfo.processInfo.environment["OPENMED_CONSUMER_CONTRACT"]
            ?? "consumer_contract.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode(ConsumerContract.self, from: data)
    }
}
