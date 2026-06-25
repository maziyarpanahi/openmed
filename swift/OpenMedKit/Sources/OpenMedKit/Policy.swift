import Foundation

/// A de-identification action from a bundled OpenMed policy profile.
public enum PolicyAction: String, Codable, Equatable, Sendable {
    case keep
    case redact
    case replace
    case mask
    case remove
    case hash

    var redactionEquivalent: PolicyAction {
        switch self {
        case .redact:
            return .mask
        case .keep, .replace, .mask, .remove, .hash:
            return self
        }
    }
}

/// Errors raised while loading bundled policy profiles.
public enum PolicyError: Error, Equatable, LocalizedError, Sendable {
    case blankName
    case unknownProfile(String, allowed: [String])
    case missingResource(String)
    case invalidProfile(String)

    public var errorDescription: String? {
        switch self {
        case .blankName:
            return "policy name must not be blank"
        case .unknownProfile(let name, let allowed):
            return "unknown policy profile '\(name)'; expected one of: \(allowed.joined(separator: ", "))"
        case .missingResource(let name):
            return "bundled policy profile '\(name)' was not found"
        case .invalidProfile(let reason):
            return "invalid bundled policy profile: \(reason)"
        }
    }
}

/// A bundled, versioned OpenMed de-identification policy profile.
///
/// Profiles are sourced from the Python package's `openmed/core/policies`
/// definitions and loaded from SwiftPM resources. The profile exposes
/// per-canonical-label actions plus policy posture and arbitration metadata.
public struct Policy: Equatable, Sendable {
    public static let defaultName = "hipaa_safe_harbor"

    public static let bundledProfileNames = [
        "hipaa_safe_harbor",
        "hipaa_expert_review_assist",
        "gdpr_pseudonymization",
        "research_limited_dataset",
        "strict_no_leak",
        "clinical_minimal_redaction",
        "canada_pipeda",
    ]

    public static let aliases = [
        "gdpr": "gdpr_pseudonymization",
        "pipeda": "canada_pipeda",
    ]

    public let schemaVersion: Int
    public let name: String
    public let posture: String
    public let thresholdProfile: String
    public let defaultAction: PolicyAction
    public let defaultActionBias: String
    public let arbitrationMode: String
    public let strictNoLeak: Bool
    public let safetySweepMandatory: Bool
    public let keepMapping: Bool
    public let reversibleID: Bool
    public let forcedCascadeTiers: [String]
    public let policyLabelActions: [String: PolicyAction]
    public let actions: [String: PolicyAction]

    /// Load a bundled policy profile by canonical name or alias.
    public init(named name: String) throws {
        let canonicalName = try Self.canonicalProfileName(name)
        guard let resourceURL = Self.resourceURL(for: canonicalName) else {
            throw PolicyError.missingResource(canonicalName)
        }

        let payload: PolicyPayload
        do {
            let data = try Data(contentsOf: resourceURL)
            payload = try JSONDecoder.openMedPolicy.decode(PolicyPayload.self, from: data)
        } catch let error as PolicyError {
            throw error
        } catch {
            throw PolicyError.invalidProfile(error.localizedDescription)
        }

        try self.init(payload: payload)
    }

    init(
        name: String,
        schemaVersion: Int = 1,
        posture: String = "test",
        thresholdProfile: String = "balanced",
        defaultAction: PolicyAction,
        defaultActionBias: String? = nil,
        arbitrationMode: String = "balanced",
        strictNoLeak: Bool = false,
        safetySweepMandatory: Bool = false,
        keepMapping: Bool = false,
        reversibleID: Bool = false,
        forcedCascadeTiers: [String] = [],
        policyLabelActions: [String: PolicyAction] = [:],
        actions: [String: PolicyAction] = [:]
    ) {
        self.schemaVersion = schemaVersion
        self.name = name
        self.posture = posture
        self.thresholdProfile = thresholdProfile
        self.defaultAction = defaultAction
        self.defaultActionBias = defaultActionBias ?? defaultAction.rawValue
        self.arbitrationMode = arbitrationMode
        self.strictNoLeak = strictNoLeak
        self.safetySweepMandatory = safetySweepMandatory
        self.keepMapping = keepMapping
        self.reversibleID = reversibleID
        self.forcedCascadeTiers = forcedCascadeTiers
        self.policyLabelActions = policyLabelActions
        self.actions = actions
    }

    /// Resolve the action configured for a source or canonical entity label.
    public func action(for label: String) -> PolicyAction {
        let canonicalLabel = Self.canonicalLabel(for: label)
        let action =
            actions[canonicalLabel]
            ?? policyLabelActions[Self.policyLabel(for: canonicalLabel)]
            ?? defaultAction
        if strictNoLeak && action == .keep {
            return .mask
        }
        return action
    }

    /// Resolve an entity label to the canonical OpenMed policy taxonomy.
    public static func canonicalLabel(for label: String) -> String {
        let key = labelKey(label)
        if let alias = labelAliases[key] {
            return alias
        }

        let upper =
            label
            .uppercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: " ", with: "_")
            .filter { $0.isLetter || $0.isNumber || $0 == "_" }
        if canonicalLabels.contains(upper) {
            return upper
        }
        return "OTHER"
    }

    public static func canonicalProfileName(_ name: String) throws -> String {
        let normalized =
            name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        guard !normalized.isEmpty else {
            throw PolicyError.blankName
        }

        let canonical = aliases[normalized] ?? normalized
        guard bundledProfileNames.contains(canonical) else {
            throw PolicyError.unknownProfile(name, allowed: allowedProfileNames)
        }
        return canonical
    }

    private init(payload: PolicyPayload) throws {
        guard payload.schemaVersion == 1 else {
            throw PolicyError.invalidProfile(
                "schema_version \(payload.schemaVersion) is not supported"
            )
        }

        let canonicalName = try Self.canonicalProfileName(payload.name)
        self.init(
            name: canonicalName,
            schemaVersion: payload.schemaVersion,
            posture: payload.posture,
            thresholdProfile: payload.thresholdProfile,
            defaultAction: payload.defaultAction,
            defaultActionBias: payload.defaultActionBias,
            arbitrationMode: payload.arbitrationMode,
            strictNoLeak: payload.strictNoLeak,
            safetySweepMandatory: payload.safetySweepMandatory,
            keepMapping: payload.keepMapping,
            reversibleID: payload.reversibleID,
            forcedCascadeTiers: payload.forcedCascadeTiers,
            policyLabelActions: payload.policyLabelActions,
            actions: payload.actions
        )
    }

    private static var allowedProfileNames: [String] {
        (bundledProfileNames + aliases.keys).sorted()
    }

    private static func resourceURL(for canonicalName: String) -> URL? {
        Bundle.module.url(
            forResource: canonicalName,
            withExtension: "json",
            subdirectory: "policies"
        )
            ?? Bundle.module.url(
                forResource: canonicalName,
                withExtension: "json",
                subdirectory: "Resources/policies"
            )
            ?? Bundle.module.url(forResource: canonicalName, withExtension: "json")
    }

    private static func labelKey(_ label: String) -> String {
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        let withoutBIOES: String
        if trimmed.count > 2,
            let first = trimmed.first,
            ["B", "I", "E", "S"].contains(String(first)),
            trimmed[trimmed.index(after: trimmed.startIndex)] == "-"
        {
            withoutBIOES = String(trimmed.dropFirst(2))
        } else {
            withoutBIOES = trimmed
        }

        return
            withoutBIOES
            .lowercased()
            .filter { $0.isLetter || $0.isNumber }
    }

    private static func policyLabel(for canonicalLabel: String) -> String {
        if clinicalConceptLabels.contains(canonicalLabel) {
            return "CLINICAL_CONCEPT"
        }
        if quasiIdentifierLabels.contains(canonicalLabel) {
            return "QUASI_IDENTIFIER"
        }
        return "DIRECT_IDENTIFIER"
    }

    private static let clinicalConceptLabels: Set<String> = [
        "MICROORGANISM",
        "ANTIBIOTIC",
        "SUSCEPTIBILITY",
        "OTHER",
    ]

    private static let quasiIdentifierLabels: Set<String> = [
        "LOCATION",
        "ZIPCODE",
        "ORDINAL_DIRECTION",
        "DATE",
        "TIME",
        "AGE",
        "CREDIT_CARD_ISSUER",
        "AMOUNT",
        "CURRENCY",
        "GENDER",
        "EYE_COLOR",
        "HEIGHT",
        "ORGANIZATION",
        "JOB_TITLE",
        "JOB_DEPARTMENT",
        "OCCUPATION",
    ]

    private static let canonicalLabels: Set<String> = [
        "PERSON",
        "FIRST_NAME",
        "LAST_NAME",
        "MIDDLE_NAME",
        "PREFIX",
        "USERNAME",
        "EMAIL",
        "PHONE",
        "URL",
        "LOCATION",
        "STREET_ADDRESS",
        "BUILDING_NUMBER",
        "ZIPCODE",
        "GPS_COORDINATES",
        "ORDINAL_DIRECTION",
        "DATE",
        "DATE_OF_BIRTH",
        "TIME",
        "AGE",
        "ID_NUM",
        "SSN",
        "ACCOUNT_NUMBER",
        "PASSWORD",
        "PIN",
        "API_KEY",
        "CREDIT_CARD",
        "CREDIT_CARD_ISSUER",
        "CVV",
        "IBAN",
        "BIC",
        "AMOUNT",
        "CURRENCY",
        "BITCOIN_ADDRESS",
        "ETHEREUM_ADDRESS",
        "LITECOIN_ADDRESS",
        "MASKED_NUMBER",
        "GENDER",
        "EYE_COLOR",
        "HEIGHT",
        "ORGANIZATION",
        "JOB_TITLE",
        "JOB_DEPARTMENT",
        "OCCUPATION",
        "IP_ADDRESS",
        "MAC_ADDRESS",
        "USER_AGENT",
        "VIN",
        "VEHICLE_REGISTRATION",
        "IMEI",
        "MICROORGANISM",
        "ANTIBIOTIC",
        "SUSCEPTIBILITY",
        "OTHER",
    ]

    private static let labelAliases: [String: String] = [
        "name": "PERSON",
        "person": "PERSON",
        "patient": "PERSON",
        "doctor": "PERSON",
        "fullname": "PERSON",
        "firstname": "FIRST_NAME",
        "givenname": "FIRST_NAME",
        "lastname": "LAST_NAME",
        "surname": "LAST_NAME",
        "familyname": "LAST_NAME",
        "middlename": "MIDDLE_NAME",
        "prefix": "PREFIX",
        "title": "PREFIX",
        "username": "USERNAME",
        "userhandle": "USERNAME",
        "email": "EMAIL",
        "emailaddress": "EMAIL",
        "phone": "PHONE",
        "phonenumber": "PHONE",
        "telephone": "PHONE",
        "fax": "PHONE",
        "url": "URL",
        "urlpersonal": "URL",
        "website": "URL",
        "personalurl": "URL",
        "location": "LOCATION",
        "city": "LOCATION",
        "state": "LOCATION",
        "country": "LOCATION",
        "county": "LOCATION",
        "region": "LOCATION",
        "place": "LOCATION",
        "address": "STREET_ADDRESS",
        "street": "STREET_ADDRESS",
        "streetaddress": "STREET_ADDRESS",
        "secondaryaddress": "STREET_ADDRESS",
        "buildingnumber": "BUILDING_NUMBER",
        "zipcode": "ZIPCODE",
        "zip": "ZIPCODE",
        "postcode": "ZIPCODE",
        "postalcode": "ZIPCODE",
        "gpscoordinates": "GPS_COORDINATES",
        "gps": "GPS_COORDINATES",
        "ordinaldirection": "ORDINAL_DIRECTION",
        "date": "DATE",
        "dateofbirth": "DATE_OF_BIRTH",
        "dob": "DATE_OF_BIRTH",
        "birthdate": "DATE_OF_BIRTH",
        "time": "TIME",
        "age": "AGE",
        "idnum": "ID_NUM",
        "id": "ID_NUM",
        "identifier": "ID_NUM",
        "medicalrecordnumber": "ID_NUM",
        "mrn": "ID_NUM",
        "nationalid": "ID_NUM",
        "cpf": "ID_NUM",
        "cnpj": "ID_NUM",
        "nir": "ID_NUM",
        "steuerid": "ID_NUM",
        "codicefiscale": "ID_NUM",
        "dni": "ID_NUM",
        "nie": "ID_NUM",
        "bsn": "ID_NUM",
        "aadhaar": "ID_NUM",
        "npi": "ID_NUM",
        "insuranceid": "ID_NUM",
        "memberid": "ID_NUM",
        "policy": "ID_NUM",
        "policynumber": "ID_NUM",
        "driverlicense": "ID_NUM",
        "passportnumber": "ID_NUM",
        "passport": "ID_NUM",
        "employeeid": "ID_NUM",
        "encounternumber": "ID_NUM",
        "documentid": "ID_NUM",
        "documentnumber": "ID_NUM",
        "ssn": "SSN",
        "socialsecuritynumber": "SSN",
        "accountnumber": "ACCOUNT_NUMBER",
        "accountname": "ACCOUNT_NUMBER",
        "bankaccount": "ACCOUNT_NUMBER",
        "routing": "ACCOUNT_NUMBER",
        "routingnumber": "ACCOUNT_NUMBER",
        "password": "PASSWORD",
        "pin": "PIN",
        "apikey": "API_KEY",
        "creditcard": "CREDIT_CARD",
        "creditdebitcard": "CREDIT_CARD",
        "creditcardnumber": "CREDIT_CARD",
        "paymentcard": "CREDIT_CARD",
        "creditcardissuer": "CREDIT_CARD_ISSUER",
        "cvv": "CVV",
        "iban": "IBAN",
        "bic": "BIC",
        "swift": "BIC",
        "amount": "AMOUNT",
        "currency": "CURRENCY",
        "currencycode": "CURRENCY",
        "currencyname": "CURRENCY",
        "currencysymbol": "CURRENCY",
        "bitcoinaddress": "BITCOIN_ADDRESS",
        "ethereumaddress": "ETHEREUM_ADDRESS",
        "litecoinaddress": "LITECOIN_ADDRESS",
        "maskednumber": "MASKED_NUMBER",
        "gender": "GENDER",
        "sex": "GENDER",
        "eyecolor": "EYE_COLOR",
        "height": "HEIGHT",
        "organization": "ORGANIZATION",
        "company": "ORGANIZATION",
        "employer": "ORGANIZATION",
        "jobtitle": "JOB_TITLE",
        "jobdepartment": "JOB_DEPARTMENT",
        "department": "JOB_DEPARTMENT",
        "occupation": "OCCUPATION",
        "profession": "OCCUPATION",
        "ipaddress": "IP_ADDRESS",
        "ip": "IP_ADDRESS",
        "ipv4": "IP_ADDRESS",
        "ipv6": "IP_ADDRESS",
        "macaddress": "MAC_ADDRESS",
        "useragent": "USER_AGENT",
        "vin": "VIN",
        "vrm": "VEHICLE_REGISTRATION",
        "licenseplate": "VEHICLE_REGISTRATION",
        "imei": "IMEI",
        "microorganism": "MICROORGANISM",
        "microbe": "MICROORGANISM",
        "organism": "MICROORGANISM",
        "pathogen": "MICROORGANISM",
        "antibiotic": "ANTIBIOTIC",
        "antimicrobial": "ANTIBIOTIC",
        "susceptibility": "SUSCEPTIBILITY",
        "susceptibilityresult": "SUSCEPTIBILITY",
    ]
}

/// A single policy action applied to a detected span.
public struct DeidentifiedSpanAction: Equatable, Sendable {
    public let label: String
    public let canonicalLabel: String
    public let action: PolicyAction
    public let start: Int
    public let end: Int
    public let confidence: Float
    public let replacement: String?
}

/// Result of an on-device de-identification run.
public struct DeidentificationResult: Equatable, Sendable {
    public let redactedText: String
    public let policyName: String
    public let actions: [DeidentifiedSpanAction]
}

private struct PolicyPayload: Decodable {
    let schemaVersion: Int
    let name: String
    let posture: String
    let thresholdProfile: String
    let defaultAction: PolicyAction
    let defaultActionBias: String
    let arbitrationMode: String
    let strictNoLeak: Bool
    let safetySweepMandatory: Bool
    let keepMapping: Bool
    let reversibleID: Bool
    let forcedCascadeTiers: [String]
    let policyLabelActions: [String: PolicyAction]
    let actions: [String: PolicyAction]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case name
        case posture
        case thresholdProfile = "threshold_profile"
        case defaultAction = "default_action"
        case defaultActionBias = "default_action_bias"
        case arbitrationMode = "arbitration_mode"
        case strictNoLeak = "strict_no_leak"
        case safetySweepMandatory = "safety_sweep_mandatory"
        case keepMapping = "keep_mapping"
        case reversibleID = "reversible_id"
        case forcedCascadeTiers = "forced_cascade_tiers"
        case policyLabelActions = "policy_label_actions"
        case actions
    }
}

extension JSONDecoder {
    fileprivate static var openMedPolicy: JSONDecoder {
        JSONDecoder()
    }
}
