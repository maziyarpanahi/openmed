import Foundation

/// The portable identifier checksum validators, ported from the Python
/// references named in `tests/fixtures/parity/identifier_validators_v1.json`.
///
/// Each function returns whether a candidate passes the same format and
/// checksum rules as its Python reference. They never throw, log or keep the
/// candidate. Normalization follows each reference exactly: some strip only
/// ASCII digits (`_digits_only`), some remove Python whitespace or upper-case
/// first, and some accept any Unicode decimal digit where Python's `\d` or
/// `int()` does. Detection patterns do not call these validators yet.
public enum OpenMedIdentifierValidators {
    /// Luhn checksum over the ASCII digits, at least 13 of them
    /// (`clinical_ids.validate_luhn`).
    public static func luhn(_ text: String) -> Bool {
        let digits = asciiDigits(text)
        guard digits.count >= 13, let last = digits.last else { return false }
        return luhnCheckDigit(digits.dropLast()) == last
    }

    /// US Social Security Number: nine ASCII digits, area not `000`, `666` or
    /// `9xx`, group not `00`, serial not `0000` (`clinical_ids.validate_ssn`).
    public static func ssn(_ text: String) -> Bool {
        let digits = asciiDigits(text)
        guard digits.count == 9 else { return false }
        let area = Array(digits[0..<3])
        if area == [0, 0, 0] || area == [6, 6, 6] || area[0] == 9 { return false }
        if digits[3..<5].allSatisfy({ $0 == 0 }) || digits[5..<9].allSatisfy({ $0 == 0 }) { return false }
        return true
    }

    /// US National Provider Identifier: ten ASCII digits with the Luhn check
    /// over the `80840` prefix (`clinical_ids.validate_npi`).
    public static func npi(_ text: String) -> Bool {
        let digits = asciiDigits(text)
        guard digits.count == 10, let last = digits.last else { return false }
        return luhnCheckDigit([8, 0, 8, 4, 0] + digits.dropLast()) == last
    }

    /// IBAN with the ISO 13616 mod-97 checksum and the per-country length
    /// (`clinical_ids.validate_iban`).
    public static func iban(_ text: String) -> Bool {
        let cleaned = Array(
            String(String.UnicodeScalarView(text.unicodeScalars.filter { !isPythonSpace($0) && $0 != "-" }))
                .uppercased().unicodeScalars)
        // [A-Z]{2}\d{2}[A-Z0-9]{11,30}
        guard (15...34).contains(cleaned.count),
            cleaned[0..<2].allSatisfy(isASCIIUppercase),
            cleaned[2..<4].allSatisfy({ decimalValue($0) != nil }),
            cleaned[4...].allSatisfy({ isASCIIUppercase($0) || isASCIIDigit($0) })
        else { return false }
        let country = String(String.UnicodeScalarView(cleaned[0..<2]))
        if let length = ibanLengths[country], cleaned.count != length { return false }
        var remainder = 0
        for scalar in cleaned[4...] + cleaned[0..<4] {
            if let digit = decimalValue(scalar) {
                remainder = (remainder * 10 + digit) % 97
            } else {
                // A letter counts as its two-digit value, A = 10 ... Z = 35.
                let value = Int(scalar.value) - 55
                remainder = (remainder * 10 + value / 10) % 97
                remainder = (remainder * 10 + value % 10) % 97
            }
        }
        return remainder == 1
    }

    /// Indian Aadhaar: twelve digits, or the `4 4 4` display form, first digit
    /// 2 to 9, with the Verhoeff checksum; Gurmukhi and Arabic-Indic digits
    /// count as ASCII (`pii_i18n.validate_aadhaar`).
    public static func aadhaar(_ text: String) -> Bool {
        let candidate = pythonStrip(text).unicodeScalars.map(foldIndicDigit)
        let compact = candidate.count == 12
        let spaced = candidate.count == 14 && candidate[4] == " " && candidate[9] == " "
        guard compact || spaced else { return false }
        let digitScalars = candidate.filter { $0 != " " }
        guard digitScalars.count == 12, digitScalars.allSatisfy(isASCIIDigit), digitScalars[0] >= "2" else {
            return false
        }
        var check = 0
        for (index, scalar) in digitScalars.reversed().enumerated() {
            check = verhoeffD[check][verhoeffP[index % 8][Int(scalar.value - 48)]]
        }
        return check == 0
    }

    /// UK NHS Number: ten ASCII digits with the Modulus 11 check
    /// (`clinical_ids.validate_uk_nhs_number`).
    public static func ukNHSNumber(_ text: String) -> Bool {
        let digits = asciiDigits(text)
        guard digits.count == 10 else { return false }
        let total = zip(stride(from: 10, to: 1, by: -1), digits[0..<9]).reduce(0) { $0 + $1.0 * $1.1 }
        var check = 11 - total % 11
        if check == 11 { check = 0 }
        return check != 10 && digits[9] == check
    }

    /// Dutch BSN: eight or nine ASCII digits, zero-padded to nine, not all
    /// zeros, passing the Elfproef (`pii_i18n.validate_dutch_bsn`).
    public static func dutchBSN(_ text: String) -> Bool {
        var digits = asciiDigits(text)
        guard digits.count == 8 || digits.count == 9 else { return false }
        if digits.count == 8 { digits.insert(0, at: 0) }
        guard digits.contains(where: { $0 != 0 }) else { return false }
        let weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
        return zip(digits, weights).reduce(0) { $0 + $1.0 * $1.1 } % 11 == 0
    }

    /// Spanish DNI: eight digits and the check letter for the number modulo 23
    /// (`pii_i18n.validate_spanish_dni`).
    public static func spanishDNI(_ text: String) -> Bool {
        let cleaned = Array(
            String(String.UnicodeScalarView(text.unicodeScalars.filter { !isPythonSpace($0) })).uppercased()
                .unicodeScalars)
        guard cleaned.count == 9, isASCIIUppercase(cleaned[8]) else { return false }
        var number = 0
        for scalar in cleaned[0..<8] {
            guard let digit = decimalValue(scalar) else { return false }
            number = number * 10 + digit
        }
        return cleaned[8] == dniLetters[number % 23]
    }

    /// French NIR: fifteen characters starting with 1 or 2, Corsica `2A` and
    /// `2B` read as 19 and 18, and the key `97 - (body mod 97)`
    /// (`pii_i18n.validate_french_nir`).
    public static func frenchNIR(_ text: String) -> Bool {
        let removed = text.unicodeScalars.filter { !isPythonSpace($0) && $0 != "." && $0 != "-" }
        let cleaned = Array(String(String.UnicodeScalarView(removed)).uppercased().unicodeScalars)
        guard cleaned.count == 15, cleaned[0] == "1" || cleaned[0] == "2",
            let key = pythonTwoCharacterInt(cleaned[13], cleaned[14])
        else { return false }
        var body = Array(cleaned[0..<13])
        if body.contains("A") || body.contains("B") {
            // ^[12]\d{4}2[AB]\d{6}$
            guard body[1..<5].allSatisfy({ decimalValue($0) != nil }), body[5] == "2",
                body[6] == "A" || body[6] == "B", body[7...].allSatisfy({ decimalValue($0) != nil })
            else { return false }
            body.replaceSubrange(5..<7, with: body[6] == "A" ? ["1", "9"] : ["1", "8"])
        }
        var number = 0
        for scalar in body {
            guard let digit = decimalValue(scalar) else { return false }
            number = number * 10 + digit
        }
        return key == 97 - number % 97
    }

    /// Mainland China 18-digit resident ID: province prefix, prefecture and
    /// county structure, a real birth date not after `today`, a non-zero
    /// sequence and the ISO 7064 MOD 11-2 check character
    /// (`pii_i18n.validate_chinese_resident_id`).
    ///
    /// - Parameters:
    ///   - text: The candidate; a lowercase final `x` is accepted.
    ///   - today: The date birth dates are compared with, in the current time
    ///     zone like Python's `date.today()`. Defaults to now.
    public static func chineseResidentID(_ text: String, today: Date = Date()) -> Bool {
        let cleaned = Array(text.uppercased().unicodeScalars)
        guard cleaned.count == 18, cleaned[0..<17].allSatisfy(isASCIIDigit),
            isASCIIDigit(cleaned[17]) || cleaned[17] == "X"
        else { return false }
        let digits = cleaned[0..<17].map { Int($0.value - 48) }
        let prefix = digits[0] * 10 + digits[1]
        let prefecture = digits[2] * 10 + digits[3]
        guard chineseRegionPrefixes.contains(prefix), prefecture <= 70 || prefecture == 90, prefecture != 0,
            digits[4] != 0 || digits[5] != 0
        else { return false }
        let year = digits[6..<10].reduce(0) { $0 * 10 + $1 }
        let month = digits[10] * 10 + digits[11]
        let day = digits[12] * 10 + digits[13]
        guard year >= 1, (1...12).contains(month), (1...daysInMonth(year: year, month: month)).contains(day) else {
            return false
        }
        let now = Calendar(identifier: .gregorian).dateComponents(in: TimeZone.current, from: today)
        guard let nowYear = now.year, let nowMonth = now.month, let nowDay = now.day,
            (year, month, day) <= (nowYear, nowMonth, nowDay),
            digits[14..<17].contains(where: { $0 != 0 })
        else { return false }
        let total = zip(digits, chineseWeights).reduce(0) { $0 + $1.0 * $1.1 }
        return cleaned[17] == chineseCheckCharacters[total % 11]
    }

    /// Canadian SIN: nine digits, or `NNN NNN NNN` / `NNN-NNN-NNN` with one
    /// separator, not starting with 0, passing the Luhn check
    /// (`clinical_ids.validate_canadian_sin`).
    public static func canadianSIN(_ text: String) -> Bool {
        let candidate = Array(pythonStrip(text).unicodeScalars)
        let isDigit = { (scalar: Unicode.Scalar) in decimalValue(scalar) != nil }
        let plain = candidate.count == 9 && candidate.allSatisfy(isDigit)
        let separated =
            candidate.count == 11 && (candidate[3] == " " || candidate[3] == "-") && candidate[7] == candidate[3]
            && candidate.enumerated().allSatisfy { $0.offset == 3 || $0.offset == 7 || isDigit($0.element) }
        guard plain || separated else { return false }
        let digits = asciiDigits(String(String.UnicodeScalarView(candidate)))
        guard digits.count == 9, digits[0] != 0, let last = digits.last else { return false }
        return luhnCheckDigit(digits.dropLast()) == last
    }

    // MARK: - Shared helpers

    /// Python's `re.sub(r"[^0-9]", "", text)`, as digit values.
    private static func asciiDigits(_ text: String) -> [Int] {
        text.unicodeScalars.filter(isASCIIDigit).map { Int($0.value - 48) }
    }

    private static func luhnCheckDigit<C: Collection>(_ digits: C) -> Int where C.Element == Int {
        var total = 0
        for (index, digit) in digits.reversed().enumerated() {
            var value = digit
            if index % 2 == 0 {
                value *= 2
                if value > 9 { value -= 9 }
            }
            total += value
        }
        return (10 - total % 10) % 10
    }

    /// The value of a Unicode decimal digit (`\d`, or what `int()` reads).
    private static func decimalValue(_ scalar: Unicode.Scalar) -> Int? {
        guard scalar.properties.numericType == .decimal, let value = scalar.properties.numericValue else { return nil }
        return Int(value)
    }

    /// Python's `int()` of a two-character string after whitespace removal:
    /// two decimal digits, or `+` and one digit.
    private static func pythonTwoCharacterInt(_ first: Unicode.Scalar, _ second: Unicode.Scalar) -> Int? {
        guard let low = decimalValue(second) else { return nil }
        if first == "+" { return low }
        guard let high = decimalValue(first) else { return nil }
        return high * 10 + low
    }

    /// Python's `str.isspace()`: Unicode White_Space plus U+001C to U+001F.
    private static func isPythonSpace(_ scalar: Unicode.Scalar) -> Bool {
        scalar.properties.isWhitespace || (0x1C...0x1F).contains(scalar.value)
    }

    /// Python's `str.strip()`.
    private static func pythonStrip(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        guard let start = scalars.firstIndex(where: { !isPythonSpace($0) }),
            let end = scalars.lastIndex(where: { !isPythonSpace($0) })
        else { return "" }
        return String(String.UnicodeScalarView(scalars[start...end]))
    }

    private static func isASCIIDigit(_ scalar: Unicode.Scalar) -> Bool { ("0"..."9").contains(scalar) }

    private static func isASCIIUppercase(_ scalar: Unicode.Scalar) -> Bool { ("A"..."Z").contains(scalar) }

    /// Folds Gurmukhi (U+0A66...) and Arabic-Indic (U+0660..., U+06F0...)
    /// digits to ASCII, one scalar at a time.
    private static func foldIndicDigit(_ scalar: Unicode.Scalar) -> Unicode.Scalar {
        for base: UInt32 in [0x0A66, 0x0660, 0x06F0] where (base...(base + 9)).contains(scalar.value) {
            return Unicode.Scalar(48 + scalar.value - base)!
        }
        return scalar
    }

    private static func daysInMonth(year: Int, month: Int) -> Int {
        switch month {
        case 2: return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 ? 29 : 28
        case 4, 6, 9, 11: return 30
        default: return 31
        }
    }

    private static let dniLetters = Array("TRWAGMYFPDXBNJZSQVHLCKE".unicodeScalars)
    private static let chineseWeights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    private static let chineseCheckCharacters = Array("10X98765432".unicodeScalars)
    private static let chineseRegionPrefixes: Set<Int> = [
        11, 12, 13, 14, 15, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54,
        61, 62, 63, 64, 65,
    ]
    private static let verhoeffD: [[Int]] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ]
    private static let verhoeffP: [[Int]] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
    ]
    private static let ibanLengths: [String: Int] = [
        "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22, "BR": 29,
        "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29,
        "ES": 24, "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28,
        "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28,
        "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MR": 27,
        "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24,
        "RS": 22, "SA": 24, "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "TN": 24, "TR": 26, "UA": 29,
        "VA": 22, "VG": 24, "XK": 20,
    ]
}
