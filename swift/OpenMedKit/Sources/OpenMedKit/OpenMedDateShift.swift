import CryptoKit
import Foundation

/// Patient-keyed date-shift offsets, matching `openmed.core.date_shift`.
///
/// Python's `stable_offset_for()` and `stable_offset_from_seed()` give every
/// patient one deterministic, non-zero day offset in
/// `[-maxDays, -1] ∪ [1, maxDays]`. These functions compute the same offset on
/// device, so a note shifted on an iPhone and a record shifted in Python move
/// the same patient by the same number of days:
///
/// ```swift
/// let days = try OpenMedDateShift.stableOffset(
///     patientKey: patientKey, secret: secret, maxDays: OpenMedDateShift.defaultMaxDays)
/// let shifted = try OpenMedDateShift.shift(isoDate: "2024-03-01", byDays: days)
/// ```
///
/// The patient key is only the HMAC-SHA256 message under the caller's secret;
/// neither value is stored, logged or put into an error. The 256-bit digest is
/// reduced modulo `2 * maxDays` byte by byte, which equals Python's
/// `int.from_bytes(digest, "big") % (2 * max_days)` without a big integer.
public enum OpenMedDateShift {
    /// Python's `DEFAULT_DATE_SHIFT_MAX_DAYS`.
    public static let defaultMaxDays = 365
    /// The largest `maxDays` accepted, so `2 * maxDays` fits in an `Int`.
    public static let maximumMaxDays = Int.max / 2

    /// The domain prefix of seeded offsets, `openmed-date-shift-seed-v1\0`.
    static let seedDomain = Data("openmed-date-shift-seed-v1".utf8) + Data([0])

    // MARK: - Offsets

    /// The stable offset for a patient key.
    ///
    /// - Parameters:
    ///   - patientKey: The caller's stable patient key bytes.
    ///   - secret: The HMAC secret bytes.
    ///   - maxDays: The largest shift, in days; at least 1.
    /// - Returns: A non-zero offset in `-maxDays...maxDays`.
    /// - Throws: ``OpenMedDateShiftError``.
    public static func stableOffset(patientKey: Data, secret: Data, maxDays: Int) throws -> Int {
        try validate(maxDays: maxDays)
        guard !patientKey.isEmpty else { throw OpenMedDateShiftError.patientKeyEmpty }
        guard !secret.isEmpty else { throw OpenMedDateShiftError.secretEmpty }
        let digest = HMAC<SHA256>.authenticationCode(for: patientKey, using: SymmetricKey(data: secret))
        return offset(fromDigest: Data(digest), maxDays: maxDays)
    }

    /// The stable offset for a patient key given as text, hashed as UTF-8.
    ///
    /// - Throws: ``OpenMedDateShiftError``.
    public static func stableOffset(patientKey: String, secret: Data, maxDays: Int) throws -> Int {
        try stableOffset(patientKey: Data(patientKey.utf8), secret: secret, maxDays: maxDays)
    }

    /// The stable offset for a patient key and a text secret, both UTF-8.
    ///
    /// - Throws: ``OpenMedDateShiftError``.
    public static func stableOffset(patientKey: String, secret: String, maxDays: Int) throws -> Int {
        try stableOffset(patientKey: Data(patientKey.utf8), secret: Data(secret.utf8), maxDays: maxDays)
    }

    /// The stable offset for a request seed, as `stable_offset_from_seed()`.
    ///
    /// - Parameters:
    ///   - seed: The request seed.
    ///   - maxDays: The largest shift, in days; at least 1.
    /// - Returns: A non-zero offset in `-maxDays...maxDays`.
    /// - Throws: ``OpenMedDateShiftError/maxDaysInvalid``.
    public static func stableOffset(seed: Int, maxDays: Int) throws -> Int {
        try stableOffset(seedDecimal: String(seed), maxDays: maxDays)
    }

    /// The stable offset for a seed given as canonical decimal text, for seeds
    /// outside the `Int` range.
    ///
    /// - Parameters:
    ///   - seedDecimal: The seed as Python's `str(seed)` writes it: an optional
    ///     `-`, then digits without leading zeros, and `0` for zero.
    ///   - maxDays: The largest shift, in days; at least 1.
    /// - Returns: A non-zero offset in `-maxDays...maxDays`.
    /// - Throws: ``OpenMedDateShiftError``.
    public static func stableOffset(seedDecimal: String, maxDays: Int) throws -> Int {
        try validate(maxDays: maxDays)
        guard isCanonicalDecimal(seedDecimal) else { throw OpenMedDateShiftError.seedInvalid }
        let digest = SHA256.hash(data: seedDomain + Data(seedDecimal.utf8))
        return offset(fromDigest: Data(digest), maxDays: maxDays)
    }

    /// Maps a digest onto `[-maxDays, -1] ∪ [1, maxDays]`.
    static func offset(fromDigest digest: Data, maxDays: Int) -> Int {
        let modulus = UInt64(maxDays) * 2
        var remainder: UInt64 = 0
        for byte in digest {
            // (remainder * 256 + byte) mod modulus, in 128-bit intermediate width.
            let product = remainder.multipliedFullWidth(by: 256)
            let (low, carry) = product.low.addingReportingOverflow(UInt64(byte))
            remainder = modulus.dividingFullWidth((product.high + (carry ? 1 : 0), low)).remainder
        }
        let bucket = Int(remainder)
        return bucket < maxDays ? bucket - maxDays : bucket - maxDays + 1
    }

    // MARK: - Dates

    /// Shifts a `YYYY-MM-DD` date by whole days in the proleptic Gregorian
    /// calendar, as Python's `date + timedelta(days=...)` does.
    ///
    /// - Parameters:
    ///   - isoDate: A date such as `2024-03-01`, with a four-digit year from
    ///     0001 to 9999.
    ///   - days: The offset, positive or negative.
    /// - Returns: The shifted date in the same format.
    /// - Throws: ``OpenMedDateShiftError/dateInvalid`` for anything that is not
    ///   a real `YYYY-MM-DD` date, or ``OpenMedDateShiftError/dateOutOfRange``
    ///   when the result leaves 0001-01-01...9999-12-31.
    public static func shift(isoDate: String, byDays days: Int) throws -> String {
        guard let (year, month, day) = parse(isoDate) else { throw OpenMedDateShiftError.dateInvalid }
        let start = daysFromCivil(year: year, month: month, day: day)
        let (shifted, overflow) = start.addingReportingOverflow(days)
        guard !overflow, (minimumDay...maximumDay).contains(shifted) else {
            throw OpenMedDateShiftError.dateOutOfRange
        }
        let result = civilFromDays(shifted)
        return pad(result.year, 4) + "-" + pad(result.month, 2) + "-" + pad(result.day, 2)
    }

    private static let minimumDay = daysFromCivil(year: 1, month: 1, day: 1)
    private static let maximumDay = daysFromCivil(year: 9999, month: 12, day: 31)

    private static func parse(_ text: String) -> (Int, Int, Int)? {
        let scalars = Array(text.unicodeScalars)
        guard scalars.count == 10, scalars[4] == "-", scalars[7] == "-" else { return nil }
        func number(_ range: Range<Int>) -> Int? {
            var value = 0
            for scalar in scalars[range] {
                guard ("0"..."9").contains(scalar) else { return nil }
                value = value * 10 + Int(scalar.value - 48)
            }
            return value
        }
        guard let year = number(0..<4), let month = number(5..<7), let day = number(8..<10),
            year >= 1, (1...12).contains(month), (1...daysInMonth(year: year, month: month)).contains(day)
        else { return nil }
        return (year, month, day)
    }

    private static func daysInMonth(year: Int, month: Int) -> Int {
        switch month {
        case 2: return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 ? 29 : 28
        case 4, 6, 9, 11: return 30
        default: return 31
        }
    }

    /// Days since 1970-01-01 in the proleptic Gregorian calendar.
    private static func daysFromCivil(year: Int, month: Int, day: Int) -> Int {
        let shiftedYear = month <= 2 ? year - 1 : year
        let era = (shiftedYear >= 0 ? shiftedYear : shiftedYear - 399) / 400
        let yearOfEra = shiftedYear - era * 400
        let dayOfYear = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1
        let dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear
        return era * 146_097 + dayOfEra - 719_468
    }

    private static func civilFromDays(_ days: Int) -> (year: Int, month: Int, day: Int) {
        let shifted = days + 719_468
        let era = (shifted >= 0 ? shifted : shifted - 146_096) / 146_097
        let dayOfEra = shifted - era * 146_097
        let yearOfEra = (dayOfEra - dayOfEra / 1460 + dayOfEra / 36_524 - dayOfEra / 146_096) / 365
        let dayOfYear = dayOfEra - (365 * yearOfEra + yearOfEra / 4 - yearOfEra / 100)
        let monthIndex = (5 * dayOfYear + 2) / 153
        let day = dayOfYear - (153 * monthIndex + 2) / 5 + 1
        let month = monthIndex < 10 ? monthIndex + 3 : monthIndex - 9
        return (yearOfEra + era * 400 + (month <= 2 ? 1 : 0), month, day)
    }

    private static func pad(_ value: Int, _ width: Int) -> String {
        let digits = String(value)
        return String(repeating: "0", count: max(0, width - digits.count)) + digits
    }

    // MARK: - Validation

    private static func validate(maxDays: Int) throws {
        guard (1...maximumMaxDays).contains(maxDays) else { throw OpenMedDateShiftError.maxDaysInvalid }
    }

    /// `0`, or an optional `-` followed by digits without a leading zero.
    private static func isCanonicalDecimal(_ text: String) -> Bool {
        let digits = text.hasPrefix("-") ? Substring(text.dropFirst()) : Substring(text)
        guard let first = digits.unicodeScalars.first,
            digits.unicodeScalars.allSatisfy({ ("0"..."9").contains($0) })
        else { return false }
        if first == "0" { return digits.count == 1 && !text.hasPrefix("-") }
        return true
    }
}

/// A rejected date-shift input. The raw value is a value-free reason code; it
/// never carries a patient key, secret, seed or date.
public enum OpenMedDateShiftError: String, Error, LocalizedError, Sendable, CaseIterable {
    /// The patient key is empty.
    case patientKeyEmpty = "patient_key_empty"
    /// The secret is empty.
    case secretEmpty = "secret_empty"
    /// `maxDays` is below 1 or above ``OpenMedDateShift/maximumMaxDays``.
    case maxDaysInvalid = "max_days_invalid"
    /// The seed text is not canonical decimal.
    case seedInvalid = "seed_invalid"
    /// The date is not a real `YYYY-MM-DD` date.
    case dateInvalid = "date_invalid"
    /// The shifted date falls outside 0001-01-01...9999-12-31.
    case dateOutOfRange = "date_out_of_range"

    public var errorDescription: String? { rawValue }
}
