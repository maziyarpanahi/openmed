import Foundation
import MLX
import ZIPFoundation

enum OpenMedMLXWeightError: LocalizedError {
    case unsupportedWeightFile(URL)
    case missingWeights([URL])
    case invalidNPYHeader(String)
    case unsupportedNPYDType(String)
    case unsupportedNPYEndianness(String)
    case unsupportedFortranOrder(String)
    case invalidNPZArchive(URL)

    var errorDescription: String? {
        switch self {
        case .unsupportedWeightFile(let url):
            return "Unsupported MLX weight file: \(url.lastPathComponent)"
        case .missingWeights(let urls):
            let checked = urls.map(\.lastPathComponent).joined(separator: ", ")
            return "No MLX weights found. Checked: \(checked)"
        case .invalidNPYHeader(let name):
            return "Invalid NumPy array header in \(name)"
        case .unsupportedNPYDType(let dtype):
            return "Unsupported NumPy dtype for Swift MLX loading: \(dtype)"
        case .unsupportedNPYEndianness(let dtype):
            return "Unsupported non-little-endian NumPy dtype: \(dtype)"
        case .unsupportedFortranOrder(let name):
            return "Fortran-ordered NumPy arrays are not supported in \(name)"
        case .invalidNPZArchive(let url):
            return "Unable to open NPZ archive at \(url.path)"
        }
    }
}

enum OpenMedMLXWeightArchive {
    static func loadWeights(from candidateURLs: [URL]) throws -> [String: MLXArray] {
        for url in candidateURLs where FileManager.default.fileExists(atPath: url.path) {
            switch url.pathExtension.lowercased() {
            case "safetensors":
                return try MLX.loadArrays(url: url)
            case "npz":
                return try loadNPZ(url: url)
            default:
                throw OpenMedMLXWeightError.unsupportedWeightFile(url)
            }
        }

        throw OpenMedMLXWeightError.missingWeights(candidateURLs)
    }

    private static func loadNPZ(url: URL) throws -> [String: MLXArray] {
        let archive: Archive
        do {
            archive = try Archive(url: url, accessMode: .read)
        } catch {
            throw OpenMedMLXWeightError.invalidNPZArchive(url)
        }

        var arrays = [String: MLXArray]()
        for entry in archive where entry.path.hasSuffix(".npy") {
            var data = Data()
            _ = try archive.extract(entry) { chunk in
                data.append(chunk)
            }

            let key = URL(fileURLWithPath: entry.path).deletingPathExtension().lastPathComponent
            arrays[key] = try loadNPY(data: data, name: entry.path)
        }

        return arrays
    }

    private static func loadNPY(data: Data, name: String) throws -> MLXArray {
        let header = try parseNPYHeader(data: data, name: name)
        guard !header.fortranOrder else {
            throw OpenMedMLXWeightError.unsupportedFortranOrder(name)
        }

        let payload = data.subdata(in: header.dataOffset..<data.count)
        return MLXArray(payload, header.shape, dtype: header.dtype)
    }

    private struct ParsedNPYHeader {
        let dtype: DType
        let shape: [Int]
        let fortranOrder: Bool
        let dataOffset: Int
    }

    private static func parseNPYHeader(data: Data, name: String) throws -> ParsedNPYHeader {
        let minimumLength = 10
        guard data.count >= minimumLength else {
            throw OpenMedMLXWeightError.invalidNPYHeader(name)
        }

        let magic = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])
        guard data.prefix(6) == magic else {
            throw OpenMedMLXWeightError.invalidNPYHeader(name)
        }

        let major = Int(data[6])
        let headerLength: Int
        let headerOffset: Int
        switch major {
        case 1:
            headerLength = Int(data[8]) | (Int(data[9]) << 8)
            headerOffset = 10
        case 2, 3:
            guard data.count >= 12 else {
                throw OpenMedMLXWeightError.invalidNPYHeader(name)
            }
            headerLength = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16)
                | (Int(data[11]) << 24)
            headerOffset = 12
        default:
            throw OpenMedMLXWeightError.invalidNPYHeader(name)
        }

        let endOffset = headerOffset + headerLength
        guard data.count >= endOffset else {
            throw OpenMedMLXWeightError.invalidNPYHeader(name)
        }

        let headerData = data.subdata(in: headerOffset..<endOffset)
        guard let header = String(data: headerData, encoding: .ascii) else {
            throw OpenMedMLXWeightError.invalidNPYHeader(name)
        }

        let descr = try match(pattern: "'descr'\\s*:\\s*'([^']+)'", in: header, name: name)
        let fortranText = try match(
            pattern: "'fortran_order'\\s*:\\s*(True|False)",
            in: header,
            name: name
        )
        let shapeText = try match(pattern: "'shape'\\s*:\\s*\\(([^\\)]*)\\)", in: header, name: name)

        let dtype = try dtype(from: descr)
        let shape = parseShape(shapeText)

        return ParsedNPYHeader(
            dtype: dtype,
            shape: shape,
            fortranOrder: fortranText == "True",
            dataOffset: endOffset
        )
    }

    private static func match(pattern: String, in header: String, name: String) throws -> String {
        let regex = try NSRegularExpression(pattern: pattern)
        let range = NSRange(header.startIndex..<header.endIndex, in: header)
        guard
            let result = regex.firstMatch(in: header, options: [], range: range),
            result.numberOfRanges > 1,
            let captureRange = Range(result.range(at: 1), in: header)
        else {
            throw OpenMedMLXWeightError.invalidNPYHeader(name)
        }
        return String(header[captureRange])
    }

    private static func parseShape(_ shapeText: String) -> [Int] {
        shapeText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .compactMap(Int.init)
    }

    private static func dtype(from descriptor: String) throws -> DType {
        if descriptor.hasPrefix(">") {
            throw OpenMedMLXWeightError.unsupportedNPYEndianness(descriptor)
        }

        switch descriptor {
        case "|b1", "?":
            return .bool
        case "|u1", "<u1":
            return .uint8
        case "<u2":
            return .uint16
        case "<u4":
            return .uint32
        case "<u8":
            return .uint64
        case "|i1", "<i1":
            return .int8
        case "<i2":
            return .int16
        case "<i4":
            return .int32
        case "<i8":
            return .int64
        case "<f2":
            return .float16
        case "<f4":
            return .float32
        case "<f8":
            return .float64
        default:
            throw OpenMedMLXWeightError.unsupportedNPYDType(descriptor)
        }
    }
}
