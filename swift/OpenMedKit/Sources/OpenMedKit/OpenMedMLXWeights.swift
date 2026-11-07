#if canImport(MLX) && canImport(ZIPFoundation) && !os(watchOS) && !os(visionOS)
    import Foundation
    import MLX
    import ZIPFoundation

    /// Value-free MLX weight failures.
    ///
    /// A candidate index points into the artifact's ordered weight candidates
    /// and an entry index into the archive's `.npy` entries. Cases never hold a
    /// URL, file or entry name, or header text.
    enum OpenMedMLXWeightError: LocalizedError, Equatable {
        case unsupportedWeightFile(candidateIndex: Int)
        case missingWeights(candidateCount: Int)
        case unreadableWeights(candidateIndex: Int)
        case invalidNPZArchive(candidateIndex: Int)
        case invalidNPYHeader(entryIndex: Int)
        case unsupportedNPYDType(entryIndex: Int)
        case unsupportedNPYEndianness(entryIndex: Int)
        case unsupportedFortranOrder(entryIndex: Int)

        var errorDescription: String? {
            switch self {
            case .unsupportedWeightFile(let index):
                return "MLX weight candidate \(index) has an unsupported file type."
            case .missingWeights(let count):
                return "No MLX weights were found among \(count) candidates."
            case .unreadableWeights(let index):
                return "MLX weight candidate \(index) could not be loaded."
            case .invalidNPZArchive(let index):
                return "MLX weight candidate \(index) is not a readable NPZ archive."
            case .invalidNPYHeader(let index):
                return "NPZ entry \(index) has an invalid NumPy array header."
            case .unsupportedNPYDType(let index):
                return "NPZ entry \(index) uses a NumPy dtype that Swift MLX loading does not support."
            case .unsupportedNPYEndianness(let index):
                return "NPZ entry \(index) uses a non-little-endian NumPy dtype."
            case .unsupportedFortranOrder(let index):
                return "NPZ entry \(index) uses Fortran order, which is not supported."
            }
        }
    }

    enum OpenMedMLXWeightArchive {
        static func loadWeights(from candidateURLs: [URL]) throws -> [String: MLXArray] {
            for (index, url) in candidateURLs.enumerated()
            where FileManager.default.fileExists(atPath: url.path) {
                switch url.pathExtension.lowercased() {
                case "safetensors":
                    // MLX load errors embed the absolute path in their message.
                    do {
                        return try MLX.loadArrays(url: url)
                    } catch {
                        throw OpenMedMLXWeightError.unreadableWeights(candidateIndex: index)
                    }
                case "npz":
                    return try loadNPZ(url: url, candidateIndex: index)
                default:
                    throw OpenMedMLXWeightError.unsupportedWeightFile(candidateIndex: index)
                }
            }

            throw OpenMedMLXWeightError.missingWeights(candidateCount: candidateURLs.count)
        }

        private static func loadNPZ(url: URL, candidateIndex: Int) throws -> [String: MLXArray] {
            let archive: Archive
            do {
                archive = try Archive(url: url, accessMode: .read)
            } catch {
                throw OpenMedMLXWeightError.invalidNPZArchive(candidateIndex: candidateIndex)
            }

            var arrays = [String: MLXArray]()
            for (entryIndex, entry) in archive.filter({ $0.path.hasSuffix(".npy") }).enumerated() {
                var data = Data()
                do {
                    _ = try archive.extract(entry) { chunk in
                        data.append(chunk)
                    }
                } catch {
                    throw OpenMedMLXWeightError.invalidNPZArchive(candidateIndex: candidateIndex)
                }

                let key = URL(fileURLWithPath: entry.path).deletingPathExtension().lastPathComponent
                arrays[key] = try loadNPY(data: data, entryIndex: entryIndex)
            }

            return arrays
        }

        private static func loadNPY(data: Data, entryIndex: Int) throws -> MLXArray {
            let header = try parseNPYHeader(data: data, entryIndex: entryIndex)
            guard !header.fortranOrder else {
                throw OpenMedMLXWeightError.unsupportedFortranOrder(entryIndex: entryIndex)
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

        private static func parseNPYHeader(data: Data, entryIndex: Int) throws -> ParsedNPYHeader {
            let minimumLength = 10
            guard data.count >= minimumLength else {
                throw OpenMedMLXWeightError.invalidNPYHeader(entryIndex: entryIndex)
            }

            let magic = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])
            guard data.prefix(6) == magic else {
                throw OpenMedMLXWeightError.invalidNPYHeader(entryIndex: entryIndex)
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
                    throw OpenMedMLXWeightError.invalidNPYHeader(entryIndex: entryIndex)
                }
                headerLength =
                    Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16)
                    | (Int(data[11]) << 24)
                headerOffset = 12
            default:
                throw OpenMedMLXWeightError.invalidNPYHeader(entryIndex: entryIndex)
            }

            let endOffset = headerOffset + headerLength
            guard data.count >= endOffset else {
                throw OpenMedMLXWeightError.invalidNPYHeader(entryIndex: entryIndex)
            }

            let headerData = data.subdata(in: headerOffset..<endOffset)
            guard let header = String(data: headerData, encoding: .ascii) else {
                throw OpenMedMLXWeightError.invalidNPYHeader(entryIndex: entryIndex)
            }

            let descr = try match(pattern: "'descr'\\s*:\\s*'([^']+)'", in: header, entryIndex: entryIndex)
            let fortranText = try match(
                pattern: "'fortran_order'\\s*:\\s*(True|False)",
                in: header,
                entryIndex: entryIndex
            )
            let shapeText = try match(
                pattern: "'shape'\\s*:\\s*\\(([^\\)]*)\\)",
                in: header,
                entryIndex: entryIndex
            )

            let dtype = try dtype(from: descr, entryIndex: entryIndex)
            let shape = parseShape(shapeText)

            return ParsedNPYHeader(
                dtype: dtype,
                shape: shape,
                fortranOrder: fortranText == "True",
                dataOffset: endOffset
            )
        }

        private static func match(pattern: String, in header: String, entryIndex: Int) throws -> String {
            let regex = try NSRegularExpression(pattern: pattern)
            let range = NSRange(header.startIndex..<header.endIndex, in: header)
            guard
                let result = regex.firstMatch(in: header, options: [], range: range),
                result.numberOfRanges > 1,
                let captureRange = Range(result.range(at: 1), in: header)
            else {
                throw OpenMedMLXWeightError.invalidNPYHeader(entryIndex: entryIndex)
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

        private static func dtype(from descriptor: String, entryIndex: Int) throws -> DType {
            if descriptor.hasPrefix(">") {
                throw OpenMedMLXWeightError.unsupportedNPYEndianness(entryIndex: entryIndex)
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
                throw OpenMedMLXWeightError.unsupportedNPYDType(entryIndex: entryIndex)
            }
        }
    }
#endif
