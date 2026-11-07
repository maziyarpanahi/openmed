// swift-tools-version: 5.10

import PackageDescription

// Consumer-facing manifest: SwiftPM resolves this file when an app adds the
// repository URL. swift/OpenMedKit/Package.swift mirrors it for the demos and
// simulator lanes and may differ only in target source paths; CI enforces that
// with scripts/release/check_swift_manifest_drift.py. Change both together.

let package = Package(
    name: "OpenMedKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .watchOS(.v10),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "OpenMedSpeech", targets: ["OpenMedSpeech"]),
        .library(
            name: "OpenMedKit",
            targets: ["OpenMedKit"]
        ),
        .library(
            name: "OpenMedExtensionSupport",
            targets: ["OpenMedExtensionSupport"]
        ),
        .library(
            name: "OpenMedShareExtension",
            targets: ["OpenMedShareExtension"]
        ),
        .library(
            name: "OpenMedActionExtension",
            targets: ["OpenMedActionExtension"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/argmax-oss-swift.git", exact: "1.1.1"),
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            exact: "1.0.0"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.32.3"
        ),
        // 3.32.3 is the first tag that contains upstream #419, which preserves
        // multimodal RoPE state from prefill into autoregressive decode. A tag is
        // required here: SwiftPM rejects revision pins inside packages that
        // consumers require by version.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            exact: "3.32.3"
        ),
        .package(
            url: "https://github.com/weichsel/ZIPFoundation.git",
            from: "0.9.19"
        ),
    ],
    targets: [
        .target(
            name: "OpenMedSpeech",
            dependencies: ["OpenMedKit", .product(name: "WhisperKit", package: "argmax-oss-swift")],
            path: "swift/OpenMedKit/Sources/OpenMedSpeech"
        ),
        .testTarget(
            name: "OpenMedSpeechTests",
            dependencies: ["OpenMedSpeech", "OpenMedKit"],
            path: "swift/OpenMedKit/Tests/OpenMedSpeechTests"
        ),
        .target(
            name: "OpenMedKit",
            dependencies: [
                .product(
                    name: "Transformers",
                    package: "swift-transformers",
                    condition: .when(platforms: [.iOS, .macOS])
                ),
                .product(
                    name: "MLX",
                    package: "mlx-swift",
                    condition: .when(platforms: [.iOS, .macOS])
                ),
                .product(
                    name: "MLXNN",
                    package: "mlx-swift",
                    condition: .when(platforms: [.iOS, .macOS])
                ),
                .product(
                    name: "MLXLMCommon",
                    package: "mlx-swift-lm",
                    condition: .when(platforms: [.iOS, .macOS])
                ),
                .product(
                    name: "MLXLLM",
                    package: "mlx-swift-lm",
                    condition: .when(platforms: [.iOS, .macOS])
                ),
                .product(
                    name: "ZIPFoundation",
                    package: "ZIPFoundation",
                    condition: .when(platforms: [.iOS, .macOS])
                ),
            ],
            path: "swift/OpenMedKit/Sources/OpenMedKit",
            resources: [
                .process("Resources")
            ]
        ),
        .target(
            name: "OpenMedExtensionSupport",
            dependencies: ["OpenMedKit"],
            path: "swift/OpenMedKit/Sources/OpenMedExtensionSupport"
        ),
        .target(
            name: "OpenMedShareExtension",
            dependencies: ["OpenMedExtensionSupport", "OpenMedKit"],
            path: "swift/OpenMedKit/Sources/ShareExtension"
        ),
        .target(
            name: "OpenMedActionExtension",
            dependencies: ["OpenMedExtensionSupport"],
            path: "swift/OpenMedKit/Sources/ActionExtension"
        ),
        .testTarget(
            name: "OpenMedKitTests",
            dependencies: ["OpenMedKit"],
            path: "swift/OpenMedKit/Tests/OpenMedKitTests"
        ),
        .testTarget(
            name: "ExtensionTests",
            dependencies: [
                "OpenMedKit",
                "OpenMedExtensionSupport",
                "OpenMedShareExtension",
                "OpenMedActionExtension",
            ],
            path: "swift/OpenMedKit/Tests/ExtensionTests"
        ),
    ]
)
