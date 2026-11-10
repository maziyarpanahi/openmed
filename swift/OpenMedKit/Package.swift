// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "OpenMedKit",
    defaultLocalization: "en",
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
        // swift-transformers for HuggingFace-compatible tokenization
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.0.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6"),
        // Includes upstream #419, which preserves multimodal RoPE state from
        // prefill into autoregressive decode. Pin until the next tagged release.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            revision: "42f08a872075fd07f9f1f40ec1a5e191e6aad86e"
        ),
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.19"),
    ],
    targets: [
        .target(
            name: "OpenMedSpeech",
            dependencies: ["OpenMedKit", .product(name: "WhisperKit", package: "argmax-oss-swift")],
            path: "Sources/OpenMedSpeech"
        ),
        .testTarget(
            name: "OpenMedSpeechTests",
            dependencies: ["OpenMedSpeech", "OpenMedKit"],
            path: "Tests/OpenMedSpeechTests"
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
            resources: [
                .process("Resources")
            ]
        ),
        .target(
            name: "OpenMedExtensionSupport",
            dependencies: ["OpenMedKit"],
            resources: [
                .process("Resources")
            ]
        ),
        .target(
            name: "OpenMedShareExtension",
            dependencies: ["OpenMedExtensionSupport", "OpenMedKit"],
            path: "Sources/ShareExtension",
            resources: [
                .process("Resources")
            ]
        ),
        .target(
            name: "OpenMedActionExtension",
            dependencies: ["OpenMedExtensionSupport"],
            path: "Sources/ActionExtension",
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "OpenMedKitTests",
            dependencies: ["OpenMedKit"]
        ),
        .testTarget(
            name: "ExtensionTests",
            dependencies: [
                "OpenMedKit",
                "OpenMedExtensionSupport",
                "OpenMedShareExtension",
                "OpenMedActionExtension",
            ]
        ),
    ]
)
