// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "OpenMedKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .watchOS(.v10),
        .visionOS(.v1),
    ],
    products: [
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
            targets: ["ShareExtension"]
        ),
        .library(
            name: "OpenMedActionExtension",
            targets: ["ActionExtension"]
        ),
    ],
    dependencies: [
        // swift-transformers for HuggingFace-compatible tokenization
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.12"),
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
            dependencies: ["OpenMedKit"]
        ),
        .target(
            name: "ShareExtension",
            dependencies: ["OpenMedExtensionSupport"]
        ),
        .target(
            name: "ActionExtension",
            dependencies: ["OpenMedExtensionSupport"]
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
                "ShareExtension",
                "ActionExtension",
            ]
        ),
    ]
)
