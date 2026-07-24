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
        )
    ],
    dependencies: [
        // swift-transformers for HuggingFace-compatible tokenization
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.12"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.3"),
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
                    name: "ZIPFoundation",
                    package: "ZIPFoundation",
                    condition: .when(platforms: [.iOS, .macOS])
                ),
            ],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "OpenMedKitTests",
            dependencies: ["OpenMedKit"]
        ),
    ]
)
