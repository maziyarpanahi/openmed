// swift-tools-version: 5.9

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
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            from: "0.1.12"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.31.3"
        ),
        .package(
            url: "https://github.com/weichsel/ZIPFoundation.git",
            from: "0.9.19"
        ),
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
            path: "swift/OpenMedKit/Sources/OpenMedKit",
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "OpenMedKitTests",
            dependencies: ["OpenMedKit"],
            path: "swift/OpenMedKit/Tests/OpenMedKitTests"
        ),
    ]
)
