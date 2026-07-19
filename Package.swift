// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "OpenMedKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
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
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
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
            name: "ShareExtension",
            dependencies: ["OpenMedExtensionSupport"],
            path: "swift/OpenMedKit/Sources/ShareExtension"
        ),
        .target(
            name: "ActionExtension",
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
                "ShareExtension",
                "ActionExtension",
            ],
            path: "swift/OpenMedKit/Tests/ExtensionTests"
        ),
    ]
)
