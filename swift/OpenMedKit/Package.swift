// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "OpenMedKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "OpenMedKit",
            targets: ["OpenMedKit"]
        ),
    ],
    dependencies: [
        // swift-transformers for HuggingFace-compatible tokenization
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.12"),
    ],
    targets: [
        .target(
            name: "OpenMedKit",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "OpenMedKitTests",
            dependencies: ["OpenMedKit"]
        ),
    ]
)
