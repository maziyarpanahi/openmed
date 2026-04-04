// swift-tools-version: 5.9

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
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            from: "0.1.12"
        ),
    ],
    targets: [
        .target(
            name: "OpenMedKit",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/OpenMedKit/Sources/OpenMedKit"
        ),
        .testTarget(
            name: "OpenMedKitTests",
            dependencies: ["OpenMedKit"],
            path: "swift/OpenMedKit/Tests/OpenMedKitTests"
        ),
    ]
)
