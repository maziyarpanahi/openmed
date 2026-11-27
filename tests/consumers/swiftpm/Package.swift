// swift-tools-version: 5.10

import PackageDescription

// Installed-package consumer for OpenMedKit.
//
// scripts/release/verify_installed_consumers.py stages a copy of the published
// OpenMedKit package beside this directory and, in git mode, rewrites the
// dependency below to a tagged `file://` repository. Nothing here imports
// repository sources by relative path.
let package = Package(
    name: "OpenMedKitInstalledConsumer",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    dependencies: [
        .package(path: "../OpenMedKit")  // openmedkit-dependency
    ],
    targets: [
        .executableTarget(
            name: "OpenMedKitInstalledConsumer",
            dependencies: [
                .product(name: "OpenMedKit", package: "OpenMedKit")
            ]
        )
    ]
)
