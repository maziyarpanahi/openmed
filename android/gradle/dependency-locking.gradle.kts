// Dependency locking shared by every Android module (library and demos).
//
// Every resolvable configuration resolves only the exact versions recorded in
// the module's committed `gradle.lockfile`. STRICT mode also fails when a
// locked configuration has no lock state, so a new module or configuration
// cannot silently resolve unpinned versions. A dependency change without a
// lockfile update fails resolution, and therefore the build.
//
// Two kinds of configuration are not locked because the build never resolves
// them: Kotlin's IDE-only `*DependenciesMetadata` configurations, which cannot
// resolve BOM-managed or project dependencies in an Android-only build, and
// the legacy `default` and `archives` publishing configurations, whose
// resolution Gradle has deprecated.
//
// Artifact checksums live separately in `gradle/verification-metadata.xml`.
//
// After an intentional dependency change, regenerate both from `android/`
// (see docs/security/dependency-policy.md for the full command):
//
//   ./gradlew verifyDependencyLocks --write-locks
//   ./gradlew <build tasks> --write-verification-metadata sha256

val unlockedConfigurations = setOf("default", "archives")

fun isLockedConfiguration(name: String): Boolean =
    name !in unlockedConfigurations && !name.endsWith("DependenciesMetadata")

dependencyLocking {
    lockMode.set(LockMode.STRICT)
}

configurations.matching { isLockedConfiguration(it.name) }.configureEach {
    resolutionStrategy.activateDependencyLocking()
}

tasks.register("verifyDependencyLocks") {
    group = "verification"
    description = "Resolves every locked configuration against gradle.lockfile."
    notCompatibleWithConfigurationCache("Resolves configurations at execution time.")

    doLast {
        // Resolve each dependency graph (versions and metadata) against the
        // lock state without downloading every tool jar. Graph resolution is
        // lenient, so unresolved edges are collected and fail the task.
        val unresolved = configurations
            .filter { it.isCanBeResolved && isLockedConfiguration(it.name) }
            .flatMap { configuration ->
                configuration.incoming.resolutionResult.allDependencies
                    .filterIsInstance<UnresolvedDependencyResult>()
                    .map { "${configuration.name}: ${it.attempted.displayName}" }
            }
            .distinct()
            .sorted()
        if (unresolved.isNotEmpty()) {
            throw GradleException(
                "Dependencies did not resolve against the lock state:\n" +
                    unresolved.joinToString("\n"),
            )
        }
    }
}
