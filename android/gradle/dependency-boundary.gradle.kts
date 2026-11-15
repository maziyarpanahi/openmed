// Consumer dependency-boundary check shared by the OpenMedKit Android artifacts.
//
// A consumer that adds an AAR inherits the AAR's runtime dependency graph and
// every manifest entry those dependencies declare: the app's manifest merger
// adds them unless the app explicitly removes them. This check scans what a
// consumer inherits from the release variant:
//   * the resolved release runtime classpath,
//   * the generated release POM, and
//   * the module manifest plus every dependency manifest on that classpath.
//
// Apply it after setting `extra["openmedkitDependencyBoundary"]`:
//   * "core": fail on any vendor OCR/telemetry artifact, network permission, or
//     dependency-contributed provider, service, or receiver.
//   * "optional-vendor": the negative control for the optional ML Kit module;
//     fail unless the scan detects those findings, which proves the scanner
//     sees the classpath, the POM, and dependency manifests.
//
// Reports list Maven coordinates, permission names, and component class names
// only; they contain no file paths or document content.

import javax.xml.parsers.DocumentBuilderFactory
import org.gradle.api.artifacts.component.ModuleComponentIdentifier
import org.gradle.api.artifacts.component.ProjectComponentIdentifier
import org.gradle.api.attributes.Attribute
import org.w3c.dom.Element

val boundaryMode = extra.properties["openmedkitDependencyBoundary"] as? String
    ?: throw GradleException(
        "Set extra[\"openmedkitDependencyBoundary\"] before applying " +
            "dependency-boundary.gradle.kts.",
    )
if (boundaryMode != "core" && boundaryMode != "optional-vendor") {
    throw GradleException(
        "openmedkitDependencyBoundary must be \"core\" or \"optional-vendor\", " +
            "got \"$boundaryMode\".",
    )
}

val forbiddenGroups = listOf(
    "com.google.mlkit",
    "com.google.android.gms",
    "com.google.android.datatransport",
    "com.google.firebase",
)
val networkPermissions = setOf(
    "android.permission.INTERNET",
    "android.permission.ACCESS_NETWORK_STATE",
    "android.permission.ACCESS_WIFI_STATE",
    "android.permission.CHANGE_NETWORK_STATE",
    "android.permission.CHANGE_WIFI_STATE",
)
val autoStartingComponents = setOf("provider", "service", "receiver")
val androidNamespace = "http://schemas.android.com/apk/res/android"
val toolsNamespace = "http://schemas.android.com/tools"
val artifactTypeAttribute = Attribute.of("artifactType", String::class.java)

val boundaryClasspath = "releaseRuntimeClasspath"
val boundaryPom = layout.buildDirectory.file("publications/release/pom-default.xml")
val boundaryReport = layout.buildDirectory.file("reports/dependency-boundary/release.txt")
val moduleManifest = layout.projectDirectory.file("src/main/AndroidManifest.xml")

fun isForbiddenGroup(group: String): Boolean =
    forbiddenGroups.any { group == it || group.startsWith("$it.") }

fun parseXml(file: File): org.w3c.dom.Document {
    val factory = DocumentBuilderFactory.newInstance()
    factory.isNamespaceAware = true
    factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true)
    return factory.newDocumentBuilder().parse(file)
}

fun Element.childElements(): List<Element> {
    val children = childNodes
    return (0 until children.length).mapNotNull { children.item(it) as? Element }
}

fun Element.isRemovedByTools(): Boolean =
    getAttributeNS(toolsNamespace, "node") in setOf("remove", "removeAll")

fun resolveComponentName(manifestPackage: String, name: String): String =
    if (name.startsWith(".")) "$manifestPackage$name" else name

/** Returns `permission:` and `component:` entries a manifest contributes. */
fun manifestEntries(file: File, includeComponents: Boolean): List<String> {
    val root = parseXml(file).documentElement
    val manifestPackage = root.getAttribute("package")
    val entries = mutableListOf<String>()
    root.childElements().forEach { element ->
        val name = element.getAttributeNS(androidNamespace, "name")
        when {
            element.isRemovedByTools() -> Unit
            element.localName.startsWith("uses-permission") && name in networkPermissions ->
                entries += "permission:$name"
            element.localName == "application" && includeComponents ->
                element.childElements()
                    .filter { it.localName in autoStartingComponents && !it.isRemovedByTools() }
                    .forEach { component ->
                        val componentName = resolveComponentName(
                            manifestPackage,
                            component.getAttributeNS(androidNamespace, "name"),
                        )
                        entries += "component:${component.localName}:$componentName"
                    }
        }
    }
    return entries
}

tasks.register("verifyDependencyBoundary") {
    group = "verification"
    description = if (boundaryMode == "core") {
        "Fails when the release artifact would give consumers vendor OCR or telemetry " +
            "artifacts, network permissions, or dependency-contributed components."
    } else {
        "Negative control: fails unless the boundary scan detects the vendor stack " +
            "this optional artifact intentionally adds."
    }
    dependsOn("generatePomFileForReleasePublication")
    inputs.file(moduleManifest)
    inputs.file(boundaryPom)
    outputs.file(boundaryReport)
    // The resolved dependency graph is the real input; never skip the gate.
    outputs.upToDateWhen { false }

    doLast {
        val classpath = project.configurations.getByName(boundaryClasspath)
        val findings = sortedSetOf<String>()

        val modules = classpath.incoming.resolutionResult.allComponents
            .mapNotNull { it.id as? ModuleComponentIdentifier }
        modules.filter { isForbiddenGroup(it.group) }.forEach {
            findings += "classpath:${it.group}:${it.module}:${it.version}"
        }

        val pomFile = boundaryPom.get().asFile
        if (!pomFile.isFile) {
            throw GradleException("Release POM was not generated for ${project.path}.")
        }
        val pomDependencies = parseXml(pomFile)
            .getElementsByTagNameNS("*", "dependency")
            .let { nodes -> (0 until nodes.length).map { nodes.item(it) as Element } }
            .map { dependency ->
                fun child(tag: String): String =
                    dependency.getElementsByTagNameNS("*", tag).item(0)?.textContent?.trim()
                        .orEmpty()
                child("groupId") to child("artifactId")
            }
        pomDependencies
            .filter { (group, _) -> isForbiddenGroup(group) }
            .forEach { (group, artifact) -> findings += "pom:$group:$artifact" }

        manifestEntries(moduleManifest.asFile, includeComponents = false).forEach {
            findings += "$it <- ${project.path}"
        }
        val dependencyManifests = classpath.incoming.artifactView {
            attributes { attribute(artifactTypeAttribute, "android-manifest") }
            lenient(true)
        }.artifacts.artifacts
        dependencyManifests.forEach { artifact ->
            val owner = when (val id = artifact.id.componentIdentifier) {
                is ModuleComponentIdentifier -> "${id.group}:${id.module}:${id.version}"
                is ProjectComponentIdentifier -> id.projectPath
                else -> id.displayName
            }
            manifestEntries(artifact.file, includeComponents = true).forEach {
                findings += "$it <- $owner"
            }
        }

        val report = boundaryReport.get().asFile
        report.parentFile.mkdirs()
        report.writeText(
            buildString {
                appendLine("mode=$boundaryMode")
                appendLine("classpathModules=${modules.size}")
                appendLine("pomDependencies=${pomDependencies.size}")
                appendLine("dependencyManifests=${dependencyManifests.size}")
                appendLine("findings=${findings.size}")
                findings.forEach { appendLine("finding=$it") }
            },
        )

        if (boundaryMode == "core") {
            if (dependencyManifests.isEmpty()) {
                throw GradleException(
                    "Dependency-boundary scan for ${project.path} read no dependency " +
                        "manifests; the scan cannot prove the consumer manifest is clean.",
                )
            }
            if (findings.isNotEmpty()) {
                throw GradleException(
                    "${project.path} must not give consumers vendor OCR or telemetry " +
                        "dependencies, network permissions, or auto-starting components:\n" +
                        findings.joinToString(separator = "\n") { "  $it" },
                )
            }
            logger.lifecycle(
                "${project.path} dependency boundary: clean " +
                    "(${modules.size} modules, ${dependencyManifests.size} manifests).",
            )
        } else {
            val missing = listOf("classpath:", "pom:", "permission:", "component:")
                .filter { prefix -> findings.none { it.startsWith(prefix) } }
            if (missing.isNotEmpty()) {
                throw GradleException(
                    "Negative control for ${project.path} did not detect " +
                        "${missing.joinToString()} findings; the boundary scan is not " +
                        "seeing the optional vendor stack.",
                )
            }
            logger.lifecycle(
                "${project.path} dependency boundary: ${findings.size} expected vendor findings.",
            )
        }
    }
}

tasks.named("check") {
    dependsOn("verifyDependencyBoundary")
}
