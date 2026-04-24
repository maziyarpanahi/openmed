import Foundation
import Combine
import os.log

/// Read/write store for the saved clinical label presets.
/// Built-in presets are hard-coded; user presets persist to UserDefaults.
@MainActor
public final class ClinicalPresetsStore: ObservableObject {
    @Published public private(set) var userPresets: [ClinicalPreset] = []
    @Published public var selectedID: UUID

    public let defaults: UserDefaults
    private let storageKey: String
    private let log = Logger(subsystem: "com.openmed.scan", category: "presets")
    public static let maxUserPresets = 20

    public init(
        defaults: UserDefaults = .standard,
        storageKey: String = "com.openmed.clinical-presets",
        selectedID: UUID = ClinicalPreset.builtInSummary.id
    ) {
        self.defaults = defaults
        self.storageKey = storageKey
        self.selectedID = selectedID
        self.userPresets = loadUserPresets()
    }

    public var allPresets: [ClinicalPreset] {
        ClinicalPreset.builtIns + userPresets
    }

    public var selectedPreset: ClinicalPreset {
        allPresets.first(where: { $0.id == selectedID }) ?? .builtInSummary
    }

    public func select(_ preset: ClinicalPreset) {
        selectedID = preset.id
    }

    /// Updates the labels on the currently selected preset. If the selection
    /// points at a built-in, an auto-named user preset is created and selected
    /// so built-ins are never mutated.
    public func updateLabelsOnSelected(to newLabels: [String]) {
        let normalized = ClinicalPreset.normalize(newLabels)
        let current = selectedPreset
        if current.isBuiltIn {
            guard userPresets.count < Self.maxUserPresets else { return }
            let fork = ClinicalPreset(
                name: "\(current.name) (edited)",
                summary: current.summary,
                labels: normalized,
                isBuiltIn: false
            )
            userPresets.append(fork)
            selectedID = fork.id
            persist()
        } else {
            guard let idx = userPresets.firstIndex(where: { $0.id == current.id }) else { return }
            userPresets[idx].labels = normalized
            persist()
        }
    }

    /// Saves the given labels as a new user preset with the provided name.
    /// Returns the new preset so callers can immediately select it.
    @discardableResult
    public func saveAsNew(name: String, labels: [String]) -> ClinicalPreset? {
        guard userPresets.count < Self.maxUserPresets else { return nil }
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else { return nil }
        let normalized = ClinicalPreset.normalize(labels)
        guard !normalized.isEmpty else { return nil }
        let preset = ClinicalPreset(
            name: trimmedName,
            labels: normalized,
            isBuiltIn: false
        )
        userPresets.append(preset)
        selectedID = preset.id
        persist()
        return preset
    }

    public func delete(_ preset: ClinicalPreset) {
        guard !preset.isBuiltIn else { return }
        userPresets.removeAll { $0.id == preset.id }
        if selectedID == preset.id {
            selectedID = ClinicalPreset.builtInSummary.id
        }
        persist()
    }

    public func rename(_ preset: ClinicalPreset, to newName: String) {
        guard !preset.isBuiltIn else { return }
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let idx = userPresets.firstIndex(where: { $0.id == preset.id }) else { return }
        userPresets[idx].name = trimmed
        persist()
    }

    // MARK: - Persistence

    private func persist() {
        do {
            let data = try JSONEncoder().encode(userPresets)
            defaults.set(data, forKey: storageKey)
        } catch {
            log.error("Failed to encode clinical presets: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func loadUserPresets() -> [ClinicalPreset] {
        guard let data = defaults.data(forKey: storageKey) else { return [] }
        do {
            return try JSONDecoder().decode([ClinicalPreset].self, from: data)
        } catch {
            log.error("Corrupt preset store, resetting: \(error.localizedDescription, privacy: .public)")
            defaults.removeObject(forKey: storageKey)
            return []
        }
    }
}
