import Foundation

/// Groups entities by `EntityCategory` for the summary list.
public enum EntityGrouping {
    public struct Section: Identifiable, Hashable {
        public let category: EntityCategory
        public var entities: [DetectedEntity]
        public var id: EntityCategory { category }
        public var count: Int { entities.count }
    }

    public static func group(
        _ entities: [DetectedEntity],
        filter category: EntityCategory? = nil
    ) -> [Section] {
        let filtered = category.map { target in entities.filter { $0.category == target } } ?? entities
        var buckets: [EntityCategory: [DetectedEntity]] = [:]
        for entity in filtered {
            buckets[entity.category, default: []].append(entity)
        }
        // Stable order: preserve the enum's declared order.
        return EntityCategory.allCases.compactMap { cat in
            guard let bucket = buckets[cat], !bucket.isEmpty else { return nil }
            return Section(
                category: cat,
                entities: bucket.sorted { $0.start < $1.start }
            )
        }
    }

    /// Returns `[(category, count)]` in declared order for the filter chip bar.
    public static func categoryCounts(_ entities: [DetectedEntity]) -> [(EntityCategory, Int)] {
        var counts: [EntityCategory: Int] = [:]
        for entity in entities {
            counts[entity.category, default: 0] += 1
        }
        return EntityCategory.allCases.compactMap { cat in
            counts[cat].map { (cat, $0) }
        }
    }
}
