package com.openmed.openmedkit.decode

/**
 * Score aggregation strategy for multi-token entity spans.
 */
enum class AggregationStrategy {
    FIRST,
    AVERAGE,
    MAX,
}
