import React, { useMemo, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import {
  deidentify,
  loadModel,
  type OpenMedBackendKind,
  type OpenMedDeidentifyResult,
} from "../../js/openmedkit-react-native/src";

const SYNTHETIC_NOTE =
  "Patient Maya Shah was born on 1984-02-14. Email maya@example.org.";

export interface RedactScreenProps {
  modelPath: string;
  backend?: OpenMedBackendKind;
  policy?: string;
}

export function RedactScreen({
  modelPath,
  backend = "mlx",
  policy = "hipaa_safe_harbor",
}: RedactScreenProps) {
  const [result, setResult] = useState<OpenMedDeidentifyResult | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "failed">("idle");
  const redactedText = useMemo(
    () => result?.deidentifiedText ?? SYNTHETIC_NOTE,
    [result],
  );

  async function runRedaction() {
    setStatus("running");
    setResult(null);
    try {
      await loadModel({
        modelPath,
        backend,
        cacheKey: `${backend}:${modelPath}`,
      });
      const nextResult = await deidentify(SYNTHETIC_NOTE, {
        policy,
        docId: "rn-example",
        detector: "openmedkit-react-native",
      });
      setResult(nextResult);
      setStatus("idle");
    } catch {
      setStatus("failed");
    }
  }

  return (
    <View style={styles.screen}>
      <View style={styles.header}>
        <Text style={styles.title}>OpenMedKit redaction</Text>
        <Text style={styles.count}>{result?.spans.length ?? 0} spans</Text>
      </View>
      <View style={styles.panel}>
        <Text style={styles.label}>Synthetic note</Text>
        <Text style={styles.note}>{SYNTHETIC_NOTE}</Text>
      </View>
      <View style={styles.panel}>
        <Text style={styles.label}>Redacted output</Text>
        <Text style={styles.note}>{redactedText}</Text>
      </View>
      <Pressable
        accessibilityRole="button"
        disabled={status === "running"}
        onPress={runRedaction}
        style={({ pressed }) => [
          styles.button,
          pressed && status !== "running" ? styles.buttonPressed : null,
          status === "running" ? styles.buttonDisabled : null,
        ]}
      >
        <Text style={styles.buttonText}>
          {status === "running" ? "Redacting" : "Run redaction"}
        </Text>
      </Pressable>
      {status === "failed" ? (
        <Text style={styles.error}>Native model configuration failed.</Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    gap: 16,
    padding: 20,
    backgroundColor: "#f7f8fa",
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  title: {
    color: "#111827",
    fontSize: 22,
    fontWeight: "700",
  },
  count: {
    color: "#374151",
    fontSize: 14,
    fontWeight: "600",
  },
  panel: {
    gap: 8,
    padding: 14,
    borderRadius: 8,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#d1d5db",
    backgroundColor: "#ffffff",
  },
  label: {
    color: "#4b5563",
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  note: {
    color: "#111827",
    fontSize: 16,
    lineHeight: 23,
  },
  button: {
    alignItems: "center",
    justifyContent: "center",
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: "#0f766e",
  },
  buttonPressed: {
    backgroundColor: "#115e59",
  },
  buttonDisabled: {
    backgroundColor: "#6b7280",
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "700",
  },
  error: {
    color: "#b91c1c",
    fontSize: 14,
  },
});
