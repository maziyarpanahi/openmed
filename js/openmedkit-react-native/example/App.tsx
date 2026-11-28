import React, { useEffect, useState } from "react";
import { NativeModules, SafeAreaView, ScrollView, StyleSheet, Text, View } from "react-native";
import * as OpenMedKit from "@openmed/openmedkit-react-native";

import { CORPUS_MANIFEST, CORPUS_REGISTRY } from "./src/parity/corpusRegistry";
import { resolveNativeProbe } from "./src/parity/probe";
import { runParity, type ParityReport } from "./src/parity/runner";

type HarnessState =
  | { phase: "running" }
  | { phase: "done"; report: ParityReport; written: boolean }
  | { phase: "unavailable"; reason: string };

/**
 * In-app parity harness.
 *
 * On launch it replays the shared corpora through the real `OpenMedKitRN`
 * bridge and through the direct OpenMedKit probe, writes the value-free report
 * to app storage, and logs one `OPENMED_RN_PARITY` line for CI. The screen shows
 * counts and outcomes only, never corpus text.
 */
export default function App(): React.JSX.Element {
  const [state, setState] = useState<HarnessState>({ phase: "running" });

  useEffect(() => {
    let active = true;
    void (async () => {
      const probe = resolveNativeProbe(NativeModules as Record<string, unknown>);
      if (!probe) {
        console.log("OPENMED_RN_PARITY status=fail reason=parity_probe_unavailable");
        if (active) {
          setState({ phase: "unavailable", reason: "parity_probe_unavailable" });
        }
        return;
      }
      const report = await runParity({
        bridge: OpenMedKit,
        probe,
        manifest: CORPUS_MANIFEST,
        registry: CORPUS_REGISTRY,
        jsMode: __DEV__ ? "development" : "release",
      });
      let written = false;
      try {
        written = await probe.writeReport(JSON.stringify(report));
      } catch {
        written = false;
      }
      console.log(
        `OPENMED_RN_PARITY status=${report.status} eligible=${String(
          report.evidence_eligible,
        )} written=${String(written)} digest=${report.report_digest ?? "none"}`,
      );
      if (active) {
        setState({ phase: "done", report, written });
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>OpenMedKit bridge parity</Text>
        <Text testID="openmed-parity-status" style={styles.status}>
          {state.phase === "done" ? state.report.status : state.phase}
        </Text>
        {state.phase === "unavailable" ? (
          <Text style={styles.row}>{state.reason}</Text>
        ) : null}
        {state.phase === "done" ? <ReportSummary report={state.report} written={state.written} /> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function ReportSummary({
  report,
  written,
}: {
  report: ParityReport;
  written: boolean;
}): React.JSX.Element {
  return (
    <View>
      <Text style={styles.row}>
        {report.platform} / {report.runtime_backend} / {report.device_class}
      </Text>
      <Text style={styles.row}>
        cases {report.cases.matched}/{report.cases.total} matched
      </Text>
      <Text style={styles.row}>bridge {report.bridge_source}</Text>
      <Text style={styles.row}>evidence eligible {String(report.evidence_eligible)}</Text>
      <Text style={styles.row}>report written {String(written)}</Text>
      {report.failure_reasons.map((reason) => (
        <Text key={reason} style={styles.failure}>
          {reason}
        </Text>
      ))}
      {report.lifecycle.map((check) => (
        <Text key={check.check} style={styles.row}>
          {check.check}: {check.outcome}
        </Text>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: "#ffffff" },
  content: { padding: 16 },
  title: { fontSize: 20, fontWeight: "600", color: "#111111" },
  status: { fontSize: 28, fontWeight: "700", marginVertical: 12, color: "#111111" },
  row: { fontSize: 14, marginBottom: 4, color: "#333333" },
  failure: { fontSize: 14, marginBottom: 4, color: "#a40000" },
});
