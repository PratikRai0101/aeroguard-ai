// app/(tabs)/validation.tsx
import { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  ActivityIndicator,
} from "react-native";
import { getValidation, ValidationReport } from "../../lib/api";

const CATEGORY_NAMES = [
  "Good",
  "Moderate",
  "Poor",
  "Very Poor",
  "Severe",
  "Hazardous",
];

function format(value?: number | null, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

function percent(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export default function ValidationScreen() {
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadData() {
    try {
      setError(null);
      const data = await getValidation();
      setReport(data);
    } catch (err: any) {
      setError(err.message || "Failed to load validation report");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#00A699" />
        <Text style={styles.loadingText}>Loading validation report...</Text>
      </View>
    );
  }

  if (error || !report) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>⚠️ {error || "No validation data"}</Text>
        <Text style={styles.hintText}>
          The backend exposes this from model_metadata.json after training.
        </Text>
      </View>
    );
  }

  const dataset = report.dataset;
  const calibration = report.calibration?.metrics;
  const regression = report.lr?.validation?.regression || report.lr?.metrics;
  const categories = report.lr?.validation?.categories;
  const confusion = categories?.confusion;

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.title}>Validation vs Reference</Text>
        <Text style={styles.subtitle}>
          Measured against a certified reference analyzer
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Dataset</Text>
        <Text style={styles.row}>
          Source: {report.dataset_source?.toUpperCase() || "—"}
        </Text>
        {dataset && (
          <>
            <Text style={styles.row}>Rows: {dataset.rows}</Text>
            <Text style={styles.row}>
              Period: {dataset.start?.slice(0, 10)} → {dataset.end?.slice(0, 10)}
            </Text>
            <Text style={styles.row}>
              AQI range: {format(dataset.aqi_min, 0)} – {format(dataset.aqi_max, 0)}
            </Text>
          </>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Gas Calibration</Text>
        <Text style={styles.metric}>
          R² (log): {format(calibration?.r2_log, 3)}
        </Text>
        <Text style={styles.metric}>
          RMSE: {format(calibration?.rmse)} · MAE: {format(calibration?.mae)}
        </Text>
        <Text style={styles.note}>{report.calibration?.note}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Model Performance</Text>
        <Text style={styles.metric}>
          Random Forest accuracy: {percent(report.rf?.accuracy)}
        </Text>
        <Text style={styles.metric}>
          Linear Regression R²: {format(regression?.r2, 3)} · r:{" "}
          {format(regression?.pearson, 3)}
        </Text>
        <Text style={styles.metric}>
          MAE: {format(regression?.mae)} · RMSE: {format(regression?.rmse)}
        </Text>
        <Text style={styles.metric}>
          LSTM validation accuracy: {percent(report.lstm?.val_accuracy)}
        </Text>
        <Text style={styles.row}>Evaluated on n = {regression?.n ?? "—"} hours</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>AQI Category Agreement</Text>
        <Text style={styles.metric}>
          Exact: {percent(categories?.exact_match)} · Within one:{" "}
          {percent(categories?.within_one)}
        </Text>
      </View>

      {confusion && (
        <View style={styles.card}>
          <Text style={styles.label}>Confusion Matrix</Text>
          <Text style={styles.note}>Rows = actual, columns = predicted</Text>
          <View style={styles.matrixHeader}>
            <Text style={[styles.matrixCell, styles.matrixHeadCell]} />
            {CATEGORY_NAMES.map((name) => (
              <Text key={name} style={[styles.matrixCell, styles.matrixHeadCell]}>
                {name.slice(0, 4)}
              </Text>
            ))}
          </View>
          {confusion.map((row, rowIndex) => (
            <View key={rowIndex} style={styles.matrixHeader}>
              <Text style={[styles.matrixCell, styles.matrixHeadCell]}>
                {CATEGORY_NAMES[rowIndex]?.slice(0, 4)}
              </Text>
              {row.map((value, colIndex) => (
                <Text key={colIndex} style={styles.matrixCell}>
                  {value}
                </Text>
              ))}
            </View>
          ))}
        </View>
      )}

      {report.notes && report.notes.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.label}>Notes</Text>
          {report.notes.map((note, index) => (
            <Text key={index} style={styles.row}>
              • {note}
            </Text>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: "#666",
  },
  errorText: {
    fontSize: 16,
    color: "#d32f2f",
    textAlign: "center",
    marginBottom: 8,
  },
  hintText: {
    fontSize: 14,
    color: "#666",
    textAlign: "center",
  },
  header: {
    padding: 20,
    backgroundColor: "#0E1117",
    alignItems: "center",
  },
  title: {
    fontSize: 22,
    fontWeight: "bold",
    color: "#fff",
  },
  subtitle: {
    fontSize: 13,
    color: "#aaa",
    marginTop: 4,
    textAlign: "center",
  },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 16,
    marginVertical: 8,
    borderLeftWidth: 5,
    borderLeftColor: "#00A699",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  label: {
    fontSize: 12,
    color: "#666",
    textTransform: "uppercase",
    fontWeight: "600",
    marginBottom: 6,
  },
  metric: {
    fontSize: 16,
    color: "#333",
    fontWeight: "600",
    marginTop: 3,
  },
  row: {
    fontSize: 13,
    color: "#555",
    marginTop: 3,
  },
  note: {
    fontSize: 11,
    color: "#999",
    marginTop: 6,
    fontStyle: "italic",
  },
  matrixHeader: {
    flexDirection: "row",
  },
  matrixCell: {
    flex: 1,
    fontSize: 10,
    color: "#333",
    textAlign: "center",
    paddingVertical: 3,
  },
  matrixHeadCell: {
    fontWeight: "700",
    color: "#666",
  },
});
