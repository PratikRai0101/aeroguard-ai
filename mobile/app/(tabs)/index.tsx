// app/(tabs)/index.tsx
import { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  ActivityIndicator,
} from "react-native";
import { getStats, getReadings, DashboardStats, Reading } from "../../lib/api";

export default function DashboardScreen() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [readings, setReadings] = useState<Reading[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadData() {
    try {
      setError(null);
      const [statsData, readingsData] = await Promise.all([
        getStats(),
        getReadings(20),
      ]);
      setStats(statsData);
      setReadings(readingsData);
    } catch (err: any) {
      setError(err.message || "Failed to load dashboard data");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#00A699" />
        <Text style={styles.loadingText}>Loading AeroGuard AI...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>⚠️ {error}</Text>
        <Text style={styles.hintText}>
          Make sure the backend is running at the configured API URL.
        </Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.title}>🛡️ AeroGuard AI</Text>
        <Text style={styles.subtitle}>Real-time air quality monitoring</Text>
      </View>

      {stats && (
        <>
          <View style={[styles.card, { borderLeftColor: stats.color || "#00E400" }]}>
            <Text style={styles.label}>Indoor Air Quality</Text>
            <Text style={styles.aqiValue}>
              {stats.emoji} {stats.aqi}
            </Text>
            <Text style={[styles.statusText, { color: stats.color || "#000" }]}>
              {stats.status}
            </Text>
            <Text style={styles.adviceText}>{stats.advice}</Text>
          </View>

          <View style={styles.row}>
            <View style={[styles.card, styles.smallCard]}>
              <Text style={styles.label}>Temperature</Text>
              <Text style={styles.metricValue}>{stats.temp}°C</Text>
            </View>
            <View style={[styles.card, styles.smallCard]}>
              <Text style={styles.label}>Humidity</Text>
              <Text style={styles.metricValue}>{stats.hum}%</Text>
            </View>
          </View>

          <View style={styles.row}>
            <View style={[styles.card, styles.smallCard]}>
              <Text style={styles.label}>Gas/VOC</Text>
              <Text style={styles.metricValue}>{stats.gas}</Text>
            </View>
            <View style={[styles.card, styles.smallCard]}>
              <Text style={styles.label}>Mode</Text>
              <Text style={styles.metricValue}>{stats.mode}</Text>
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.label}>Trend</Text>
            <Text style={styles.metricValue}>
              {stats.trend === "rising" ? "📈 Rising" : stats.trend === "falling" ? "📉 Falling" : "➡️ Stable"}
            </Text>
          </View>

          {stats.outdoor_aqi !== null && stats.outdoor_aqi !== undefined && (
            <View style={styles.card}>
              <Text style={styles.label}>Outdoor AQI (Nagpur)</Text>
              <Text style={styles.metricValue}>{stats.outdoor_aqi}</Text>
              {stats.outdoor_pm25 !== null && (
                <Text style={styles.adviceText}>
                  PM2.5: {stats.outdoor_pm25} µg/m³
                </Text>
              )}
            </View>
          )}

          <View style={styles.card}>
            <Text style={styles.label}>Preventive Measures</Text>
            {stats.preventive_measures.map((measure, index) => (
              <Text key={index} style={styles.bulletPoint}>
                • {measure}
              </Text>
            ))}
          </View>

          <View style={styles.card}>
            <Text style={styles.label}>Recent Readings ({readings.length})</Text>
            {readings.slice(-5).map((reading, index) => (
              <Text key={index} style={styles.readingText}>
                {reading.time}: AQI {reading.aqi}, T {reading.temp}°C, H {reading.hum}%
              </Text>
            ))}
          </View>
        </>
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
    fontSize: 24,
    fontWeight: "bold",
    color: "#fff",
  },
  subtitle: {
    fontSize: 14,
    color: "#aaa",
    marginTop: 4,
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
  smallCard: {
    flex: 1,
    marginHorizontal: 8,
  },
  row: {
    flexDirection: "row",
    marginHorizontal: 8,
  },
  label: {
    fontSize: 12,
    color: "#666",
    textTransform: "uppercase",
    fontWeight: "600",
    marginBottom: 4,
  },
  aqiValue: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#333",
  },
  statusText: {
    fontSize: 18,
    fontWeight: "600",
    marginTop: 2,
  },
  metricValue: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#333",
  },
  adviceText: {
    fontSize: 14,
    color: "#555",
    marginTop: 6,
  },
  bulletPoint: {
    fontSize: 14,
    color: "#555",
    marginTop: 4,
  },
  readingText: {
    fontSize: 13,
    color: "#666",
    marginTop: 3,
  },
});
