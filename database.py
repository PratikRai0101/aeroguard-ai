# database.py
"""
SQLite Database Module for AeroGuard AI
Persists sensor readings and model predictions
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_FILE = 'aeroguard.db'


class SensorDatabase:
    """SQLite database for sensor data persistence"""
    
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        # Sensor readings table
        c.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                temp REAL,
                hum REAL,
                gas REAL,
                aqi REAL,
                status TEXT,
                source TEXT DEFAULT 'sensor'
            )
        ''')
        
        # Predictions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rf_status TEXT,
                rf_confidence REAL,
                lstm_status TEXT,
                lstm_confidence REAL,
                trend TEXT,
                aqi_predicted REAL
            )
        ''')
        
        # Alerts table
        c.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                aqi REAL,
                category TEXT,
                message TEXT,
                severity INTEGER
            )
        ''')
        
        # Create indexes
        c.execute('CREATE INDEX IF NOT EXISTS idx_readings_time ON readings(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(timestamp)')
        
        conn.commit()
        conn.close()
    
    def add_reading(self, temp, hum, gas, aqi, status, source='sensor'):
        """Add a sensor reading"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute('''
            INSERT INTO readings (timestamp, temp, hum, gas, aqi, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, temp, hum, gas, aqi, status, source))
        
        conn.commit()
        conn.close()
    
    def add_prediction(self, rf_status, rf_conf, lstm_status, lstm_conf, trend, aqi_pred):
        """Add a prediction"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute('''
            INSERT INTO predictions (timestamp, rf_status, rf_confidence, lstm_status, lstm_confidence, trend, aqi_predicted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, rf_status, rf_conf, lstm_status, lstm_conf, trend, aqi_pred))
        
        conn.commit()
        conn.close()
    
    def add_alert(self, aqi, category, message, severity):
        """Add an alert"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute('''
            INSERT INTO alerts (timestamp, aqi, category, message, severity)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, aqi, category, message, severity))
        
        conn.commit()
        conn.close()
    
    def get_recent_readings(self, limit=100, include_source=False):
        """Get the last N readings, optionally including their data source."""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        columns = "timestamp, temp, hum, gas, aqi, status"
        if include_source:
            columns += ", source"
        c.execute(f'''
            SELECT {columns}
            FROM readings
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        
        rows = c.fetchall()
        conn.close()
        
        return rows
    
    def get_readings_for_chart(self, limit=50):
        """Get readings formatted for charting"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        c.execute('''
            SELECT timestamp, temp, hum, gas, aqi 
            FROM readings 
            ORDER BY id DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = c.fetchall()
        conn.close()
        
        # Reverse to get chronological order
        return list(reversed(rows))
    
    def get_statistics(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM readings')
        reading_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM predictions')
        pred_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM alerts')
        alert_count = c.fetchone()[0]
        
        if reading_count > 0:
            c.execute('SELECT MIN(temp), MAX(temp), AVG(temp) FROM readings')
            temp_min, temp_max, temp_avg = c.fetchone()
            
            c.execute('SELECT MIN(aqi), MAX(aqi), AVG(aqi) FROM readings')
            aqi_min, aqi_max, aqi_avg = c.fetchone()
        else:
            temp_min = temp_max = temp_avg = 0
            aqi_min = aqi_max = aqi_avg = 0
        
        conn.close()
        
        return {
            'readings': reading_count,
            'predictions': pred_count,
            'alerts': alert_count,
            'temp_range': (temp_min, temp_max, temp_avg),
            'aqi_range': (aqi_min, aqi_max, aqi_avg)
        }
    
    def clear_old_readings(self, days=7):
        """Clear readings older than N days"""
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        
        c.execute('''
            DELETE FROM readings 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        deleted = c.rowcount
        conn.commit()
        conn.close()
        
        return deleted


def init_db(db_file=DB_FILE):
    """Initialize database"""
    return SensorDatabase(db_file)


def get_readings(limit=100, db_file=DB_FILE):
    """Get recent readings (convenience function)"""
    db = SensorDatabase(db_file)
    return db.get_recent_readings(limit)


if __name__ == '__main__':
    # Test database
    db = SensorDatabase('test.db')
    
    # Add sample reading
    db.add_reading(25.5, 55.0, 150.0, 75.0, 'moderate')
    db.add_prediction('moderate', 0.85, 'good', 0.72, 'stable', 70.0)
    db.add_alert(75.0, 'moderate', 'Air quality is acceptable', 1)
    
    # Get stats
    stats = db.get_statistics()
    print(f"Database stats: {stats}")
    
    # Get recent
    readings = db.get_recent_readings(5)
    print(f"Recent readings: {readings}")