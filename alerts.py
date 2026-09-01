# alerts.py
"""
Health Alert System
Generates health recommendations based on AQI levels
"""

# Health Alert Configuration
ALERT_THRESHOLDS = {
    0: {
        'name': 'Good',
        'color': 'green',
        'description': 'Air quality is satisfactory.',
        'advice': 'Outdoor activities are safe for everyone.',
        'sensitive_advice': 'No restrictions needed.',
    },
    1: {
        'name': 'Moderate',
        'color': 'yellow',
        'description': 'Air quality is acceptable.',
        'advice': 'Normal outdoor activities.',
        'sensitive_advice': 'Sensitive individuals should limit prolonged outdoor exertion.',
    },
    2: {
        'name': 'Poor',
        'color': 'orange',
        'description': 'Air quality is poor.',
        'advice': 'Everyone may experience some health effects. Limit outdoor activities.',
        'sensitive_advice': 'Children and elderly should avoid prolonged outdoor exertion.',
    },
    3: {
        'name': 'Very Poor',
        'color': 'red',
        'description': 'Air quality is very poor.',
        'advice': 'Avoid outdoor activities. Keep windows closed.',
        'sensitive_advice': 'Stay indoors. Use air purifiers if available.',
    },
    4: {
        'name': 'Severe',
        'color': 'purple',
        'description': 'Air quality is severe.',
        'advice': 'Emergency warning. Stay indoors.',
        'sensitive_advice': 'Wear N95 mask if going outside is absolutely necessary.',
    },
    5: {
        'name': 'Hazardous',
        'color': 'maroon',
        'description': 'Air quality is hazardous.',
        'advice': 'Health emergency. Stay indoors with windows closed.',
        'sensitive_advice': 'Seek medical attention if experiencing health effects.',
    },
}

# Disease Risk Mapping
DISEASE_RISK_MAP = {
    'asthma': {
        'threshold': 100,
        'symptoms': 'Wheezing, shortness of breath, chest tightness',
        'prevention': 'Keep windows closed, use air purifier, avoid outdoor exercise',
    },
    'bronchitis': {
        'threshold': 150,
        'symptoms': 'Persistent cough, mucus production, chest discomfort',
        'prevention': 'Wear mask outdoors, avoid pollution sources',
    },
    'copd': {
        'threshold': 100,
        'symptoms': 'Difficulty breathing, chronic cough, fatigue',
        'prevention': 'Stay indoors, use humidifier',
    },
    'lung_infection': {
        'threshold': 200,
        'symptoms': 'Fever, cough, breathing difficulty',
        'prevention': 'Avoid crowds, wear mask, maintain hygiene',
    },
    'eye_irritation': {
        'threshold': 100,
        'symptoms': 'Redness, itching, watering',
        'prevention': 'Wear protective glasses outdoors',
    },
    'skin_allergy': {
        'threshold': 150,
        'symptoms': 'Rash, itching, dryness',
        'prevention': 'Moisturize skin, avoid direct contact with outdoor air',
    },
}


def get_alert(aqi, include_diseases=False):
    """Get health alert for given AQI"""
    category = get_category(aqi)
    alert = ALERT_THRESHOLDS.get(category, ALERT_THRESHOLDS[5]).copy()
    alert['category'] = category
    alert['aqi'] = aqi
    
    if include_diseases:
        alert['diseases'] = get_disease_risks(aqi)
    
    return alert


def get_category(aqi):
    """Get AQI category number"""
    if aqi <= 50:
        return 0
    elif aqi <= 100:
        return 1
    elif aqi <= 200:
        return 2
    elif aqi <= 300:
        return 3
    elif aqi <= 400:
        return 4
    else:
        return 5


def get_health_message(aqi, user_type='general'):
    """Get personalized health message"""
    alert = get_alert(aqi)
    
    if user_type == 'sensitive':
        advice = alert['sensitive_advice']
    else:
        advice = alert['advice']
    
    return {
        'status': alert['name'],
        'color': alert['color'],
        'description': alert['description'],
        'advice': advice,
        'aqi': aqi
    }


def get_disease_risks(aqi):
    """Get disease risks based on AQI"""
    risks = []
    
    for disease, info in DISEASE_RISK_MAP.items():
        if aqi >= info['threshold']:
            risks.append({
                'disease': disease,
                'risk_level': 'High' if aqi >= info['threshold'] * 1.5 else 'Moderate',
                'symptoms': info['symptoms'],
                'prevention': info['prevention'],
            })
    
    return risks


def get_preventive_measures(aqi):
    """Get preventive measures based on AQI"""
    measures = []
    
    if aqi > 50:
        measures.append("Keep windows closed")
    if aqi > 100:
        measures.append("Use air purifier if available")
    if aqi > 150:
        measures.append("Wear N95 mask outdoors")
    if aqi > 200:
        measures.append("Avoid all outdoor activities")
    if aqi > 300:
        measures.append("Seal windows with tape if necessary")
    
    if not measures:
        measures.append("Normal activities safe")
    
    return measures


def get_alert_color(aqi):
    """Get alert color for UI"""
    category = get_category(aqi)
    return ALERT_THRESHOLDS[category]['color']


def format_alert_message(aqi, prediction=None):
    """Format alert message for display"""
    alert = get_alert(aqi)
    
    msg = f"🛡️ **AQI {int(aqi)}** ({alert['name']})\n\n"
    msg += f"{alert['description']}\n\n"
    msg += f"💡 **Recommendation:** {alert['advice']}\n"
    
    if prediction:
        if prediction > aqi:
            msg += f"\n⚠️ **Warning:** AQI expected to rise to {int(prediction)}"
        elif prediction < aqi:
            msg += f"\n✅ **Outlook:** AQI expected to improve to {int(prediction)}"
    
    return msg


class AlertManager:
    """Manage health alerts"""
    
    def __init__(self):
        self.alerts = []
        self.max_alerts = 20
    
    def add_alert(self, aqi, source='sensor'):
        """Add an alert"""
        alert = {
            'aqi': aqi,
            'category': get_category(aqi),
            'source': source,
            'message': get_alert(aqi)
        }
        self.alerts.append(alert)
        
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)
    
    def get_recent_alerts(self, limit=5):
        """Get recent alerts"""
        return self.alerts[-limit:]
    
    def get_critical_alerts(self):
        """Get critical alerts (category >= 3)"""
        return [a for a in self.alerts if a['category'] >= 3]
    
    def is_critical(self):
        """Check if current state is critical"""
        if not self.alerts:
            return False
        return self.alerts[-1]['category'] >= 3