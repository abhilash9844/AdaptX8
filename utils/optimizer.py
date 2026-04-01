"""AI Optimizer - Analysis Module"""

from typing import Dict
import numpy as np


class AIOptimizer:
    """Provides analysis and suggestions."""
    
    def analyze_state(self, state: Dict, predicted_efficiency: float) -> Dict:
        servers = state.get('servers', 10)
        workload = state.get('workload', 50)
        current_eff = state.get('current_efficiency', 50)
        temps = state.get('server_temperatures', [35] * 10)
        server_states = state.get('server_states', [True] * 10)
        max_temp = state.get('max_temperature', 35)
        
        analysis = {
            'efficiency_status': '',
            'temp_status': '',
            'load_status': '',
            'suggestions': [],
            'critical_count': 0
        }
        
        # Efficiency
        if current_eff >= 82:
            analysis['efficiency_status'] = '🌟 Excellent'
        elif current_eff >= 70:
            analysis['efficiency_status'] = '✅ Good'
        elif current_eff >= 55:
            analysis['efficiency_status'] = '📊 Average'
        else:
            analysis['efficiency_status'] = '⚠️ Needs Work'
        
        # Temperature
        critical = sum(1 for i, t in enumerate(temps) if server_states[i] and t > 55)
        hot = sum(1 for i, t in enumerate(temps) if server_states[i] and 45 <= t <= 55)
        analysis['critical_count'] = critical
        
        if critical > 0:
            analysis['temp_status'] = f'🔴 {critical} Critical'
        elif hot > 0:
            analysis['temp_status'] = f'🟠 {hot} Hot'
        elif max_temp > 38:
            analysis['temp_status'] = '🟡 Warm'
        else:
            analysis['temp_status'] = '🟢 Normal'
        
        # Load
        load_per_server = workload / servers if servers > 0 else 0
        if load_per_server > 75:
            analysis['load_status'] = '🔴 Overloaded'
        elif load_per_server > 60:
            analysis['load_status'] = '🟠 High'
        elif load_per_server >= 35:
            analysis['load_status'] = '🟢 Optimal'
        else:
            analysis['load_status'] = '🔵 Light'
        
        # Suggestions
        if critical > 0:
            analysis['suggestions'].append(f"🚨 {critical} server(s) critical!")
        if load_per_server < 30 and servers > 2:
            analysis['suggestions'].append("💤 Turn off idle servers")
        if load_per_server > 75:
            analysis['suggestions'].append("📈 Add more servers")
        if max_temp < 30 and servers > 0:
            analysis['suggestions'].append("💚 Reduce fans to save energy")
        
        if not analysis['suggestions']:
            analysis['suggestions'].append("✅ System optimal")
        
        return analysis