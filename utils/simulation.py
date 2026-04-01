"""
Infrastructure Simulation Engine
================================
Per-server temperature with realistic thermal dynamics.
"""

import numpy as np
from typing import Dict, List


class InfrastructureSimulator:
    """Data center simulator with per-server temperature."""
    
    NUM_SERVERS = 10
    
    # Energy constants
    BASE_ENERGY_PER_SERVER = 45
    MAX_ENERGY_PER_SERVER = 120
    ENERGY_PER_FAN_LEVEL = 8
    
    # Temperature constants
    AMBIENT_TEMP = 22.0
    MAX_TEMP = 75.0
    HEAT_GENERATION_RATE = 0.35
    COOLING_PER_FAN_LEVEL = 0.7
    NATURAL_COOLING_RATE = 0.04
    THERMAL_INERTIA = 0.88
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset to initial state."""
        self.server_states = [True] * self.NUM_SERVERS
        self.fan_states = [2] * self.NUM_SERVERS
        self.server_temperatures = [30.0 + np.random.uniform(-2, 2) for _ in range(self.NUM_SERVERS)]
        self.server_cpu_loads = [45.0 + np.random.uniform(-5, 5) for _ in range(self.NUM_SERVERS)]
        
        self.workload = 50.0
        self.energy = 600.0
        self.time_step = 0
        
        self.workload_base = 50.0
        self.workload_trend = 0.0
        self.workload_phase = np.random.uniform(0, 2 * np.pi)
        self.trend_duration = 0
        
    def get_active_servers(self) -> int:
        return sum(self.server_states)
    
    def get_active_fans(self) -> int:
        return sum(1 for i, s in enumerate(self.fan_states) if s > 0 and self.server_states[i])
    
    def get_average_temperature(self) -> float:
        temps = [self.server_temperatures[i] for i in range(self.NUM_SERVERS) if self.server_states[i]]
        return sum(temps) / len(temps) if temps else self.AMBIENT_TEMP
    
    def get_max_temperature(self) -> float:
        temps = [self.server_temperatures[i] for i in range(self.NUM_SERVERS) if self.server_states[i]]
        return max(temps) if temps else self.AMBIENT_TEMP
    
    def get_cpu_usage(self) -> float:
        cpus = [self.server_cpu_loads[i] for i in range(self.NUM_SERVERS) if self.server_states[i]]
        return sum(cpus) / len(cpus) if cpus else 0.0
    
    def set_server_state(self, server_id: int, state: bool) -> bool:
        if 0 <= server_id < self.NUM_SERVERS:
            old = self.server_states[server_id]
            self.server_states[server_id] = state
            if not state:
                self.fan_states[server_id] = 0
                self.server_cpu_loads[server_id] = 0
            elif state and not old:
                self.fan_states[server_id] = 2
                self.server_cpu_loads[server_id] = 50.0
                self.server_temperatures[server_id] = 32.0
            return old != state
        return False
    
    def set_fan_state(self, server_id: int, speed: int) -> bool:
        if 0 <= server_id < self.NUM_SERVERS and 0 <= speed <= 3:
            if self.server_states[server_id] or speed == 0:
                old = self.fan_states[server_id]
                self.fan_states[server_id] = speed
                return old != speed
        return False
    
    def _generate_workload(self) -> float:
        self.workload_phase += 0.05
        if self.workload_phase > 2 * np.pi:
            self.workload_phase -= 2 * np.pi
        
        self.trend_duration -= 1
        if self.trend_duration <= 0:
            self.workload_trend = np.random.uniform(-0.5, 0.5)
            self.trend_duration = np.random.randint(40, 100)
        
        self.workload_base += self.workload_trend
        self.workload_base = np.clip(self.workload_base, 25, 85)
        
        cyclical = 12 * np.sin(self.workload_phase) + 5 * np.sin(self.workload_phase * 2.5)
        noise = np.random.normal(0, 0.8)
        
        return np.clip(self.workload_base + cyclical + noise, 10, 100)
    
    def _update_workload(self):
        target = self._generate_workload()
        self.workload = 0.92 * self.workload + 0.08 * target
        self.workload = np.clip(self.workload, 10, 100)
    
    def _distribute_workload(self):
        active = self.get_active_servers()
        if active == 0:
            return
        
        base_load = self.workload / active
        
        for i in range(self.NUM_SERVERS):
            if self.server_states[i]:
                variation = np.random.normal(0, 2)
                target = base_load * 10 + variation
                self.server_cpu_loads[i] = 0.85 * self.server_cpu_loads[i] + 0.15 * target
                self.server_cpu_loads[i] = np.clip(self.server_cpu_loads[i], 5, 100)
            else:
                self.server_cpu_loads[i] = 0
    
    def _update_server_temperature(self, server_id: int):
        if not self.server_states[server_id]:
            cooling = 0.12
            self.server_temperatures[server_id] += (self.AMBIENT_TEMP - self.server_temperatures[server_id]) * cooling
            return
        
        temp = self.server_temperatures[server_id]
        cpu = self.server_cpu_loads[server_id]
        fan = self.fan_states[server_id]
        
        # Heat generation
        heat = (cpu / 100.0) * self.HEAT_GENERATION_RATE
        if cpu > 70:
            heat += (cpu - 70) / 100.0 * 0.15
        
        # Cooling
        temp_diff = temp - self.AMBIENT_TEMP
        fan_cooling = fan * self.COOLING_PER_FAN_LEVEL * (1 + temp_diff / 60.0)
        natural_cooling = temp_diff * self.NATURAL_COOLING_RATE
        total_cooling = fan_cooling + natural_cooling
        
        # Net change
        net = heat - total_cooling / 8.0
        new_temp = self.THERMAL_INERTIA * temp + (1 - self.THERMAL_INERTIA) * (temp + net)
        new_temp += np.random.normal(0, 0.08)
        
        self.server_temperatures[server_id] = np.clip(new_temp, self.AMBIENT_TEMP, self.MAX_TEMP)
    
    def _update_all_temperatures(self):
        for i in range(self.NUM_SERVERS):
            self._update_server_temperature(i)
    
    def _update_energy(self):
        total = 0
        for i in range(self.NUM_SERVERS):
            if self.server_states[i]:
                base = self.BASE_ENERGY_PER_SERVER
                load = (self.MAX_ENERGY_PER_SERVER - base) * (self.server_cpu_loads[i] / 100.0)
                fan = self.fan_states[i] * self.ENERGY_PER_FAN_LEVEL
                total += base + load + fan
        
        self.energy = 0.85 * self.energy + 0.15 * total
        self.energy = max(10, self.energy)
    
    def calculate_current_efficiency(self) -> float:
        active = self.get_active_servers()
        if active == 0:
            return 20.0
        
        avg_cpu = self.get_cpu_usage()
        max_temp = self.get_max_temperature()
        
        # Utilization efficiency
        if avg_cpu < 20:
            util_eff = 35 + avg_cpu * 1.2
        elif avg_cpu < 40:
            util_eff = 59 + (avg_cpu - 20) * 1.1
        elif avg_cpu <= 70:
            util_eff = 81 + (avg_cpu - 40) * 0.35
        elif avg_cpu <= 85:
            util_eff = 91.5 - (avg_cpu - 70) * 0.4
        else:
            util_eff = 85.5 - (avg_cpu - 85) * 1.2
        util_eff = np.clip(util_eff, 20, 92)
        
        # Energy efficiency
        if self.energy > 0 and active > 0:
            work = self.workload * active
            eff = 95 - (self.energy / max(work, 1)) * 8
            energy_eff = np.clip(eff, 25, 92)
        else:
            energy_eff = 50
        
        # Temperature efficiency
        if max_temp < 32:
            temp_eff = 88
        elif max_temp <= 40:
            temp_eff = 94
        elif max_temp <= 48:
            temp_eff = 94 - (max_temp - 40) * 1.5
        elif max_temp <= 58:
            temp_eff = 82 - (max_temp - 48) * 2.2
        else:
            temp_eff = 60 - (max_temp - 58) * 2
        temp_eff = np.clip(temp_eff, 15, 94)
        
        # Fan efficiency
        fan_eff = 88
        for i in range(self.NUM_SERVERS):
            if self.server_states[i]:
                t = self.server_temperatures[i]
                f = self.fan_states[i]
                optimal = 3 if t > 52 else (2 if t > 40 else (1 if t > 30 else 0))
                fan_eff -= abs(f - optimal) * 1.5
        fan_eff = np.clip(fan_eff, 45, 92)
        
        efficiency = util_eff * 0.30 + energy_eff * 0.25 + temp_eff * 0.28 + fan_eff * 0.17
        return round(np.clip(efficiency, 20, 92), 2)
    
    def step(self) -> Dict:
        self.time_step += 1
        self._update_workload()
        self._distribute_workload()
        self._update_all_temperatures()
        self._update_energy()
        return self.get_state()
    
    def get_state(self) -> Dict:
        return {
            'servers': self.get_active_servers(),
            'workload': round(self.workload, 2),
            'cpu': round(self.get_cpu_usage(), 2),
            'energy': round(self.energy, 2),
            'temperature': round(self.get_average_temperature(), 2),
            'max_temperature': round(self.get_max_temperature(), 2),
            'current_efficiency': self.calculate_current_efficiency(),
            'server_states': self.server_states.copy(),
            'fan_states': self.fan_states.copy(),
            'server_temperatures': [round(t, 2) for t in self.server_temperatures],
            'server_cpu_loads': [round(c, 2) for c in self.server_cpu_loads],
            'active_fans': self.get_active_fans(),
            'time_step': self.time_step
        }
    
    def get_features_for_prediction(self) -> np.ndarray:
        return np.array([[
            self.get_active_servers(),
            self.workload,
            self.get_cpu_usage(),
            self.energy,
            self.get_average_temperature()
        ]])
    
    def clone(self) -> 'InfrastructureSimulator':
        c = InfrastructureSimulator()
        c.server_states = self.server_states.copy()
        c.fan_states = self.fan_states.copy()
        c.server_temperatures = self.server_temperatures.copy()
        c.server_cpu_loads = self.server_cpu_loads.copy()
        c.workload = self.workload
        c.energy = self.energy
        c.time_step = self.time_step
        c.workload_base = self.workload_base
        c.workload_trend = self.workload_trend
        c.workload_phase = self.workload_phase
        c.trend_duration = self.trend_duration
        return c
    
    def copy_workload_pattern_from(self, other: 'InfrastructureSimulator'):
        self.workload_base = other.workload_base
        self.workload_trend = other.workload_trend
        self.workload_phase = other.workload_phase
        self.trend_duration = other.trend_duration
        self.workload = other.workload