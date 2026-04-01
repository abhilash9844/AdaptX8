"""
Server Manager with SMART ML-DRIVEN AI
=======================================
AI uses ML model to evaluate MANY scenarios and picks the BEST one.
Designed to consistently beat human decisions.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import time
import numpy as np
from itertools import combinations


@dataclass
class ServerStatus:
    server_id: int
    name: str
    is_online: bool
    fan_speed: int
    temperature: float
    cpu_load: float


class ServerManager:
    """Server manager with intelligent ML-driven AI."""
    
    NUM_SERVERS = 10
    MIN_SERVERS = 2
    
    def __init__(self):
        self.servers: Dict[int, ServerStatus] = {}
        self._init_servers()
        self.action_history: List[str] = []
        self.ml_model = None
        self.ml_scaler = None
        
    def _init_servers(self):
        for i in range(self.NUM_SERVERS):
            self.servers[i] = ServerStatus(
                server_id=i,
                name=f"Server {i + 1}",
                is_online=True,
                fan_speed=2,
                temperature=32.0,
                cpu_load=50.0
            )
    
    def set_ml_model(self, model, scaler):
        """Set ML model for AI decisions."""
        self.ml_model = model
        self.ml_scaler = scaler
    
    def get_active_count(self) -> int:
        return sum(1 for s in self.servers.values() if s.is_online)
    
    def set_server_online(self, server_id: int, online: bool, 
                          actor: str = "Manual") -> Tuple[bool, str]:
        if server_id not in self.servers:
            return False, "Invalid server"
        
        if not online and self.get_active_count() <= self.MIN_SERVERS:
            return False, f"Min {self.MIN_SERVERS} servers required"
        
        server = self.servers[server_id]
        old = server.is_online
        server.is_online = online
        
        if not online:
            server.fan_speed = 0
        elif online and not old:
            server.fan_speed = 2
        
        action = f"{actor}: S{server_id + 1} → {'ON' if online else 'OFF'}"
        self.action_history.append(action)
        if len(self.action_history) > 30:
            self.action_history.pop(0)
        
        return True, action
    
    def set_fan_speed(self, server_id: int, speed: int,
                      actor: str = "Manual") -> Tuple[bool, str]:
        if server_id not in self.servers or not 0 <= speed <= 3:
            return False, "Invalid"
        
        server = self.servers[server_id]
        if not server.is_online and speed > 0:
            return False, "Server offline"
        
        server.fan_speed = speed
        speed_names = {0: "OFF", 1: "LOW", 2: "MED", 3: "HIGH"}
        
        action = f"{actor}: S{server_id + 1} fan → {speed_names[speed]}"
        self.action_history.append(action)
        if len(self.action_history) > 30:
            self.action_history.pop(0)
        
        return True, action
    
    def sync_with_simulator(self, simulator):
        for i in range(self.NUM_SERVERS):
            self.servers[i].is_online = simulator.server_states[i]
            self.servers[i].fan_speed = simulator.fan_states[i]
            self.servers[i].temperature = simulator.server_temperatures[i]
            self.servers[i].cpu_load = simulator.server_cpu_loads[i]
    
    def _predict_efficiency(self, simulator) -> float:
        """Use ML model to predict efficiency."""
        if self.ml_model is None or self.ml_scaler is None:
            return simulator.calculate_current_efficiency()
        
        features = simulator.get_features_for_prediction()
        scaled = self.ml_scaler.transform(features)
        pred = self.ml_model.predict(scaled)[0]
        return np.clip(pred, 20, 92)
    
    def _simulate_and_evaluate(self, simulator, server_changes: Dict, 
                                fan_changes: Dict) -> Tuple[float, float]:
        """Apply changes to clone and evaluate."""
        test = simulator.clone()
        
        for sid, state in server_changes.items():
            test.set_server_state(sid, state)
        
        for sid, speed in fan_changes.items():
            test.set_fan_state(sid, speed)
        
        test._distribute_workload()
        test._update_all_temperatures()
        test._update_energy()
        
        current = test.calculate_current_efficiency()
        predicted = self._predict_efficiency(test)
        
        return current, predicted
    
    def _get_optimal_fan(self, temp: float) -> int:
        """Optimal fan speed for temperature."""
        if temp >= 55:
            return 3
        elif temp >= 45:
            return 3
        elif temp >= 38:
            return 2
        elif temp >= 30:
            return 1
        else:
            return 0
    
    def _get_optimal_server_count(self, workload: float) -> int:
        """Optimal servers for workload."""
        # Target ~55% CPU per server
        optimal = int(np.ceil(workload / 5.5))
        return np.clip(optimal, self.MIN_SERVERS, self.NUM_SERVERS)
    
    def ai_smart_control(self, simulator, current_efficiency: float) -> Dict:
        """
        SMART ML-DRIVEN AI
        
        Evaluates MANY scenarios using ML model:
        1. All possible fan configurations for each server
        2. Server count optimization
        3. Combined optimizations
        4. Picks the configuration with HIGHEST predicted efficiency
        
        This AI is designed to consistently beat human decisions.
        """
        state = simulator.get_state()
        server_states = state['server_states']
        fan_states = state['fan_states']
        temperatures = state['server_temperatures']
        cpu_loads = state['server_cpu_loads']
        workload = state['workload']
        active = state['servers']
        
        result = {
            'actions': [],
            'server_changes': {},
            'fan_changes': {},
            'reasoning': [],
            'scenarios_tested': 0,
            'best_predicted': current_efficiency,
            'should_act': False
        }
        
        scenarios = []
        
        # === BASELINE ===
        curr_pred = self._predict_efficiency(simulator)
        scenarios.append({
            'name': 'Current',
            'server_changes': {},
            'fan_changes': {},
            'score': curr_pred * 0.6 + current_efficiency * 0.4
        })
        
        # === STRATEGY 1: OPTIMIZE EACH FAN INDIVIDUALLY ===
        for i in range(self.NUM_SERVERS):
            if server_states[i]:
                optimal_fan = self._get_optimal_fan(temperatures[i])
                if optimal_fan != fan_states[i]:
                    curr, pred = self._simulate_and_evaluate(
                        simulator, {}, {i: optimal_fan}
                    )
                    scenarios.append({
                        'name': f'S{i+1} fan → {optimal_fan}',
                        'server_changes': {},
                        'fan_changes': {i: optimal_fan},
                        'score': pred * 0.6 + curr * 0.4
                    })
        
        # === STRATEGY 2: OPTIMIZE ALL FANS TOGETHER ===
        all_fan_changes = {}
        for i in range(self.NUM_SERVERS):
            if server_states[i]:
                optimal = self._get_optimal_fan(temperatures[i])
                if optimal != fan_states[i]:
                    all_fan_changes[i] = optimal
        
        if all_fan_changes:
            curr, pred = self._simulate_and_evaluate(simulator, {}, all_fan_changes)
            scenarios.append({
                'name': 'Optimize all fans',
                'server_changes': {},
                'fan_changes': all_fan_changes.copy(),
                'score': pred * 0.6 + curr * 0.4
            })
        
        # === STRATEGY 3: MAX COOLING ===
        max_cool = {i: 3 for i in range(self.NUM_SERVERS) 
                    if server_states[i] and fan_states[i] < 3}
        if max_cool:
            curr, pred = self._simulate_and_evaluate(simulator, {}, max_cool)
            scenarios.append({
                'name': 'Max cooling',
                'server_changes': {},
                'fan_changes': max_cool.copy(),
                'score': pred * 0.6 + curr * 0.4
            })
        
        # === STRATEGY 4: ENERGY SAVE (reduce fans when cool) ===
        energy_save = {}
        for i in range(self.NUM_SERVERS):
            if server_states[i] and temperatures[i] < 32 and fan_states[i] > 1:
                energy_save[i] = 1
        if energy_save:
            curr, pred = self._simulate_and_evaluate(simulator, {}, energy_save)
            scenarios.append({
                'name': 'Energy save',
                'server_changes': {},
                'fan_changes': energy_save.copy(),
                'score': pred * 0.6 + curr * 0.4
            })
        
        # === STRATEGY 5: TURN OFF SERVERS (low workload) ===
        load_per_server = workload / active if active > 0 else 0
        
        if load_per_server < 40 and active > self.MIN_SERVERS:
            online = sorted(
                [i for i in range(self.NUM_SERVERS) if server_states[i]],
                key=lambda x: cpu_loads[x]
            )
            
            # Try turning off 1 server
            for s in online[:3]:
                if active - 1 >= self.MIN_SERVERS:
                    curr, pred = self._simulate_and_evaluate(
                        simulator, {s: False}, {}
                    )
                    scenarios.append({
                        'name': f'Turn off S{s+1}',
                        'server_changes': {s: False},
                        'fan_changes': {},
                        'score': pred * 0.6 + curr * 0.4
                    })
            
            # Try turning off 2 servers
            if active > self.MIN_SERVERS + 1 and load_per_server < 30:
                for combo in combinations(online[:4], 2):
                    curr, pred = self._simulate_and_evaluate(
                        simulator, {combo[0]: False, combo[1]: False}, {}
                    )
                    scenarios.append({
                        'name': f'Turn off S{combo[0]+1} & S{combo[1]+1}',
                        'server_changes': {combo[0]: False, combo[1]: False},
                        'fan_changes': {},
                        'score': pred * 0.6 + curr * 0.4
                    })
        
        # === STRATEGY 6: TURN ON SERVERS (high workload) ===
        if load_per_server > 65:
            offline = [i for i in range(self.NUM_SERVERS) if not server_states[i]]
            
            for s in offline[:3]:
                curr, pred = self._simulate_and_evaluate(
                    simulator, {s: True}, {}
                )
                scenarios.append({
                    'name': f'Turn on S{s+1}',
                    'server_changes': {s: True},
                    'fan_changes': {},
                    'score': pred * 0.6 + curr * 0.4
                })
        
        # === STRATEGY 7: COMBINED - Server changes + Fan optimization ===
        for sc in list(scenarios):
            if sc['server_changes'] and all_fan_changes:
                combined_fans = all_fan_changes.copy()
                # Remove fans for servers being turned off
                for sid in sc['server_changes']:
                    if not sc['server_changes'][sid] and sid in combined_fans:
                        del combined_fans[sid]
                
                if combined_fans:
                    curr, pred = self._simulate_and_evaluate(
                        simulator, sc['server_changes'], combined_fans
                    )
                    scenarios.append({
                        'name': f"{sc['name']} + fans",
                        'server_changes': sc['server_changes'].copy(),
                        'fan_changes': combined_fans.copy(),
                        'score': pred * 0.6 + curr * 0.4
                    })
        
        # === STRATEGY 8: AGGRESSIVE OPTIMIZATION ===
        optimal_servers = self._get_optimal_server_count(workload)
        if optimal_servers != active:
            server_changes = {}
            if optimal_servers < active:
                online = sorted(
                    [i for i in range(self.NUM_SERVERS) if server_states[i]],
                    key=lambda x: cpu_loads[x]
                )
                for i, s in enumerate(online):
                    if active - i - 1 >= optimal_servers and active - i - 1 >= self.MIN_SERVERS:
                        server_changes[s] = False
            else:
                offline = [i for i in range(self.NUM_SERVERS) if not server_states[i]]
                for s in offline[:optimal_servers - active]:
                    server_changes[s] = True
            
            if server_changes:
                curr, pred = self._simulate_and_evaluate(simulator, server_changes, all_fan_changes)
                scenarios.append({
                    'name': f'Optimize to {optimal_servers} servers + fans',
                    'server_changes': server_changes.copy(),
                    'fan_changes': all_fan_changes.copy(),
                    'score': pred * 0.6 + curr * 0.4
                })
        
        result['scenarios_tested'] = len(scenarios)
        
        # === SELECT BEST SCENARIO ===
        best = max(scenarios, key=lambda x: x['score'])
        baseline_score = scenarios[0]['score']
        
        result['reasoning'].append(f"📊 Tested {len(scenarios)} scenarios")
        
        # Show top scenarios
        sorted_scenarios = sorted(scenarios, key=lambda x: -x['score'])[:5]
        for s in sorted_scenarios:
            marker = "✓" if s == best else " "
            result['reasoning'].append(f"  {marker} {s['name']}: {s['score']:.1f}")
        
        # Apply if better
        if best['score'] > baseline_score + 0.1:
            result['should_act'] = True
            result['server_changes'] = best['server_changes']
            result['fan_changes'] = best['fan_changes']
            result['best_predicted'] = best['score']
            
            improvement = best['score'] - baseline_score
            result['reasoning'].append(f"✅ Applying '{best['name']}' (+{improvement:.2f})")
            
            # Generate actions
            speed_names = {0: "OFF", 1: "LOW", 2: "MED", 3: "HIGH"}
            for sid, state in result['server_changes'].items():
                result['actions'].append(f"🖥️ S{sid+1} → {'ON' if state else 'OFF'}")
            for sid, speed in result['fan_changes'].items():
                result['actions'].append(f"🌀 S{sid+1} → {speed_names[speed]}")
        else:
            result['reasoning'].append("✅ Current config is optimal")
        
        return result
    
    def apply_ai_actions(self, result: Dict, simulator) -> List[str]:
        if not result['should_act']:
            return []
        
        actions = []
        for sid, state in result['server_changes'].items():
            success, msg = self.set_server_online(sid, state, "AI")
            if success:
                simulator.set_server_state(sid, state)
                actions.append(msg)
        
        for sid, speed in result['fan_changes'].items():
            success, msg = self.set_fan_speed(sid, speed, "AI")
            if success:
                simulator.set_fan_state(sid, speed)
                actions.append(msg)
        
        return actions