"""Temperature Alarm System"""

import threading
import time
from typing import List


class AlarmSystem:
    def __init__(self, default_threshold: float = 50.0):
        self._threshold = default_threshold
        self.is_active = False
        self.last_alarm = 0
        self.cooldown = 8
        self.history: List[dict] = []
        
    def get_threshold(self) -> float:
        return self._threshold
    
    def set_threshold(self, threshold: float) -> bool:
        if 25 <= threshold <= 80:
            self._threshold = threshold
            return True
        return False
    
    def check_temperature(self, temperature: float) -> dict:
        result = {
            'exceeded': False,
            'triggered': False,
            'temp': temperature,
            'threshold': self._threshold,
            'message': ''
        }
        
        if temperature > self._threshold:
            result['exceeded'] = True
            now = time.time()
            
            if now - self.last_alarm >= self.cooldown and not self.is_active:
                result['triggered'] = True
                result['message'] = f"🚨 {temperature:.1f}°C > {self._threshold}°C!"
                self.history.append({'time': now, 'temp': temperature})
                self._trigger()
            else:
                result['message'] = f"⚠️ High: {temperature:.1f}°C"
        else:
            margin = self._threshold - temperature
            if margin < 5:
                result['message'] = f"⚡ Near threshold: {temperature:.1f}°C"
            else:
                result['message'] = f"✅ Safe: {temperature:.1f}°C"
        
        return result
    
    def _trigger(self):
        self.last_alarm = time.time()
        self.is_active = True
        
        def alarm():
            for _ in range(3):
                print("\a", end="", flush=True)
                time.sleep(0.3)
            self.is_active = False
        
        threading.Thread(target=alarm, daemon=True).start()