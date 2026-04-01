"""Dataset Generation"""
import numpy as np
import pandas as pd
import os
from pathlib import Path

def generate_dataset(n=5000, output_path="dataset/dataset.csv"):
    print(f"Generating {n} samples...")
    
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    
    data = {'servers': [], 'workload': [], 'cpu': [], 'energy': [], 'temperature': [], 'efficiency': []}
    
    servers, workload, cpu, energy, temp = 10, 50, 50, 600, 32
    wb, wt, wp, td = 50, 0, 0, 0
    
    for _ in range(n):
        wp += 0.05
        if wp > 2*np.pi: wp -= 2*np.pi
        td -= 1
        if td <= 0:
            wt = np.random.uniform(-0.5, 0.5)
            td = np.random.randint(40, 100)
        wb = np.clip(wb + wt, 25, 85)
        workload = np.clip(0.92*workload + 0.08*(wb + 12*np.sin(wp) + np.random.normal(0, 0.8)), 10, 100)
        
        lps = workload / servers
        if np.random.random() < 0.03:
            if lps < 30 and servers > 2: servers -= 1
            elif lps > 70 and servers < 10: servers += 1
        
        cpu = np.clip(0.85*cpu + 0.15*(lps*10 + np.random.normal(0, 2)), 5, 100)
        energy = 0.85*energy + 0.15*(servers*45 + servers*75*(cpu/100) + servers*2*8)
        
        heat = (cpu/100)*0.35 + max(0, (cpu-70)/100*0.15)
        cool = 2*0.7*(1 + (temp-22)/60) + (temp-22)*0.04
        temp = np.clip(0.88*temp + 0.12*(temp + heat - cool/8) + np.random.normal(0, 0.08), 22, 75)
        
        if cpu < 20: ue = 35 + cpu*1.2
        elif cpu < 40: ue = 59 + (cpu-20)*1.1
        elif cpu <= 70: ue = 81 + (cpu-40)*0.35
        else: ue = 91.5 - (cpu-70)*0.4
        ue = np.clip(ue, 20, 92)
        
        ee = np.clip(95 - (energy/max(workload*servers, 1))*8, 25, 92) if energy > 0 else 50
        
        if temp < 32: te = 88
        elif temp <= 40: te = 94
        elif temp <= 48: te = 94 - (temp-40)*1.5
        else: te = 82 - (temp-48)*2.2
        te = np.clip(te, 15, 94)
        
        eff = np.clip(ue*0.30 + ee*0.25 + te*0.28 + 88*0.17, 20, 92)
        
        data['servers'].append(servers)
        data['workload'].append(round(workload, 2))
        data['cpu'].append(round(cpu, 2))
        data['energy'].append(round(energy, 2))
        data['temperature'].append(round(temp, 2))
        data['efficiency'].append(round(eff, 2))
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")
    print(df.describe().round(2))
    return df

if __name__ == "__main__":
    generate_dataset()