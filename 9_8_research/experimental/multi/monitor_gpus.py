# monitor_gpus.py
"""
Monitor GPU usage during experiments
"""

import subprocess
import time

def monitor():
    while True:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("\033[2J\033[H")  # Clear screen
        print(result.stdout)
        
        # Check memory usage
        for line in result.stdout.split('\n'):
            if 'MiB' in line and '/' in line:
                used = int(line.split('|')[2].split('/')[0].strip().replace('MiB', ''))
                total = int(line.split('|')[2].split('/')[1].strip().replace('MiB', ''))
                
                if used / total > 0.9:
                    print("⚠️ WARNING: GPU memory >90% full!")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor()