#!/usr/bin/env python3
"""
Monitor experiments and handle errors
"""
import time
import subprocess
import os
import sys
from datetime import datetime

def check_process_status():
    """Check if main processes are still running"""
    try:
        # Check for pipeline process
        result = subprocess.run(['pgrep', '-f', 'run_experiment_pipeline'], capture_output=True, text=True)
        pipeline_running = bool(result.stdout.strip())
        
        # Check for main experiments
        result = subprocess.run(['pgrep', '-f', 'main_experiments'], capture_output=True, text=True)
        experiments_running = bool(result.stdout.strip())
        
        return pipeline_running, experiments_running
    except:
        return False, False

def check_for_errors():
    """Check logs for critical errors"""
    error_indicators = [
        "CUDA out of memory",
        "RuntimeError",
        "AssertionError",
        "KeyError",
        "ValueError: Expected",  # But not the model name error we already fixed
        "Process Process-",  # Multiprocessing crashes
    ]
    
    # Check the main experiment log
    try:
        with open('experiment_log_20250811_032318.log', 'r') as f:
            content = f.read()
            for indicator in error_indicators:
                if indicator in content and "not found. Valid official model names" not in content:
                    return True, indicator
    except:
        pass
    
    return False, None

def main():
    print(f"Started monitoring at {datetime.now()}")
    
    check_count = 0
    max_checks = 480  # 8 hours * 60 min/hour / 1 min per check
    
    while check_count < max_checks:
        time.sleep(60)  # Check every minute
        check_count += 1
        
        pipeline_running, experiments_running = check_process_status()
        has_error, error_type = check_for_errors()
        
        if not pipeline_running and not experiments_running:
            print(f"[{datetime.now()}] Experiments completed or stopped")
            break
            
        if has_error:
            print(f"[{datetime.now()}] Error detected: {error_type}")
            # Log it but don't try to fix - let pipeline handle it
            
        if check_count % 60 == 0:  # Every hour
            print(f"[{datetime.now()}] Still running... (hour {check_count//60})")
    
    print(f"Monitoring ended at {datetime.now()}")

if __name__ == "__main__":
    main()