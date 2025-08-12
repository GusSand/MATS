#!/usr/bin/env python3
"""
System Monitoring Script for Paperspace Machine
Tracks resources and logs to identify shutdown causes
"""

import subprocess
import time
import json
import os
from datetime import datetime
import psutil
import threading
import signal
import sys

class SystemMonitor:
    def __init__(self, log_file="system_monitor.log", interval=30):
        self.log_file = log_file
        self.interval = interval
        self.running = True
        self.monitoring_data = []
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
        self.save_data()
    
    def get_cpu_info(self):
        """Get CPU usage and temperature info"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Get per-core usage
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "cpu_freq_mhz": cpu_freq.current if cpu_freq else None,
                "cpu_per_core": cpu_per_core
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_memory_info(self):
        """Get memory usage info"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "memory_total_gb": memory.total / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_gpu_info(self):
        """Get GPU usage and temperature info"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            gpu_data = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 7:
                        gpu_data.append({
                            "gpu_index": parts[0],
                            "gpu_name": parts[1],
                            "temperature": int(parts[2]) if parts[2].isdigit() else None,
                            "utilization": int(parts[3]) if parts[3].isdigit() else None,
                            "memory_used_mb": int(parts[4]) if parts[4].isdigit() else None,
                            "memory_total_mb": int(parts[5]) if parts[5].isdigit() else None,
                            "power_draw_w": float(parts[6]) if parts[6].replace('.', '').isdigit() else None,
                            "power_limit_w": float(parts[7]) if len(parts) > 7 and parts[7].replace('.', '').isdigit() else None
                        })
            
            return gpu_data
        except Exception as e:
            return {"error": str(e)}
    
    def get_disk_info(self):
        """Get disk usage info"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                "disk_total_gb": disk_usage.total / (1024**3),
                "disk_used_gb": disk_usage.used / (1024**3),
                "disk_percent": disk_usage.percent,
                "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
                "disk_write_bytes": disk_io.write_bytes if disk_io else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_network_info(self):
        """Get network usage info"""
        try:
            network_io = psutil.net_io_counters()
            
            return {
                "network_bytes_sent": network_io.bytes_sent,
                "network_bytes_recv": network_io.bytes_recv,
                "network_packets_sent": network_io.packets_sent,
                "network_packets_recv": network_io.packets_recv
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_top_processes(self, limit=10):
        """Get top CPU and memory consuming processes"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
                try:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent'],
                        "memory_rss_mb": proc.info['memory_info'].rss / (1024**2) if proc.info['memory_info'] else 0
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            return processes[:limit]
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_logs(self):
        """Get recent system logs for errors"""
        try:
            # Get last 50 lines of system logs
            result = subprocess.run(['journalctl', '--no-pager', '-n', '50'], 
                                  capture_output=True, text=True, timeout=10)
            
            # Filter for errors, warnings, and critical messages
            error_lines = []
            for line in result.stdout.split('\n'):
                if any(keyword in line.lower() for keyword in ['error', 'warning', 'critical', 'panic', 'oops', 'segfault']):
                    error_lines.append(line.strip())
            
            return error_lines[-10:]  # Return last 10 error lines
        except Exception as e:
            return {"error": str(e)}
    
    def collect_data(self):
        """Collect all system data"""
        timestamp = datetime.now().isoformat()
        
        data = {
            "timestamp": timestamp,
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "gpu": self.get_gpu_info(),
            "disk": self.get_disk_info(),
            "network": self.get_network_info(),
            "top_processes": self.get_top_processes(),
            "system_logs": self.get_system_logs()
        }
        
        return data
    
    def log_data(self, data):
        """Log data to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def print_summary(self, data):
        """Print a human-readable summary"""
        print(f"\n=== System Status at {data['timestamp']} ===")
        
        # CPU
        if 'error' not in data['cpu']:
            print(f"CPU: {data['cpu']['cpu_percent']:.1f}% (Freq: {data['cpu']['cpu_freq_mhz']:.0f} MHz)")
        
        # Memory
        if 'error' not in data['memory']:
            print(f"Memory: {data['memory']['memory_percent']:.1f}% ({data['memory']['memory_used_gb']:.1f}GB / {data['memory']['memory_total_gb']:.1f}GB)")
        
        # GPU
        if isinstance(data['gpu'], list) and data['gpu']:
            for gpu in data['gpu']:
                if 'error' not in gpu:
                    print(f"GPU {gpu['gpu_index']}: {gpu['utilization']}% util, {gpu['temperature']}°C, {gpu['power_draw_w']:.1f}W")
        
        # Top processes
        if data['top_processes'] and len(data['top_processes']) > 0:
            print("\nTop CPU processes:")
            for proc in data['top_processes'][:3]:
                print(f"  {proc['name']}: {proc['cpu_percent']:.1f}% CPU, {proc['memory_rss_mb']:.1f}MB RAM")
    
    def save_data(self):
        """Save monitoring data to JSON file"""
        if self.monitoring_data:
            filename = f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump(self.monitoring_data, f, indent=2)
                print(f"\nMonitoring data saved to {filename}")
            except Exception as e:
                print(f"Error saving data: {e}")
    
    def run(self):
        """Main monitoring loop"""
        print(f"Starting system monitoring... (Press Ctrl+C to stop)")
        print(f"Logging to: {self.log_file}")
        print(f"Interval: {self.interval} seconds")
        
        while self.running:
            try:
                data = self.collect_data()
                self.monitoring_data.append(data)
                self.log_data(data)
                self.print_summary(data)
                
                # Check for potential issues
                self.check_alerts(data)
                
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)
        
        self.save_data()
        print("\nMonitoring stopped.")
    
    def check_alerts(self, data):
        """Check for potential issues that might cause shutdowns"""
        alerts = []
        
        # CPU alerts
        if 'error' not in data['cpu'] and data['cpu']['cpu_percent'] > 90:
            alerts.append(f"High CPU usage: {data['cpu']['cpu_percent']:.1f}%")
        
        # Memory alerts
        if 'error' not in data['memory'] and data['memory']['memory_percent'] > 95:
            alerts.append(f"High memory usage: {data['memory']['memory_percent']:.1f}%")
        
        # GPU alerts
        if isinstance(data['gpu'], list):
            for gpu in data['gpu']:
                if 'error' not in gpu:
                    if gpu['temperature'] and gpu['temperature'] > 80:
                        alerts.append(f"GPU {gpu['gpu_index']} high temperature: {gpu['temperature']}°C")
                    if gpu['power_draw_w'] and gpu['power_limit_w'] and gpu['power_draw_w'] > gpu['power_limit_w'] * 0.95:
                        alerts.append(f"GPU {gpu['gpu_index']} high power usage: {gpu['power_draw_w']:.1f}W")
        
        # Disk alerts
        if 'error' not in data['disk'] and data['disk']['disk_percent'] > 90:
            alerts.append(f"High disk usage: {data['disk']['disk_percent']:.1f}%")
        
        if alerts:
            print(f"\n⚠️  ALERTS: {', '.join(alerts)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="System monitoring script")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--log-file", default="system_monitor.log", help="Log file path")
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(log_file=args.log_file, interval=args.interval)
    monitor.run() 