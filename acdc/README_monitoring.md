# System Monitoring Scripts for Paperspace Machine

These scripts help you monitor your Paperspace machine to identify what might be causing unexpected shutdowns.

## What the Monitoring Scripts Do

### 1. **Resource Tracking**
- **CPU Usage**: Monitors overall CPU usage and per-core utilization
- **Memory Usage**: Tracks RAM usage and swap usage
- **GPU Monitoring**: For your 4x NVIDIA A100 GPUs:
  - Temperature monitoring (critical for GPU health)
  - Power consumption tracking
  - Memory usage per GPU
  - Utilization percentage
- **Disk Usage**: Monitors disk space and I/O activity
- **Network Activity**: Tracks network usage

### 2. **Process Monitoring**
- Identifies the top CPU and memory-consuming processes
- Helps spot runaway processes that might cause issues
- Tracks process changes over time

### 3. **System Log Monitoring**
- Scans system logs for errors, warnings, and critical messages
- Captures kernel panics, segfaults, and hardware errors
- Monitors for thermal events and power issues

### 4. **Alert System**
- Warns when resources approach dangerous levels:
  - CPU > 90%
  - Memory > 95%
  - GPU temperature > 80°C
  - Disk usage > 90%
  - High power consumption

### 5. **Data Logging**
- Saves all monitoring data to log files
- Creates timestamped snapshots for analysis
- Exports data in JSON format for detailed analysis

## Available Scripts

### 1. **Python Monitor** (`monitor_system.py`)
**Features:**
- Comprehensive monitoring with detailed metrics
- JSON data export for analysis
- GPU temperature and power monitoring
- Process tracking with memory usage
- System log error detection

**Usage:**
```bash
# Install required package
pip install psutil

# Run with default settings (30-second intervals)
python3 monitor_system.py

# Run with custom interval
python3 monitor_system.py --interval 60

# Specify custom log file
python3 monitor_system.py --log-file my_monitor.log
```

### 2. **Bash Monitor** (`simple_monitor.sh`)
**Features:**
- No additional dependencies required
- Basic resource monitoring
- Simple text-based logging
- Quick setup and execution

**Usage:**
```bash
# Make executable and run
chmod +x simple_monitor.sh
./simple_monitor.sh
```

## How to Use for Shutdown Investigation

### 1. **Start Monitoring Before Issues**
```bash
# Start the Python monitor in the background
nohup python3 monitor_system.py > monitor_output.log 2>&1 &

# Or use the bash script
nohup ./simple_monitor.sh > monitor_output.log 2>&1 &
```

### 2. **Monitor During Normal Operation**
- Let the script run while you work normally
- Check the logs periodically for any alerts
- Note when you experience shutdowns

### 3. **Analyze After Shutdown**
After a shutdown, check the monitoring logs:

```bash
# Check the last entries before shutdown
tail -50 system_monitor.log

# Look for high resource usage patterns
grep "ALERTS" system_monitor.log

# Check GPU temperature trends
grep "temperature" system_monitor.log

# Look for error patterns
grep "error\|warning\|critical" simple_monitor.log
```

### 4. **What to Look For**

**Resource Issues:**
- Sustained high CPU usage (>90%)
- Memory usage approaching 100%
- GPU temperatures above 80°C
- High power consumption on GPUs

**Process Issues:**
- Processes consuming excessive CPU/memory
- New processes appearing before shutdowns
- Cursor/IDE processes using too many resources

**System Issues:**
- Kernel errors or panics
- Thermal throttling events
- Power management issues
- Network connectivity problems

## Common Shutdown Causes on Paperspace

### 1. **GPU-Related Issues**
- **Overheating**: A100 GPUs can overheat under heavy load
- **Power Limits**: Hitting power consumption limits
- **Memory Issues**: GPU memory exhaustion

### 2. **Resource Exhaustion**
- **CPU Overload**: Sustained 100% CPU usage
- **Memory Pressure**: Running out of RAM
- **Disk Space**: Filling up the root filesystem

### 3. **Cloud Provider Issues**
- **Infrastructure Problems**: Paperspace maintenance or outages
- **Resource Limits**: Hitting instance limits
- **Network Issues**: Connectivity problems

### 4. **Application Issues**
- **Memory Leaks**: Applications consuming more memory over time
- **Infinite Loops**: Processes stuck in high CPU usage
- **Resource Contention**: Multiple applications competing for resources

## Recommendations

### 1. **Immediate Actions**
- Start monitoring now to capture data before next shutdown
- Check Paperspace status page for known issues
- Monitor your resource usage during normal work

### 2. **If Issues Persist**
- Contact Paperspace support with monitoring data
- Consider upgrading your instance if hitting resource limits
- Implement resource limits for your applications

### 3. **Prevention**
- Set up automated monitoring that starts on boot
- Implement application resource limits
- Monitor GPU temperatures during heavy workloads
- Keep an eye on memory usage patterns

## Example Analysis

After a shutdown, you might see in the logs:
```
=== 2024-08-11 15:13:00 ===
CPU Usage: 95.2%
Memory Usage: 98.7%
GPU Info: 0, 85, 95, 380, 35000, 40000
⚠️  ALERTS: High CPU usage: 95.2%, High memory usage: 98.7%, GPU 0 high temperature: 85°C
```

This would indicate:
- Very high CPU and memory usage
- GPU overheating (85°C)
- High power consumption (380W)

This pattern suggests the shutdown was likely due to resource exhaustion or thermal protection. 