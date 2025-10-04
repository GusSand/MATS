#!/bin/bash
# Simple System Monitor for Paperspace Machine
# Monitors basic system resources and logs to help identify shutdown causes

LOG_FILE="simple_monitor.log"
INTERVAL=30

# Function to get timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to get CPU usage
get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
}

# Function to get memory usage
get_memory_usage() {
    free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}'
}

# Function to get GPU info
get_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits | head -1
    else
        echo "No GPU detected"
    fi
}

# Function to get disk usage
get_disk_usage() {
    df / | tail -1 | awk '{print $5}' | cut -d'%' -f1
}

# Function to get top processes
get_top_processes() {
    ps aux --sort=-%cpu | head -6 | tail -3 | awk '{print $2, $3, $4, $11}' | while read pid cpu mem cmd; do
        echo "PID: $pid, CPU: ${cpu}%, MEM: ${mem}%, CMD: $cmd"
    done
}

# Function to check for recent errors in logs
check_recent_errors() {
    journalctl --since "5 minutes ago" 2>/dev/null | grep -i "error\|warning\|critical\|panic\|oops" | tail -3
}

# Function to log data
log_data() {
    local timestamp="$1"
    local cpu="$2"
    local memory="$3"
    local gpu="$4"
    local disk="$5"
    local processes="$6"
    local errors="$7"
    
    echo "=== $timestamp ===" >> "$LOG_FILE"
    echo "CPU Usage: ${cpu}%" >> "$LOG_FILE"
    echo "Memory Usage: ${memory}%" >> "$LOG_FILE"
    echo "GPU Info: $gpu" >> "$LOG_FILE"
    echo "Disk Usage: ${disk}%" >> "$LOG_FILE"
    echo "Top Processes:" >> "$LOG_FILE"
    echo "$processes" >> "$LOG_FILE"
    echo "Recent Errors:" >> "$LOG_FILE"
    echo "$errors" >> "$LOG_FILE"
    echo "---" >> "$LOG_FILE"
}

# Function to print summary
print_summary() {
    local timestamp="$1"
    local cpu="$2"
    local memory="$3"
    local gpu="$4"
    local disk="$5"
    
    echo "=== System Status at $timestamp ==="
    echo "CPU: ${cpu}%"
    echo "Memory: ${memory}%"
    echo "Disk: ${disk}%"
    echo "GPU: $gpu"
}

# Function to check for alerts
check_alerts() {
    local cpu="$1"
    local memory="$2"
    local disk="$3"
    local alerts=""
    
    if (( $(echo "$cpu > 90" | bc -l) )); then
        alerts="$alerts High CPU usage: ${cpu}%"
    fi
    
    if (( $(echo "$memory > 95" | bc -l) )); then
        alerts="$alerts High memory usage: ${memory}%"
    fi
    
    if (( $(echo "$disk > 90" | bc -l) )); then
        alerts="$alerts High disk usage: ${disk}%"
    fi
    
    if [ -n "$alerts" ]; then
        echo "⚠️  ALERTS:$alerts"
    fi
}

# Main monitoring loop
echo "Starting simple system monitoring..."
echo "Logging to: $LOG_FILE"
echo "Interval: $INTERVAL seconds"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    timestamp=$(get_timestamp)
    cpu=$(get_cpu_usage)
    memory=$(get_memory_usage)
    gpu=$(get_gpu_info)
    disk=$(get_disk_usage)
    processes=$(get_top_processes)
    errors=$(check_recent_errors)
    
    # Log data
    log_data "$timestamp" "$cpu" "$memory" "$gpu" "$disk" "$processes" "$errors"
    
    # Print summary
    print_summary "$timestamp" "$cpu" "$memory" "$gpu" "$disk"
    
    # Check for alerts
    check_alerts "$cpu" "$memory" "$disk"
    
    echo ""
    sleep "$INTERVAL"
done 