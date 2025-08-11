#!/usr/bin/env python3
"""
Master script to orchestrate the entire experiment pipeline:
1. Run GPU monitoring in background
2. Run main experiments
3. Send email notification
4. Wait 10 minutes
5. Shutdown the machine
"""

import subprocess
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import sys
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def send_email_notification(recipient, subject, body):
    """
    Send email notification via Gmail or similar
    """
    try:
        # Using a simple approach with mail command if available
        # For production, you'd want to configure SMTP properly
        message = f"Subject: {subject}\n\n{body}"
        
        # Try using mail command if available
        result = subprocess.run(
            ['mail', '-s', subject, recipient],
            input=body.encode(),
            capture_output=True,
            text=False
        )
        
        if result.returncode == 0:
            logger.info(f"Email sent successfully to {recipient}")
            return True
        else:
            logger.warning(f"mail command failed, trying alternative method")
            
            # Alternative: write to file for manual sending
            with open('email_notification.txt', 'w') as f:
                f.write(f"To: {recipient}\n")
                f.write(f"Subject: {subject}\n\n")
                f.write(body)
            logger.info("Email content saved to email_notification.txt")
            return True
            
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

def run_gpu_monitoring():
    """
    Start GPU monitoring in background
    """
    logger.info("Starting GPU monitoring in background...")
    monitor_process = subprocess.Popen(
        ['python', 'monitor_gpus.py'],
        stdout=open('gpu_monitor.log', 'w'),
        stderr=subprocess.STDOUT
    )
    logger.info(f"GPU monitor started with PID: {monitor_process.pid}")
    return monitor_process

def run_main_experiments():
    """
    Run the main experiments
    """
    logger.info("Starting main experiments...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', 'main_experiments.py'],
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed_time = (time.time() - start_time) / 3600
        logger.info(f"Experiments completed successfully in {elapsed_time:.2f} hours")
        
        # Save output
        with open('experiment_output.log', 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
        
        return True, elapsed_time, "Success"
        
    except subprocess.CalledProcessError as e:
        elapsed_time = (time.time() - start_time) / 3600
        error_msg = f"Experiments failed after {elapsed_time:.2f} hours\nError: {e}\nOutput: {e.stdout}\nError Output: {e.stderr}"
        logger.error(error_msg)
        
        # Save error output
        with open('experiment_error.log', 'w') as f:
            f.write(error_msg)
        
        return False, elapsed_time, error_msg
    
    except Exception as e:
        elapsed_time = (time.time() - start_time) / 3600
        error_msg = f"Unexpected error after {elapsed_time:.2f} hours: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return False, elapsed_time, error_msg

def wait_and_shutdown(wait_minutes=10):
    """
    Wait specified minutes then shutdown
    """
    logger.info(f"Waiting {wait_minutes} minutes before shutdown...")
    
    # Countdown timer
    for remaining in range(wait_minutes * 60, 0, -60):
        minutes_left = remaining // 60
        logger.info(f"Shutdown in {minutes_left} minutes...")
        time.sleep(60)
    
    logger.info("Initiating shutdown...")
    
    # Use sudo shutdown (will require proper permissions)
    result = subprocess.run(['sudo', 'shutdown', '-h', 'now'], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Shutdown failed: {result.stderr}")
        logger.info("Please shutdown manually")
    else:
        logger.info("Shutdown command issued successfully")

def main():
    """
    Main orchestration function
    """
    logger.info("="*60)
    logger.info("STARTING EXPERIMENT PIPELINE")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("="*60)
    
    # Check if we have the required files
    required_files = ['main_experiments.py', 'monitor_gpus.py', 'experiment_config.json']
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"Missing required file: {file}")
            sys.exit(1)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Start GPU monitoring
    monitor_process = run_gpu_monitoring()
    
    try:
        # Step 2: Run main experiments
        success, runtime, message = run_main_experiments()
        
        # Step 3: Stop GPU monitoring
        logger.info("Stopping GPU monitor...")
        monitor_process.terminate()
        time.sleep(2)
        if monitor_process.poll() is None:
            monitor_process.kill()
        
        # Step 4: Send email notification
        if success:
            subject = "✅ Experiments Completed Successfully"
            body = f"""
Your experiments have completed successfully!

Runtime: {runtime:.2f} hours
Estimated cost: ${runtime * 5.52:.2f}
End time: {datetime.now()}

Results saved in:
- results/positive_controls.csv
- results/causal_effects.npy
- results/phase_transition.csv
- results/path_patching.csv
- results/all_results.pt
- results/summary.txt

Check the logs for detailed information:
- experiment_log_*.log (main experiment log)
- gpu_monitor.log (GPU usage log)
- pipeline_log_*.log (this pipeline log)

The machine will shutdown in 10 minutes.
"""
        else:
            subject = "⚠️ Experiments Failed"
            body = f"""
Your experiments encountered an error.

Runtime before failure: {runtime:.2f} hours
Error details: {message}

Partial results may be saved in:
- results/partial_results.pt

Check the error logs:
- experiment_error.log
- experiment_log_*.log
- pipeline_log_*.log

The machine will shutdown in 10 minutes.
"""
        
        logger.info(f"Sending notification email to gussand@gmail.com...")
        send_email_notification("gussand@gmail.com", subject, body)
        
        # Step 5: Wait and shutdown
        wait_and_shutdown(wait_minutes=10)
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        monitor_process.terminate()
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}")
        logger.error(traceback.format_exc())
        monitor_process.terminate()
        
        # Send emergency notification
        send_email_notification(
            "gussand@gmail.com",
            "❌ Pipeline Critical Error",
            f"Critical error in experiment pipeline:\n{e}\n\nPlease check the system manually."
        )
        
        # Still try to shutdown
        wait_and_shutdown(wait_minutes=10)

if __name__ == "__main__":
    main()