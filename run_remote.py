import subprocess
import time
import sys
import threading
import os
import signal

def stream_output(pipe, prefix):
    """Reads output from a pipe and prints it."""
    for line in iter(pipe.readline, b''):
        line_str = line.decode('utf-8', errors='replace').strip()
        if line_str:
            print(f"[{prefix}] {line_str}")
            # Check for Tunnel URL
            if ("localhost.run" in line_str or "serveo.net" in line_str) and "http" in line_str:
                print(f"\n\n{'='*60}")
                print(f"PUBLIC URL: {line_str}")
                print(f"{'='*60}\n")
            # Serveo sometimes just outputs the domain
            if "Forwarding HTTP traffic from" in line_str:
                 print(f"\n\n{'='*60}")
                 print(f"PUBLIC URL: {line_str}")
                 print(f"{'='*60}\n")

def run_remote():
    print("Starting FC Online Simulation Remote Access...")
    
    # 1. Start Streamlit
    print("   -> Starting Streamlit App...")
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true"]
    streamlit_process = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, # Streamlit logs to stderr often
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # Thread to print Streamlit output
    t_st = threading.Thread(target=stream_output, args=(streamlit_process.stderr, "Streamlit"))
    t_st.daemon = True
    t_st.start()
    
    # Wait a bit for Streamlit to start
    time.sleep(3)
    
    # 2. Start SSH Tunnel (localhost.run)
    print("   -> Establishing Secure Tunnel (localhost.run)...")
    # Use 127.0.0.1 to avoid IPv6 issues
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:127.0.0.1:8501", "nokey@localhost.run"]
    
    # SSH often outputs the banner/URL to stdout
    ssh_process = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Thread to print SSH output (stdout has the URL usually)
    t_ssh_out = threading.Thread(target=stream_output, args=(ssh_process.stdout, "Tunnel"))
    t_ssh_out.daemon = True
    t_ssh_out.start()
    
    # Thread to print SSH errors (if any)
    t_ssh_err = threading.Thread(target=stream_output, args=(ssh_process.stderr, "Tunnel Log"))
    t_ssh_err.daemon = True
    t_ssh_err.start()

    print("\nServices Started. Waiting for Public URL...\n")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
            if streamlit_process.poll() is not None:
                print("Streamlit process exited unexpectedly.")
                break
            if ssh_process.poll() is not None:
                print("SSH tunnel process exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        streamlit_process.terminate()
        ssh_process.terminate()
        print("Bye!")

if __name__ == "__main__":
    run_remote()
