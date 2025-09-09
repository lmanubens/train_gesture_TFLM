import serial
import time
import os
from datetime import datetime

# === User config ===
PORT = "COM6"         # <-- change to your ESP32 port
BAUD = 9600
OUTDIR = "captures"   # directory for saved CSVs
ECHO_DATA = True      # True = print every CSV line; False = only show progress

def make_filename(label):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTDIR, f"gesture_L{label}_{ts}.csv")

def wait_and_save(ser, label):
    """Listen for one CSV block and save to file"""
    recording = False
    f = None
    filename = make_filename(label)

    print(f"[INFO] Waiting for CSV from ESP32 (label {label})...")

    current_rep = None
    line_count = 0

    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue

        if line.startswith("###CSV_BEGIN"):
            print(f"[INFO] Detected start of CSV for label {label}")
            f = open(filename, "w", encoding="utf-8")
            recording = True
            line_count = 0
            continue

        if line.startswith("###CSV_END"):
            if f:
                f.close()
                print(f"[SUCCESS] Saved file: {filename}")
                f = None
            recording = False
            break  # stop after one capture

        if recording and f:
            f.write(line + "\n")
            line_count += 1

            # Try to parse rep from CSV
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    rep_val = int(parts[1])
                    if rep_val != current_rep:
                        current_rep = rep_val
                        print(f"  [REP {rep_val}]")
                except ValueError:
                    pass  # skip header line

            # Echo data line if enabled
            if ECHO_DATA:
                print("   " + line)

def main():
    print("=== ESP32 Gesture Data Capture (Verbose) ===")
    print(f"Opening port {PORT} at {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # give board time to reset
    ser.reset_input_buffer()

    os.makedirs(OUTDIR, exist_ok=True)

    # Step 1: capture label 1
    print("\n[STEP 1] Setting label = 1")
    ser.write(b"1")
    time.sleep(0.5)

    print("[STEP 1] Starting recording (30 reps)...")
    ser.write(b"r")
    wait_and_save(ser, label=1)

    # Step 2: capture label 2
    print("\n[STEP 2] Setting label = 2")
    ser.write(b"2")
    time.sleep(0.5)

    print("[STEP 2] Starting recording (30 reps)...")
    ser.write(b"r")
    wait_and_save(ser, label=2)

    print("\n=== Capture complete! Files are saved in:", OUTDIR, "===\n")

if __name__ == "__main__":
    main()
