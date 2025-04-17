from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import json
import os
import glob
import numpy as np

def calculate_fft(accel1, accel2, sample_rate=1600):
        # Perform FFT
        N = len(accel1)
        T = 1.0 / sample_rate
        yf1 = rfft(accel1)
        yf2 = rfft(accel2)
        f_axis = rfftfreq(N, T)[:N//2]
        return [f_axis, yf1, yf2, N]

def calculate_phase_gain_from_fft(f_axis, yf1, yf2, N):
                
        # Find the dominant frequency (should be the driving frequency)
        dominant_idx = np.argmax(np.abs(yf1[1:N//2])) + 1  # Skip the DC component
        dominant_freq = f_axis[dominant_idx]
        
        # Calculate phase at the dominant frequency only
        phase1 = np.angle(yf1[dominant_idx])
        phase2 = np.angle(yf2[dominant_idx])

        # Calculate phase difference in radians
        phase_diff_rad = phase2 - phase1
        
        # Normalize to [-π, π]
        phase_diff_rad = ((phase_diff_rad + np.pi) % (2 * np.pi)) - np.pi
        
        # Convert to degrees
        phase_diff_deg = np.degrees(phase_diff_rad)

        # Calculate amplitude at the dominant frequency
        amplitude1 = 2.0/N * np.abs(yf1[dominant_idx])
        amplitude2 = 2.0/N * np.abs(yf2[dominant_idx])
        gain = amplitude2 / amplitude1
        
        print(f"Dominant frequency: {dominant_freq:.2f} Hz")
        print(f"Phase difference: {phase_diff_deg:.2f} degrees")
        print(f"Gain: {gain:.2f}")
        
        return [dominant_freq, phase_diff_deg, gain]

def plot_fft(f_axis, yf1, yf2, N):
    plt.plot(f_axis, 2.0/N * np.abs(yf1[:N//2]), label='Reference')
    plt.plot(f_axis, 2.0/N * np.abs(yf2[:N//2]), label='O-ring')

def calculate_amplitude_phase_delay(json_file=None):
    if json_file is None:
        data_dir = 'Accelerometerplotter_JSON'
        files = glob.glob(f"{data_dir}/vibration_*.json")
        if not files:
            print("No vibration data files found")
            return None
        
        # Sort by modification time (most recent last)
        files.sort(key=os.path.getmtime)
        json_file = files[-1]
        print(f"Using most recent data file: {json_file}")
    
    # Load the data
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data file: {e}")
        return None
    
    # Extract data from accelerometers
    accel1 = data.get("accelerometer1", [])
    accel2 = data.get("accelerometer2", [])
    
    if not accel1 or not accel2:
        print("No accelerometer data found in file")
        return None
    
    # Extract timestamps and convert to seconds from start
    timestamps1 = np.array([sample["timestamp"] for sample in accel1])
    timestamps1 = (timestamps1 - timestamps1[0]) / 1000000.0  # Convert to seconds
    
    timestamps2 = np.array([sample["timestamp"] for sample in accel2])
    timestamps2 = (timestamps2 - timestamps2[0]) / 1000000.0  # Convert to seconds
    
    # Extract x, y values
    x1 = np.array([sample["x"] for sample in accel1])
    y1 = np.array([sample["y"] for sample in accel1])
    
    x2 = np.array([sample["x"] for sample in accel2])
    y2 = np.array([sample["y"] for sample in accel2])
    
    # Calculate FFT
    x_axis_fft = calculate_fft(x1, x2)
    y_axis_fft = calculate_fft(y1, y2)

    print("Phase difference for X axis:")
    x_phase = calculate_phase_gain_from_fft(x_axis_fft[0], x_axis_fft[1], x_axis_fft[2], x_axis_fft[3])
    print("Phase difference for Y axis:")
    y_phase = calculate_phase_gain_from_fft(y_axis_fft[0], y_axis_fft[1], y_axis_fft[2], y_axis_fft[3])

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plot_fft(x_axis_fft[0], x_axis_fft[1], x_axis_fft[2], x_axis_fft[3])
    plt.title('FFT of X Axis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plot_fft(y_axis_fft[0], y_axis_fft[1], y_axis_fft[2], y_axis_fft[3])
    plt.title('FFT of Y Axis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()

    return x_phase, y_phase

def plot_vibration_data_2(json_file=None):
    """
    Plot the vibration data from a JSON file with both accelerometers on the same plot for each axis.
    If no file is specified, uses the most recent file in the data directory.
    
    Args:
        json_file (str, optional): Path to the JSON file containing vibration data
    """
    # Your existing code for finding files
    # If no file specified, find the most recent one
    if (json_file is None):
        data_dir = 'Accelerometerplotter_JSON'
        files = glob.glob(f"{data_dir}/vibration_*.json")
        if not files:
            print("No vibration data files found")
            # Show available files
            print("\nAvailable data files:")
            all_files = glob.glob(f"{data_dir}/*.json")
            if all_files:
                for i, file in enumerate(all_files):
                    print(f"  {i+1}. {os.path.basename(file)}")
                print("\nTo plot a specific file, use: plot <filename>")
            return
        
        # Sort by modification time (most recent last)
        files.sort(key=os.path.getmtime)
        json_file = files[-1]
        print(f"Using most recent data file: {json_file}")
        
        # Show other available files
        print("\nOther available data files:")
        for i, file in enumerate(reversed(files[:-1])):
            print(f"  {i+1}. {os.path.basename(file)}")
        print("\nTo plot a specific file, use: plot <filename>")
    
    # Load the data
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data file: {e}")
        print(f"Make sure the file exists: {os.path.abspath(json_file)}")
        return
    
    # Extract data from accelerometers
    accel1 = data.get("accelerometer1", [])
    accel2 = data.get("accelerometer2", [])
    
    if not accel1 or not accel2:
        print("No accelerometer data found in file")
        return
    
    # Extract timestamps and convert to seconds from start
    timestamps1 = np.array([sample["timestamp"] for sample in accel1])
    timestamps1 = (timestamps1 - timestamps1[0]) / 1000000.0  # Convert to seconds
    
    timestamps2 = np.array([sample["timestamp"] for sample in accel2])
    timestamps2 = (timestamps2 - timestamps2[0]) / 1000000.0  # Convert to seconds
    
    # Extract x, y values
    x1 = np.array([sample["x"] for sample in accel1])
    y1 = np.array([sample["y"] for sample in accel1])
    
    x2 = np.array([sample["x"] for sample in accel2])
    y2 = np.array([sample["y"] for sample in accel2])
    
    # Clear any existing plots
    plt.close('all')
    
    # Create figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Accelerometer Comparison - {os.path.basename(json_file)}', fontsize=16)
    
    # Plot data for each axis
    axs[0].plot(timestamps1, x1, 'r-', label='Accelerometer 1', linewidth=1.5)

    axs[0].plot(timestamps2, x2, 'b-', label='Accelerometer 2', linewidth=1.5)
    axs[0].set_title('X-Axis')
    axs[0].set_ylabel('Acceleration')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # Y-axis plot
    axs[1].plot(timestamps1, y1, 'r-', label='Accelerometer 1', linewidth=1.5)
    axs[1].plot(timestamps2, y2, 'b-', label='Accelerometer 2', linewidth=1.5)
    axs[1].set_title('Y-Axis')
    axs[1].set_ylabel('Acceleration')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    # Add metadata
    metadata = data.get("metadata", {})
    if metadata:
        sample_count = metadata.get("total_samples", len(accel1))
        timestamp = metadata.get("timestamp", "Unknown")
        frequency = metadata.get("frequency", "Unknown")
        sample_time_us = metadata.get("sample_time_us", "Unknown")
        sample_rate = 1_000_000 / sample_time_us if sample_time_us != "Unknown" else "Unknown"
        
        info_text = f"Timestamp: {timestamp} | Total samples: {sample_count} | Sample rate: {sample_rate} Hz"
        if frequency != "Unknown":
            info_text += f" | Test frequency: {frequency} Hz"
            
        plt.figtext(0.5, 0.01, info_text, 
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

if __name__ == "__main__":
    # Example usage with a specific file
    calculate_amplitude_phase_delay(r"C:\Users\asbjo\Desktop\Sort_ring\Forsøk 2\vibration_2025-04-11_14-51-12_300.0Hz.json")
    plot_vibration_data_2(r"C:\Users\asbjo\Desktop\Sort_ring\Forsøk 2\vibration_2025-04-11_14-51-12_300.0Hz.json")