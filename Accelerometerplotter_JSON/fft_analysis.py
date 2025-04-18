from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import json
import os
import glob
import numpy as np

DEFAULT_CONFIG = {
    "sample_rate": 1600,
    "data_dir": "Accelerometerplotter_JSON",
    "figure_size": (12, 6),
    "labels": {
        "accel1": "Reference",
        "accel2": "O-ring"
    }
}

def load_vibration_data(json_file=None):
    """
    Load vibration data from a JSON file or the most recent one.
    
    Args:
        json_file (str, optional): Path to JSON file
        
    Returns:
        dict: Contains processed data or None if loading failed
    """
    # Find file if not specified
    if json_file is None:
        data_dir = 'Accelerometerplotter_JSON'
        files = glob.glob(f"{data_dir}/vibration_*.json")
        if not files:
            print("No vibration data files found")
            return None
        
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
    
    # Extract data
    accel1 = data.get("accelerometer1", [])
    accel2 = data.get("accelerometer2", [])
    
    if not accel1 or not accel2:
        print("No accelerometer data found in file")
        return None
    
    # Process timestamps and acceleration data
    timestamps1 = np.array([sample["timestamp"] for sample in accel1])
    # timestamps1 = (timestamps1 - timestamps1[0]) / 1000000.0 # Convert to seconds
    
    timestamps2 = np.array([sample["timestamp"] for sample in accel2])
    # timestamps2 = (timestamps2 - timestamps2[0]) / 1000000.0 # Convert to seconds
    
    # Extract axis values
    x1 = np.array([sample["x"] for sample in accel1])
    y1 = np.array([sample["y"] for sample in accel1])
    
    x2 = np.array([sample["x"] for sample in accel2])
    y2 = np.array([sample["y"] for sample in accel2])
    
    return {
        "timestamps1": timestamps1,
        "timestamps2": timestamps2,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
        "metadata": data.get("metadata", {}),
        "filename": json_file
    }

def calculate_fft(accel1, accel2, sample_rate=1600):
    """
    Calculate FFT for two acceleration signals.
    
    Args:
        accel1: First accelerometer data
        accel2: Second accelerometer data
        sample_rate: Sampling rate in Hz
        
    Returns:
        dict: Contains frequency axis and FFT results for both signals
    """
    # Perform FFT
    N = len(accel1)
    T = 1.0 / sample_rate
    yf1 = rfft(accel1)
    yf2 = rfft(accel2)
    f_axis = rfftfreq(N, T)[:N//2]
    
    return {
        "f_axis": f_axis,
        "yf1": yf1,
        "yf2": yf2,
        "N": N
    }

def calculate_phase_gain_from_fft(fft_data):
    """
    Calculate phase and gain information from FFT data.
    
    Args:
        fft_data: Dictionary containing FFT results
        
    Returns:
        dict: Phase and gain information
    """
    f_axis = fft_data["f_axis"]
    yf1 = fft_data["yf1"]
    yf2 = fft_data["yf2"]
    N = fft_data["N"]
            
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
    
    return {
        "dominant_freq": dominant_freq,
        "phase_diff_deg": phase_diff_deg,
        "gain": gain,
        "amplitude1": amplitude1,
        "amplitude2": amplitude2,
        "dominant_idx": dominant_idx
    }

def plot_fft(f_axis, yf1, yf2, N, labels=None):
    """
    Plot FFT results with customizable labels.
    
    Args:
        f_axis: Frequency axis values
        yf1: FFT results for first signal
        yf2: FFT results for second signal
        N: Number of samples
        labels: Dict with labels for signals
    """
    if labels is None:
        labels = {"accel1": "Reference", "accel2": "O-ring"}
    
    plt.plot(f_axis, 2.0/N * np.abs(yf1[:N//2]), label=labels["accel1"])
    plt.plot(f_axis, 2.0/N * np.abs(yf2[:N//2]), label=labels["accel2"])

def display_fft_plot(x_axis_fft, y_axis_fft, config=DEFAULT_CONFIG):
    """
    Display FFT plots for both X and Y axes.
    
    Args:
        x_axis_fft: FFT data dictionary for X axis
        y_axis_fft: FFT data dictionary for Y axis
        config: Configuration dictionary
    """
    plt.figure(figsize=config["figure_size"])
    
    plt.subplot(2, 1, 1)
    plot_fft(x_axis_fft["f_axis"], x_axis_fft["yf1"], x_axis_fft["yf2"], 
             x_axis_fft["N"], labels=config["labels"])
    plt.title('FFT of X Axis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plot_fft(y_axis_fft["f_axis"], y_axis_fft["yf1"], y_axis_fft["yf2"],
             y_axis_fft["N"], labels=config["labels"])
    plt.title('FFT of Y Axis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze(json_file=None, config=None):
    """
    Complete analysis workflow for vibration data.
    
    Args:
        json_file: Path to JSON file with vibration data
        config: Configuration dictionary
    
    Returns:
        dict: Results including phase information
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    # Load and prepare data
    data = load_vibration_data(json_file)
    if data is None:
        return None
    
    # Calculate FFTs
    x_axis_fft = calculate_fft(data["x1"], data["x2"], config["sample_rate"])
    y_axis_fft = calculate_fft(data["y1"], data["y2"], config["sample_rate"])
    
    # Calculate phase information
    print("Phase difference for X axis:")
    x_phase = calculate_phase_gain_from_fft(x_axis_fft)
    
    print("Phase difference for Y axis:")
    y_phase = calculate_phase_gain_from_fft(y_axis_fft)
    
    # Display plots
    display_fft_plot(x_axis_fft, y_axis_fft, config)
    
    return {
        "x_phase": x_phase, 
        "y_phase": y_phase,
        "x_fft": x_axis_fft,
        "y_fft": y_axis_fft
    }
if __name__ == "__main__":
    # Example usage with a specific file
    analyze(r"C:\Users\asbjo\Desktop\Sort_ring\Forsøk 2\vibration_2025-04-11_14-51-12_300.0Hz.json")
    #plot_vibration_data_2(r"C:\Users\asbjo\Desktop\Sort_ring\Forsøk 2\vibration_2025-04-11_14-51-12_300.0Hz.json")