from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
import json
import os
import glob
import numpy as np
import Accelerometerplotter 

DEFAULT_CONFIG = {
    "sample_rate": 1600,
    "data_dir": "Accelerometerplotter_JSON",
    "figure_size": (6, 6),
    "labels": {
        "accel1": "Reference",
        "accel2": "O-ring"
    }
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
    
    # print(f"Dominant frequency: {dominant_freq:.2f} Hz")
    # print(f"Phase difference: {phase_diff_deg:.2f} degrees")
    # print(f"Gain: {gain:.2f}")
    
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

def display_fft_plot(x_axis_fft, y_axis_fft, config=DEFAULT_CONFIG, json_file=None):
    """
    Display FFT plots for both X and Y axes and save to PNG file.
    
    Args:
        x_axis_fft: FFT data dictionary for X axis
        y_axis_fft: FFT data dictionary for Y axis
        config: Configuration dictionary
        json_file: Path to JSON file (for naming the output PNG)
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
    
    # Save plot as PNG if a JSON file was provided
    if json_file:
        # Create output filename by appending _fft to the original filename
        output_file = os.path.splitext(json_file)[0] + "_fft.png"
        
        # Add a counter if the file already exists
        counter = 1
        base_name = os.path.splitext(json_file)[0]
        while os.path.exists(output_file):
            output_file = f"{base_name}_fft_{counter}.png"
            counter += 1
        
        plt.savefig(output_file, dpi=300)
        print(f"FFT plot saved to: {output_file}")
    
    #plt.show()

def analyze(json_file=None, config=None, doPlot=False):
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
    data = Accelerometerplotter.load_vibration_data(json_file)
    if data is None:
        return None
    
    json_file = data["filename"]
    
    # Calculate FFTs
    x_axis_fft = calculate_fft(data["x1"], data["x2"], config["sample_rate"])
    y_axis_fft = calculate_fft(data["y1"], data["y2"], config["sample_rate"])
    
    # Calculate phase information
    # print("Phase difference for X axis:")
    x_phase = calculate_phase_gain_from_fft(x_axis_fft)
    
    # print("Phase difference for Y axis:")
    y_phase = calculate_phase_gain_from_fft(y_axis_fft)
    
    if doPlot:
        # Display plots
        display_fft_plot(x_axis_fft, y_axis_fft, config, json_file)
    
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