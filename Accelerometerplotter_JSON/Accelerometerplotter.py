import paho.mqtt.client as mqtt
import time
import atexit
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib
import threading
import fft_analysis
import communication
matplotlib.use('TkAgg')  # Use TkAgg backend for better performance
plt.ioff()  # Disable interactive mode

def process_vibration_data(data, chunk_num, total_chunks, vibration_metadata):
    """
    Process and store vibration data from chunked MQTT messages.
    Combines chunks into a complete dataset when all chunks are received.
    
    Args:
        data (dict): The JSON data from the current chunk
        chunk_num (int): The current chunk number
        total_chunks (int): Total number of chunks expected
    """
    
    # File to store temporary chunks
    temp_dir = 'Accelerometerplotter_JSON/temp'
    os.makedirs(temp_dir, exist_ok=True)
    chunk_file = f'{temp_dir}/chunk_{chunk_num}.json'
    
    # Save current chunk to temporary file
    with open(chunk_file, 'w') as f:
        json.dump(data, f)
    
    # print(f"Saved chunk {chunk_num}/{total_chunks} to temporary file")
    
    # Check if we have all chunks
    if chunk_num == total_chunks:
        print("All chunks received. Combining data...")
        
        # Get frequency value for filename
        test_frequency = vibration_metadata.get("frequency", 0.0)
        
        # Initialize combined data structure
        combined_data = {
            "accelerometer1": [],
            "accelerometer2": [],
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "total_samples": 0,
                "frequency": test_frequency,
                "sample_time_us": vibration_metadata.get("sampleTime", 0)
            }
        }
        
        # Load and combine all chunks
        for i in range(1, total_chunks + 1):
            chunk_path = f'{temp_dir}/chunk_{i}.json'
            try:
                with open(chunk_path, 'r') as f:
                    chunk_data = json.load(f)
                    
                # Add data from this chunk
                combined_data["accelerometer1"].extend(chunk_data.get("accelerometer1", []))
                combined_data["accelerometer2"].extend(chunk_data.get("accelerometer2", []))
                
                # Clean up temporary chunk file
                os.remove(chunk_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error processing chunk {i}: {e}")
        
        # Update metadata
        combined_data["metadata"]["total_samples"] = len(combined_data["accelerometer1"])
        
        # Generate a timestamp-based filename with frequency included
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        freq_str = f"{test_frequency:.1f}" if test_frequency else "unknown"
        output_file = f'Accelerometerplotter_JSON/vibration_{timestamp}_{freq_str}Hz.json'
        
        # Save the combined data
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"Combined dataset saved to {output_file}")
        print(f"Total samples: {combined_data['metadata']['total_samples']}")
        print(f"Test frequency: {combined_data['metadata']['frequency']} Hz")
        
        # Try to remove temp directory if empty, but handle permission errors
        # try:
        #     if not os.listdir(temp_dir):
        #         os.rmdir(temp_dir)
        # except (PermissionError, OSError) as e:
        #     print(f"Note: Could not remove temporary directory: {e}")
        #     print("This is not critical - processing completed successfully.")
            
        return output_file
    
    return None

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
            # Show available files
            print("\nAvailable data files:")
            all_files = glob.glob(f"{data_dir}/*.json")
            if all_files:
                for i, file in enumerate(all_files):
                    print(f"  {i+1}. {os.path.basename(file)}")
                print("\nTo plot a specific file, use: plot <filename>")
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

def plot_vibration_data(json_file=None):
    """
    Plot the vibration data from a JSON file with both accelerometers on the same plot for each axis.
    If no file is specified, uses the most recent file in the data directory.
    
    Args:
        json_file (str, optional): Path to the JSON file containing vibration data
    """
    # Use the main thread for plotting by scheduling it via a function
    # Your existing code for finding files
    # If no file specified, find the most recent one
    
    # Load the data
    data = load_vibration_data(json_file)
    if data is None:
        print("No data to plot")
        return
    timestamps1 = data["timestamps1"]/1e6 # Convert to seconds
    timestamps2 = data["timestamps2"]/1e6 # Convert to seconds
    time_offset = min(timestamps1[0], timestamps2[0])  # Find the earliest timestamp
    timestamps1 -= time_offset
    timestamps2 -= time_offset
    x1 = data["x1"]
    y1 = data["y1"]
    x2 = data["x2"]
    y2 = data["y2"]
    metadata = data["metadata"]
    json_file = data["filename"]
    # Get phase analysis data
    phase_data = fft_analysis.analyze(json_file)
    
    # Clear any existing plots
    plt.close('all')
    
    # Create figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    fig.suptitle(f'{os.path.basename(json_file)}', fontsize=16)
    
    # Plot data for each axis
    axs[0].plot(timestamps1, x1, 'r-', label='Accelerometer 1', linewidth=1.5)
    axs[0].plot(timestamps2, x2, 'b-', label='Accelerometer 2', linewidth=1.5)
    axs[0].set_title('X-Axis')
    axs[0].set_ylabel('Acceleration')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # Add phase and amplitude info to X-axis plot
    if phase_data and "x_phase" in phase_data:
        x_info = phase_data["x_phase"]
        axs[0].text(0.02, 0.95, 
                   f"Frequency: {x_info['dominant_freq']:.2f} Hz\n"
                   f"Phase (FFT): {x_info['phase_diff_deg']:.2f}°\n"
                   f"Amplitude ratio: {x_info['gain']:.2f}", 
                   transform=axs[0].transAxes,
                   bbox={"facecolor":"lightgrey", "alpha":0.7, "pad":5},
                   verticalalignment='top')
    
    # Y-axis plot
    axs[1].plot(timestamps1, y1, 'r-', label='Accelerometer 1', linewidth=1.5)
    axs[1].plot(timestamps2, y2, 'b-', label='Accelerometer 2', linewidth=1.5)
    axs[1].set_title('Y-Axis')
    axs[1].set_ylabel('Acceleration')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    # Add phase and amplitude info to Y-axis plot
    if phase_data and "y_phase" in phase_data:
        y_info = phase_data["y_phase"]
        axs[1].text(0.02, 0.95, 
                   f"Frequency: {y_info['dominant_freq']:.2f} Hz\n"
                   f"Phase (FFT): {y_info['phase_diff_deg']:.2f}°\n"
                   f"Amplitude ratio: {y_info['gain']:.2f}", 
                   transform=axs[1].transAxes,
                   bbox={"facecolor":"lightgrey", "alpha":0.7, "pad":5},
                   verticalalignment='top')
    
    # Add metadata
    metadata = data.get("metadata", {})
    # if metadata:
    #     sample_count = metadata.get("total_samples", len(x1))
    #     timestamp = metadata.get("timestamp", "Unknown")
    #     frequency = metadata.get("frequency", "Unknown")
    #     sample_time_us = metadata.get("sample_time_us", "Unknown")
    #     sample_rate = 1_000_000 / sample_time_us if sample_time_us != "Unknown" else "Unknown"
        
    #     info_text = f"Total samples: {sample_count} | Sample rate: {sample_rate} Hz"
    #     if frequency != "Unknown":
    #         info_text += f" | Test frequency: {frequency} Hz"
            
    #     plt.figtext(0.5, 0.01, info_text, 
    #                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot with the same base name as the JSON file
    output_file = os.path.splitext(json_file)[0] + '_plt.png'
    try:
        # Add a counter if the file already exists
        counter = 1
        base_name = os.path.splitext(json_file)[0]
        while os.path.exists(output_file):
            output_file = f"{base_name}_plt_{counter}.png"
            counter += 1

        fig.savefig(output_file, dpi=300, bbox_inches='tight')

        print(f"Plot saved to: {output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")

def test_frequency_range(client, start_freq, end_freq, step_freq, ring_id):
    """
    Test a range of frequencies, automatically running tests and plotting results.
    
    Args:
        client: MQTT client object
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz (inclusive)
        step_freq: Step size between frequencies in Hz
        ring_id: Identifier for the ring being tested (used for organizing results)
    
    Returns:
        list: List of dictionaries containing test results for each frequency
    """
    import time
    import os
    import glob
    
    # Check if client is connected
    if not client or not client.is_connected():
        print("MQTT client not connected. Cannot run tests.")
        return []
    
    # Validate parameters
    if start_freq <= 0 or end_freq <= 0 or step_freq <= 0:
        print("Error: Frequencies must be positive values")
        return []
        
    if start_freq < end_freq:
        print("Error: Start frequency must be greater than end frequency")
        return []
    
    # Create ring-specific directory
    base_dir = 'Accelerometerplotter_JSON'
    ring_dir = os.path.join(base_dir, f"ring_{ring_id}")
    os.makedirs(ring_dir, exist_ok=True)
    print(f"Saving results to: {ring_dir}")
    
    # Store results
    test_results = []
    
    # Calculate number of tests for progress tracking
    num_tests = int((start_freq - end_freq) / step_freq) + 1
    current_test = 1
    
    print(f"Total tests: {num_tests}")
    
    # For each frequency
    curr_freq = start_freq
    while curr_freq >= end_freq:
        print(f"\n[{current_test}/{num_tests}] Testing {curr_freq} Hz...")
        
        # Run test at this frequency
        client.publish("vibration/test", str(curr_freq))
        
        # Wait for test to complete and data to be processed
        # Since MQTT is asynchronous, we need to wait for the file to appear
        # Get the count of existing files
        initial_file_count = len(glob.glob(f"{base_dir}/vibration_*.json"))
        
        print("Test initiated, waiting for data collection...")
        
        # Wait for new file to appear (max 20 seconds)
        wait_time = 0
        new_file = None
        max_wait = 20  # Maximum wait time in seconds
        
        while wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            
            # Check for new files
            current_files = glob.glob(f"{base_dir}/vibration_*.json")
            if len(current_files) > initial_file_count:
                # Sort by modification time to find the most recent
                current_files.sort(key=os.path.getmtime)
                new_file = current_files[-1]
                print(f"Data received: {os.path.basename(new_file)}")
                break
                
            if wait_time % 5 == 0:
                print(f"Still waiting for data... ({wait_time}s)")
        
        if new_file:
            # Wait a bit more for any processing to finish
            time.sleep(1)
            
            # Copy the file to the ring-specific directory with a ring-specific name
            ring_specific_file = os.path.join(ring_dir, f"ring_{ring_id}_{os.path.basename(new_file)}")
            
            # Copy file contents to ring-specific directory
            try:
                with open(new_file, 'r') as src_file, open(ring_specific_file, 'w') as dst_file:
                    dst_file.write(src_file.read())
                print(f"Data saved to: {ring_specific_file}")
            except Exception as e:
                print(f"Error copying data file to ring directory: {e}")
                ring_specific_file = new_file  # Fall back to original file
            
            # Analyze the data
            phase_results = fft_analysis.analyze(ring_specific_file, doPlot=True)
            
            # Store results
            test_results.append({
                'frequency': curr_freq,
                'file': ring_specific_file,
                'phase': phase_results
            })
            
            # Plot the data and save to ring directory
            # plot_file = os.path.splitext(ring_specific_file)[0] + '_plt.png'
            plot_vibration_data(ring_specific_file)
        else:
            print(f"No data received for {curr_freq} Hz after {max_wait} seconds")
        
        # Move to next frequency
        curr_freq -= step_freq
        current_test += 1
    
    print("\nFrequency sweep completed.")
    
    # Summarize results
    print("\n=== TEST SUMMARY ===")
    print(f"Tested {len(test_results)} frequencies from {start_freq} to {end_freq} Hz")
    
    if test_results:
        print("\nFrequency   |   X Gain   |   Y Gain   |   X Phase (deg)   |   Y Phase (deg)   |   Filename")
        print("-" * 75)
        
        # Create CSV file for the results with timestamp and ring ID in filename
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = os.path.join(ring_dir, f"ring_{ring_id}_sweep_{start_freq}-{end_freq}Hz_{timestamp}.csv")
        
        with open(csv_filename, 'w') as csvfile:
            csvfile.write("Frequency,X Gain,Y Gain,X Phase (deg),Y Phase (deg),Filename\n")
            
            for result in test_results:
                if phase_x < 0: 
                    phase_x += 360
                if phase_y < 0:
                    phase_y += 360
                freq = result['phase']['x_phase']['dominant_freq']
                gain_x = result['phase']['x_phase']['gain']
                gain_y = result['phase']['y_phase']['gain']
                phase_x = result['phase']['x_phase']["phase_diff_deg"]
                phase_y = result['phase']['y_phase']["phase_diff_deg"]
                filename = os.path.basename(result['file'])
                
                # Print to console
                print(f"{freq:8.2f}   |   {gain_x:6.2f}   |   {gain_y:6.2f}   |   {phase_x:12.2f}   |   {phase_y:12.2f}   |   {filename}")
                
                # Write to CSV
                csvfile.write(f"{freq},{gain_x},{gain_y},{phase_x},{phase_y},{filename}\n")
        
        print(f"\nTest results saved to: {csv_filename}")
    
    return test_results

def create_bode_plot(csv_file=None):
    """
    Create a Bode plot from frequency sweep test results.
    
    Args:
        csv_file (str, optional): Path to CSV file with sweep results. 
                                 If None, uses the most recent CSV file.
    
    Returns:
        str: Path to saved plot image or None if error
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import glob
    import os
    
    # Find the most recent CSV file if none specified
    if csv_file is None:
        all_rings = glob.glob('Accelerometerplotter_JSON/Ring_*/ring*_sweep_*.csv')
        if not all_rings:
            print("No frequency sweep CSV files found")
            return None
        
        all_rings.sort(key=os.path.getmtime, reverse=True)
        csv_file = all_rings[0]
        print(f"Using most recent sweep file: {csv_file}")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        return None
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
        # Sort by frequency for proper plotting
        df = df.sort_values('Frequency')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Check if expected columns exist
    required_columns = ['Frequency', 'X Gain', 'Y Gain', 'X Phase (deg)', 'Y Phase (deg)']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return None
    
    # Create figure with two subplots (magnitude and phase)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Bode Plot', fontsize=16)
    
    # Plot magnitude (gain)
    ax1.semilogx(df['Frequency'], 20*np.log10(df['X Gain']), 'r-o', label='X Axis', markersize=4)
    ax1.semilogx(df['Frequency'], 20*np.log10(df['Y Gain']), 'b-o', label='Y Axis', markersize=4)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Magnitude Response')
    ax1.grid(True, which="both", ls="-", alpha=0.7)
    ax1.legend()
    
    # Plot phase
    ax2.semilogx(df['Frequency'], df['X Phase (deg)'], 'r-o', label='X Axis', markersize=4)
    ax2.semilogx(df['Frequency'], df['Y Phase (deg)'], 'b-o', label='Y Axis', markersize=4)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Phase Response')
    ax2.grid(True, which="both", ls="-", alpha=0.7)
    ax2.legend()
    
    # Extract ring ID and frequency range from filename for the plot title
    filename = os.path.basename(csv_file)
    try:
        # Parse filename to get ring ID and frequency range
        parts = filename.split('_')
        ring_id = parts[1]
        freq_range = parts[3].replace('Hz', '')
        fig.suptitle(f'Bode Plot - Ring {ring_id} ({freq_range} Hz)', fontsize=16)
    except:
        # Use generic title if parsing fails
        pass
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.splitext(csv_file)[0] + '_bode.png'
    
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Bode plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving Bode plot: {e}")
        return None
        
    plt.close(fig)
    return plot_path

def main():
    import sys
    # Set up MQTT connection to broker
    client = communication.connect_mqtt(broker_address="192.168.68.128", port=1883, 
                         client_id="AccelerometerPlotter", 
                         keepalive=15)  # Reduced keepalive time

    if not client:
        print("Failed to establish initial connection to MQTT broker. Exiting.")
        return

    communication.setup_mqtt_callbacks(client)
    
    # Create a separate thread for heartbeat
    def heartbeat_thread():
        last_heartbeat = time.time()
        heartbeat_interval = 30  # Reduced interval for more frequent checks
        
        while True:
            try:
                current_time = time.time()
                
                # Send heartbeat every 10 seconds
                if current_time - last_heartbeat > heartbeat_interval:
                    if client.is_connected():
                        # print("Sending heartbeat...")
                        client.publish("vibration/heartbeat", str(current_time), qos=1)
                        last_heartbeat = current_time
                    else:
                        print("Connection lost. Attempting to reconnect...")
                        try:
                            client.reconnect()
                            # Re-subscribe on successful reconnection
                            if client.is_connected():
                                communication.setup_mqtt_callbacks(client)
                                print("Reconnected and resubscribed to topics")
                        except Exception as e:
                            print(f"Reconnection failed: {e}")
                
                time.sleep(2)  # Check connection every 2 seconds
                
            except Exception as e:
                print(f"Error in heartbeat thread: {e}")
                time.sleep(5)  # Wait before retrying after an error
    
    # Start heartbeat in a daemon thread
    hb_thread = threading.Thread(target=heartbeat_thread, daemon=True)
    hb_thread.start()
    
    # Main program logic
    try:
        print("Program running. Press Ctrl+C to exit.")
        # Use your original command listener
        communication.listen_for_commands(client)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Ensure clean disconnect
        print("Shutting down...")
        try:
            client.publish("vibration/status", "offline", qos=1, retain=True)
            client.loop_stop()
            client.disconnect()
        except:
            pass


if __name__ == "__main__":
    main()
