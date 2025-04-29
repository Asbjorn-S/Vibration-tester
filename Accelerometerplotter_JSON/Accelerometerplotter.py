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
from communication import received_metadata
matplotlib.use('Agg')  # Use non-interactive backend for plots
plt.ioff()  # Disable interactive mode

onlineMode = False  # Set to True if using MQTT broker for real-time data

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
    
    # Reset metadata flag before testing
    import communication
    communication.received_metadata = False

    new_file = None

    def send_test_command(freq, max_retries=3):
        initial_file_count = len(glob.glob(f"{base_dir}/vibration_*.json"))
        initial_metadata_received = False
        
        for attempt in range(1, max_retries + 1):
            print(f"Sending test command for {freq} Hz (attempt {attempt}/{max_retries})...")
            
            # Send command with QoS 2
            msg_info = client.publish("vibration/test", str(freq), qos=2)
            msg_delivered = msg_info.wait_for_publish(timeout=2)
            
            if not msg_delivered:
                print(f"Warning: Message delivery not confirmed, retrying...")
                # Continue waiting for data even if message delivery isn't confirmed
                # The message might still have been delivered
            
            # Wait for data
            wait_time = 0
            while wait_time < 10:  # Shortened timeout since we're checking for both metadata and files
                time.sleep(1)
                wait_time += 1
                
                # Check for new data files
                current_files = glob.glob(f"{base_dir}/vibration_*.json")
                if len(current_files) > initial_file_count:
                    newest_file = max(current_files, key=os.path.getmtime)
                    print(f"New data file detected: {os.path.basename(newest_file)}")
                    time.sleep(2)  # Give it a moment to complete writing
                    return True, newest_file
                
                # Also check if metadata was received, which indicates test started
                global received_metadata
                if received_metadata and not initial_metadata_received:
                    initial_metadata_received = True
                    print("Test in progress (metadata received)...")
                    
                # Status update
                if wait_time % 5 == 0 and wait_time > 0:
                    print(f"Still waiting for data... ({wait_time}s)")
                    
                    # Check if client is still connected
                    if not client.is_connected():
                        print("MQTT connection lost during wait! Reconnecting...")
                        try:
                            client.reconnect()
                        except Exception as e:
                            print(f"Reconnect failed: {e}")
                            
            # If we reach here with metadata but no file, wait a bit longer
            if initial_metadata_received:
                print("Metadata received but still waiting for data file...")
                
                # Extended wait for the file
                for extended_wait in range(10):
                    time.sleep(1)
                    current_files = glob.glob(f"{base_dir}/vibration_*.json")
                    if len(current_files) > initial_file_count:
                        newest_file = max(current_files, key=os.path.getmtime)
                        return True, newest_file
                        
                print("Timed out waiting for data file after metadata was received")
                
        return False, None  # Failed after all retries
    
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
    
    # Find the next available test number
    test_dirs = glob.glob(os.path.join(ring_dir, "test_*"))
    if test_dirs:
        test_numbers = []
        for d in test_dirs:
            # Only consider directories, not files
            if os.path.isdir(d):
                try:
                    # Extract the number after "test_"
                    test_num_str = os.path.basename(d).split("_")[-1] 
                    test_numbers.append(int(test_num_str))
                except ValueError:
                    # Skip if conversion to int fails
                    continue
        test_num = max(test_numbers) + 1 if test_numbers else 1
    else:
        test_num = 1
    
    # Create test-specific directory
    test_dir = os.path.join(ring_dir, f"test_{test_num}")
    os.makedirs(test_dir, exist_ok=True)
    print(f"Saving results to: {test_dir}")
    
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
        
        # Wait for test to complete and data to be processed
        # Since MQTT is asynchronous, we need to wait for the file to appear
        # Get the count of existing files
        
        max_wait = 20  # Maximum wait time in seconds
        success, new_file = send_test_command(curr_freq)
        if success:
        # Process the results...
            if new_file:
                # Wait a bit more for any processing to finish
                time.sleep(1)

                # Keep original filename but place in test-specific directory
                ring_specific_file = os.path.join(test_dir, f"ring_{ring_id}_{os.path.basename(new_file)}")

                # Copy file contents to test directory
                try:
                    with open(new_file, 'r') as src_file, open(ring_specific_file, 'w') as dst_file:
                        dst_file.write(src_file.read())
                    print(f"Data saved to: {ring_specific_file}")

                    # Delete the original file after successful copy
                    try:
                        os.remove(new_file)
                        print(f"Original file removed: {new_file}")
                    except Exception as e:
                        print(f"Warning: Could not remove original file: {e}")

                except Exception as e:
                    print(f"Error copying data file to test directory: {e}")
                    ring_specific_file = new_file  # Fall back to original file

                # Analyze the data and create fft plot
                phase_results = fft_analysis.analyze(ring_specific_file, doPlot=True)

                # Store results
                test_results.append({
                    'frequency': curr_freq,
                    'file': ring_specific_file,
                    'phase': phase_results
                })

                # Plot the data and save to test directory (plot_vibration_data saves to same dir as source)
                plot_vibration_data(ring_specific_file)
            else:
                print(f"No data received for {curr_freq} Hz after {max_wait} seconds")

            # Move to next frequency
            curr_freq -= step_freq
            current_test += 1
        else:
            print(f"Test failed for {curr_freq} Hz after multiple attempts")
    
    print("\nFrequency sweep completed.")
    
    # Summarize results
    print("\n=== TEST SUMMARY ===")
    print(f"Tested {len(test_results)} frequencies from {start_freq} to {end_freq} Hz")
    
    if test_results:
        print("\nFrequency   |   X Gain   |   Y Gain   |   X Phase (deg)   |   Y Phase (deg)   |   Filename")
        print("-" * 75)
        
        # Create CSV file for the results with test number in the ring directory
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        csv_filename = os.path.join(ring_dir, f"test_{test_num}_ring_{ring_id}_sweep_{start_freq}-{end_freq}Hz_{timestamp}.csv")
        
        with open(csv_filename, 'w') as csvfile:
            csvfile.write("Frequency,X Gain,Y Gain,X Phase (deg),Y Phase (deg),Filename\n")
            
            for result in test_results:
                try:
                    # Check if phase data is available
                    if not result['phase'] or 'x_phase' not in result['phase'] or 'y_phase' not in result['phase']:
                        print(f"Warning: Missing phase data for {freq} Hz test, skipping in CSV")
                        continue
                    freq = result['phase']['x_phase']['dominant_freq']
                    gain_x = result['phase']['x_phase']['gain']
                    gain_y = result['phase']['y_phase']['gain']
                    phase_x = result['phase']['x_phase']["phase_diff_deg"]
                    phase_y = result['phase']['y_phase']["phase_diff_deg"]
                    
                    # Convert negative phases to 0-360 range
                    if phase_x < 0: 
                        phase_x += 360
                    if phase_y < 0:
                        phase_y += 360
                        
                    filename = os.path.basename(result['file'])
                    
                    # Print to console
                    print(f"{freq:8.2f}   |   {gain_x:6.2f}   |   {gain_y:6.2f}   |   {phase_x:12.2f}   |   {phase_y:12.2f}   |   {filename}")
                    
                    # Write to CSV
                    csvfile.write(f"{freq},{gain_x},{gain_y},{phase_x},{phase_y},{filename}\n")
                except Exception as e:
                    print(f"Error processing result for frequency {result.get('frequency', 'unknown')}: {e}")
        
        print(f"\nTest results saved to: {csv_filename}")
        
        # Create a Bode plot automatically for this sweep
        # bode_plot_path = create_bode_plot(csv_filename)
        # if bode_plot_path:
        #     print(f"Bode plot created: {bode_plot_path}")
    
    return test_results

def create_bode_plot(csv_file=None, interval=25):
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
        ring_id = parts[3]
        freq_range = parts[5].replace('Hz', '')
        fig.suptitle(f'Bode Plot - Ring {ring_id} ({freq_range} Hz)', fontsize=16)
    except:
        # Use generic title if parsing fails
        pass

    # Set custom frequency ticks at specified interval
    min_freq = df['Frequency'].min()
    max_freq = df['Frequency'].max()
    
    # Generate tick locations at the specified interval
    # Round min/max to nearest interval for nice bounds
    start_tick = int(min_freq / interval) * interval
    if start_tick < min_freq:
        start_tick += interval
        
    end_tick = int(max_freq / interval) * interval
    if end_tick > max_freq:
        end_tick -= interval
        
    # Create frequency ticks
    freq_ticks = np.arange(start_tick, end_tick + interval, interval)
    
    # Apply to both axes (they share x-axis)
    ax2.set_xticks(freq_ticks)
    ax2.set_xticklabels([f"{x}" for x in freq_ticks])
    ax2.minorticks_off()  # Turn off minor ticks for cleaner look
    
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

def create_combined_bode_plot(csv_files=None, output_name="combined_bode_plot", directory_path=None, interval=25):
    """
    Create Bode plots with multiple test results overlaid for comparison.
    
    Args:
        csv_files (list, optional): List of CSV file paths to include in the plot.
                                   If None, asks user to select from available files.
        output_name (str): Base name for output plot files
        directory_path (str, optional): Directory to search for CSV files
                                       If None, uses the Accelerometerplotter_JSON directory.
    
    Returns:
        tuple: Paths to created plot files (x_axis, y_axis)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import glob
    import os
    
    # Default to Accelerometerplotter_JSON directory if none specified
    if directory_path is None:
        directory_path = 'Accelerometerplotter_JSON'
    
    # If no files specified, find and list available ones
    if csv_files is None:
        all_csvs = glob.glob(os.path.join(directory_path, '**', '*sweep*.csv'), recursive=True)
        if not all_csvs:
            print(f"No frequency sweep CSV files found in {directory_path}")
            return None, None
        
        # Show available files to choose from
        print("\nAvailable frequency sweep files:")
        for i, file in enumerate(all_csvs):
            print(f"[{i+1}] {os.path.basename(file)}")
        
        print("\n[0] Select ALL files")
        
        # Get user selection
        try:
            selection_input = input("\nEnter file numbers to compare (comma-separated, e.g., 1,3,4) or 0 for all: ")
            
            # Check if user wants all files
            if selection_input.strip() == "0":
                csv_files = all_csvs
                print(f"Selected all {len(csv_files)} files")
            else:
                indices = [int(x.strip())-1 for x in selection_input.split(",") if x.strip()]
                csv_files = [all_csvs[i] for i in indices if 0 <= i < len(all_csvs)]
            
            if not csv_files:
                print("No valid files selected")
                return None, None
                
        except (ValueError, IndexError) as e:
            print(f"Error selecting files: {e}")
            return None, None
    
    # Colors for different test data
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', '*', '+', 'x']
    
    # Create output directory
    output_dir = os.path.join(directory_path, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create two figures: X-axis only, Y-axis only
    fig_x, (ax1_x, ax2_x) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig_y, (ax1_y, ax2_y) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Title for the plots
    fig_x.suptitle('X-Axis Bode Plot Comparison', fontsize=16)
    fig_y.suptitle('Y-Axis Bode Plot Comparison', fontsize=16)
    
    # Legend entries
    legend_entries = []
    
    # Load and plot each dataset
    for i, csv_file in enumerate(csv_files):
        try:
            # Pick color, line style and marker based on index
            color = colors[i % len(colors)]
            line_style = line_styles[(i // len(colors)) % len(line_styles)]
            marker = markers[(i // (len(colors) * len(line_styles))) % len(markers)]
            
            # Don't combine them into a single format string
            # Instead, pass them as separate parameters
            
            # Load data
            df = pd.read_csv(csv_file)
            df = df.sort_values('Frequency')
            
            # Check required columns
            required_columns = ['Frequency', 'X Gain', 'Y Gain', 'X Phase (deg)', 'Y Phase (deg)']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                print(f"Warning: File {os.path.basename(csv_file)} is missing required columns: {missing}")
                continue
            
            # Extract test info from filename
            filename = os.path.basename(csv_file)
            try:
                parts = filename.split('_')
                ring_id = parts[3] if len(parts) > 3 else "unknown"
                test_num = parts[1] if len(parts) > 1 else "unknown"
                legend_name = f"Ring {ring_id} - Test {test_num}"
            except:
                legend_name = os.path.splitext(filename)[0]
                
            legend_entries.append(legend_name)
            
            # Plot X-axis only - passing color, linestyle, marker separately
            ax1_x.semilogx(df['Frequency'], 20*np.log10(df['X Gain']), 
                          color=color, linestyle=line_style, marker=marker, 
                          label=legend_name, markersize=4)
            ax2_x.semilogx(df['Frequency'], df['X Phase (deg)'], 
                          color=color, linestyle=line_style, marker=marker, 
                          label=legend_name, markersize=4)
            
            # Plot Y-axis only - passing color, linestyle, marker separately
            ax1_y.semilogx(df['Frequency'], 20*np.log10(df['Y Gain']), 
                          color=color, linestyle=line_style, marker=marker, 
                          label=legend_name, markersize=4)
            ax2_y.semilogx(df['Frequency'], df['Y Phase (deg)'], 
                          color=color, linestyle=line_style, marker=marker, 
                          label=legend_name, markersize=4)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_file)}: {e}")
    
    # Configure X-axis plot
    ax1_x.set_ylabel('Magnitude (dB)')
    ax1_x.set_title('X-Axis Magnitude Response')
    ax1_x.grid(True, which="both", ls="-", alpha=0.7)
    ax1_x.legend(loc='upper right', fontsize='small')
    
    ax2_x.set_xlabel('Frequency (Hz)')
    ax2_x.set_ylabel('Phase (degrees)')
    ax2_x.set_title('X-Axis Phase Response')
    ax2_x.grid(True, which="both", ls="-", alpha=0.7)
    ax2_x.legend(loc='upper right', fontsize='small')
    
    # Configure Y-axis plot
    ax1_y.set_ylabel('Magnitude (dB)')
    ax1_y.set_title('Y-Axis Magnitude Response')
    ax1_y.grid(True, which="both", ls="-", alpha=0.7)
    ax1_y.legend(loc='upper right', fontsize='small')
    
    ax2_y.set_xlabel('Frequency (Hz)')
    ax2_y.set_ylabel('Phase (degrees)')
    ax2_y.set_title('Y-Axis Phase Response')
    ax2_y.grid(True, which="both", ls="-", alpha=0.7)
    ax2_y.legend(loc='upper right', fontsize='small')
    
    # Add information about the test count
    test_count_info = f"Comparing {len(csv_files)} tests"
    fig_x.text(0.5, 0.01, test_count_info, ha='center', fontsize=10)
    fig_y.text(0.5, 0.01, test_count_info, ha='center', fontsize=10)
    
    # Generate unique output names with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M")
    x_plot_path = os.path.join(output_dir, f"{output_name}_x_axis_{timestamp}.png")
    y_plot_path = os.path.join(output_dir, f"{output_name}_y_axis_{timestamp}.png")

    # Find overall min and max frequencies across all datasets
    min_freq = float('inf')
    max_freq = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            min_freq = min(min_freq, df['Frequency'].min())
            max_freq = max(max_freq, df['Frequency'].max())
        except:
            pass
    
    # Generate tick locations at the specified interval
    # Round min/max to nearest interval for nice bounds
    start_tick = int(min_freq / interval) * interval
    if start_tick < min_freq:
        start_tick += interval
        
    end_tick = int(max_freq / interval) * interval
    if end_tick > max_freq:
        end_tick -= interval
    
    # Create frequency ticks
    freq_ticks = np.arange(start_tick, end_tick + interval, interval)
    
    # Apply to both plots
    # X-axis plot
    ax2_x.set_xticks(freq_ticks)
    ax2_x.set_xticklabels([f"{x}" for x in freq_ticks])
    ax2_x.minorticks_off()
    
    # Y-axis plot
    ax2_y.set_xticks(freq_ticks)
    ax2_y.set_xticklabels([f"{x}" for x in freq_ticks])
    ax2_y.minorticks_off()
    
    try:
        # Save the plots
        fig_x.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_x.savefig(x_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_x)
        
        fig_y.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_y.savefig(y_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_y)
        
        print(f"\nPlots saved to: {output_dir}")
        print(f"X-axis plot: {os.path.basename(x_plot_path)}")
        print(f"Y-axis plot: {os.path.basename(y_plot_path)}")
        
    except Exception as e:
        print(f"Error saving plots: {e}")
        return None, None
    
    return x_plot_path, y_plot_path

def calculate_sweep_statistics(csv_files=None, output_name="avg_sweep", directory_path=None):
    """
    Calculate average and standard deviation from multiple frequency sweep CSV files.
    Handles frequency mismatches through interpolation.
    
    Args:
        csv_files (list, optional): List of CSV file paths to analyze.
                                   If None, asks user to select from available files.
        output_name (str): Base name for output CSV file
        directory_path (str, optional): Directory to search for CSV files
                                       If None, uses the Accelerometerplotter_JSON directory.
    
    Returns:
        str: Path to the created statistics CSV file or None if error
    """
    import pandas as pd
    import numpy as np
    import glob
    import os
    import time
    from scipy.interpolate import interp1d
    
    # Default to Accelerometerplotter_JSON directory if none specified
    if directory_path is None:
        directory_path = 'Accelerometerplotter_JSON'
    
    # Extract ring_id from directory path if it follows the pattern "ring_{ring_id}"
    ring_id = "unknown"
    dir_name = os.path.basename(directory_path)
    if dir_name.startswith("ring_"):
        ring_id = dir_name.split("_", 1)[1]  # Get everything after "ring_"
    
    # If no files specified, find and list available ones
    if csv_files is None:
        all_csvs = glob.glob(os.path.join(directory_path, '**', '*sweep*.csv'), recursive=True)
        if not all_csvs:
            print(f"No frequency sweep CSV files found in {directory_path}")
            return None
        
        # Show available files to choose from
        print("\nAvailable frequency sweep files:")
        for i, file in enumerate(all_csvs):
            print(f"[{i+1}] {os.path.basename(file)}")
        
        print("\n[0] Select ALL files")
        
        # Get user selection
        try:
            selection_input = input("\nEnter file numbers to analyze (comma-separated, e.g., 1,3,4) or 0 for all: ")
            
            # Check if user wants all files
            if selection_input.strip() == "0":
                csv_files = all_csvs
                print(f"Selected all {len(csv_files)} files")
            else:
                indices = [int(x.strip())-1 for x in selection_input.split(",") if x.strip()]
                csv_files = [all_csvs[i] for i in indices if 0 <= i < len(all_csvs)]
            
            if not csv_files:
                print("No valid files selected")
                return None
                
        except (ValueError, IndexError) as e:
            print(f"Error selecting files: {e}")
            return None
    
    if len(csv_files) < 2:
        print("Need at least two CSV files for statistical analysis")
        return None
    
    # Step 1: Load all dataframes and determine frequency range
    dataframes = []
    min_freq = float('inf')
    max_freq = 0
    
    print(f"\nLoading and validating {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check required columns
            required_columns = ['Frequency', 'X Gain', 'Y Gain', 'X Phase (deg)', 'Y Phase (deg)']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                print(f"Warning: File {os.path.basename(csv_file)} is missing required columns: {missing}")
                continue
                
            # Sort by frequency (just to be sure)
            df = df.sort_values('Frequency')
            
            # Update overall min/max frequency
            min_freq = min(min_freq, df['Frequency'].min())
            max_freq = max(max_freq, df['Frequency'].max())
            
            dataframes.append(df)
            print(f"  ✓ Loaded {os.path.basename(csv_file)} - {len(df)} frequency points")
            
        except Exception as e:
            print(f"  ✗ Error reading {os.path.basename(csv_file)}: {e}")
    
    if not dataframes:
        print("No valid data found in provided CSV files")
        return None
        
    print(f"\nIdentified frequency range: {min_freq:.1f}Hz to {max_freq:.1f}Hz")
    
    # Step 2: Create a common frequency grid
    # Find the dataset with the most frequency points for reference
    reference_df = max(dataframes, key=len)
    
    # Option 1: Use frequencies from the dataset with the most points
    common_frequencies = reference_df['Frequency'].values
    
    # Option 2: Create a uniform grid (uncomment if preferred)
    # step_size = 5.0  # Hz
    # common_frequencies = np.arange(min_freq, max_freq + step_size, step_size)
    
    # Step 3: Interpolate each dataset to the common frequency grid
    print(f"Interpolating all datasets to common frequency points...")
    
    # Create storage for interpolated values
    all_x_gains = []
    all_y_gains = []
    all_x_phases = []
    all_y_phases = []
    
    for i, df in enumerate(dataframes):
        try:
            # Create interpolation functions for each column
            x_gain_interp = interp1d(df['Frequency'], df['X Gain'], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            y_gain_interp = interp1d(df['Frequency'], df['Y Gain'], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            x_phase_interp = interp1d(df['Frequency'], df['X Phase (deg)'], 
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
            y_phase_interp = interp1d(df['Frequency'], df['Y Phase (deg)'], 
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Apply interpolation to common frequency grid
            all_x_gains.append(x_gain_interp(common_frequencies))
            all_y_gains.append(y_gain_interp(common_frequencies))
            all_x_phases.append(x_phase_interp(common_frequencies))
            all_y_phases.append(y_phase_interp(common_frequencies))
            
        except Exception as e:
            print(f"  ✗ Error interpolating dataset {i+1}: {e}")
    
    # Convert to numpy arrays for calculations
    x_gains = np.array(all_x_gains)
    y_gains = np.array(all_y_gains)
    x_phases = np.array(all_x_phases)
    y_phases = np.array(all_y_phases)
    
    # Calculate statistics
    print("Calculating statistics...")
    
    # Calculate means
    mean_x_gain = np.mean(x_gains, axis=0)
    mean_y_gain = np.mean(y_gains, axis=0)
    mean_x_phase = np.mean(x_phases, axis=0)
    mean_y_phase = np.mean(y_phases, axis=0)
    
    # Calculate standard deviations
    std_x_gain = np.std(x_gains, axis=0, ddof=1)  # Using ddof=1 for sample std
    std_y_gain = np.std(y_gains, axis=0, ddof=1)
    std_x_phase = np.std(x_phases, axis=0, ddof=1)
    std_y_phase = np.std(y_phases, axis=0, ddof=1)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'Frequency': common_frequencies,
        'Mean X Gain': mean_x_gain,
        'Std X Gain': std_x_gain,
        'Mean Y Gain': mean_y_gain,
        'Std Y Gain': std_y_gain,
        'Mean X Phase (deg)': mean_x_phase,
        'Std X Phase (deg)': std_x_phase,
        'Mean Y Phase (deg)': mean_y_phase,
        'Std Y Phase (deg)': std_y_phase
    })
    
    # Create output directory
    output_dir = os.path.join('Accelerometerplotter_JSON', 'statistics')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV using the desired format with ring_id
    timestamp = time.strftime("%Y%m%d_%H%M")
    filename = f"{output_name}_{ring_id}_{timestamp}.csv"
    output_file = os.path.join(output_dir, filename)
    
    output_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\nStatistics saved to: {output_file}")
    
    # Also create a plot of the mean with shaded error bands
    try:
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig_title = f'Frequency Statistics - Ring {ring_id}'
        fig.suptitle(fig_title, fontsize=16)
        
        # Convert gain to dB for plotting
        mean_x_gain_db = 20 * np.log10(mean_x_gain)
        
        # Calculate upper and lower bounds for gain in dB
        # We need to convert to dB separately to handle the logarithmic relationship correctly
        x_gain_upper_db = 20 * np.log10(mean_x_gain + std_x_gain)
        x_gain_lower_db = 20 * np.log10(np.maximum(mean_x_gain - std_x_gain, 1e-10))  # Prevent negative values
        
        # Plot X-axis gain with shaded error region
        ax1.plot(common_frequencies, mean_x_gain_db, 'r-', label='X Axis Mean', linewidth=2)
        ax1.fill_between(common_frequencies, x_gain_lower_db, x_gain_upper_db, 
                        color='red', alpha=0.2)
        
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Mean Magnitude Response with Error Bands')
        ax1.grid(True, which="both", ls="-", alpha=0.7)
        ax1.legend()
        
        # Plot X-axis phase with shaded error region
        ax2.plot(common_frequencies, mean_x_phase, 'r-', label='X Axis Mean', linewidth=2)
        ax2.fill_between(common_frequencies, mean_x_phase - std_x_phase, mean_x_phase + std_x_phase, 
                        color='red', alpha=0.2)
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_title('Mean Phase Response with Error Bands')
        ax2.grid(True, which="both", ls="-", alpha=0.7)
        ax2.legend()

        # Generate tick locations at 25Hz intervals
        # Round min/max to nearest interval for nice bounds
        interval = 25  # Fixed 25Hz interval
        start_tick = int(min_freq / interval) * interval
        if start_tick < min_freq:
            start_tick += interval
            
        end_tick = int(max_freq / interval) * interval
        if end_tick > max_freq:
            end_tick -= interval
            
        # Create frequency ticks
        freq_ticks = np.arange(start_tick, end_tick + interval, interval)
        
        # Apply to both axes (they share x-axis)
        ax2.set_xticks(freq_ticks)
        ax2.set_xticklabels([f"{x}" for x in freq_ticks])
        ax2.minorticks_off()  # Turn off minor ticks for cleaner look
        
        # Add information about the number of files analyzed
        fig.text(0.5, 0.01, f"Analysis of {len(dataframes)} frequency sweep files", 
                ha='center', fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # Save the plot - also including the ring_id in the plot filename
        plot_path = os.path.splitext(output_file)[0] + '_plot.png'
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        
        print(f"Statistics plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Note: Could not create statistics plot: {e}")
    
    return output_file

def compare_statistics_files(stat_files=None, output_name="stats_comparison", directory_path=None, interval=25):
    """
    Create a Bode plot comparing multiple statistical CSV files (each containing means and standard deviations).
    
    Args:
        stat_files (list, optional): List of statistical CSV file paths to compare.
                                   If None, asks user to select from available files.
        output_name (str): Base name for output plot files
        directory_path (str, optional): Directory to search for statistics files
                                      If None, uses the Accelerometerplotter_JSON/statistics directory.
        interval (int): Interval for frequency tick marks on the plot
    
    Returns:
        str: Path to the created plot file or None if error
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    import os
    import time
        
    # Default to statistics directory if none specified
    if directory_path is None:
        directory_path = 'Accelerometerplotter_JSON/statistics'
        
    # Ensure directory exists
    os.makedirs(directory_path, exist_ok=True)
    
    # If no files specified, find and list available ones
    if stat_files is None:
        # Look for statistics CSV files (containing "avg_" or "stats_")
        all_stats = glob.glob(os.path.join(directory_path, '**', '*avg_*.csv'), recursive=True)
        all_stats += glob.glob(os.path.join(directory_path, '**', '*stats_*.csv'), recursive=True)
        
        if not all_stats:
            print(f"No statistics CSV files found in {directory_path}")
            return None
        
        # Show available files to choose from
        print("\nAvailable statistics files:")
        for i, file in enumerate(all_stats):
            print(f"[{i+1}] {os.path.basename(file)}")
        
        # Get user selection
        try:
            selection_input = input("\nEnter file numbers to compare (comma-separated, e.g., 1,3,4): ")
            indices = [int(x.strip())-1 for x in selection_input.split(",") if x.strip()]
            stat_files = [all_stats[i] for i in indices if 0 <= i < len(all_stats)]
            
            if not stat_files:
                print("No valid files selected")
                return None
                
        except (ValueError, IndexError) as e:
            print(f"Error selecting files: {e}")
            return None
    
    if len(stat_files) == 0:
        print("No statistics files to compare")
        return None
    
    # Colors for different datasets
    colors = ['r', 'b', 'g', 'c', 'm', 'orange', 'purple', 'brown', 'olive', 'pink']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Statistical Analysis Comparison', fontsize=16)
    
    # Keep track of min/max frequencies for axis scaling
    min_freq = float('inf')
    max_freq = 0
    
    # Process each statistics file
    for i, stat_file in enumerate(stat_files):
        try:
            # Determine color for this dataset
            color = colors[i % len(colors)]
            
            # Extract ring ID from the filename
            # Format is typically: avg_sweep_[ring_id]_[timestamp].csv
            filename = os.path.basename(stat_file)
            
            # Parse for ring ID in the filename
            ring_id = "unknown"
            
            # Example filename: avg_sweep_s1_20231001_1930.csv
            parts = filename.split('_')
            if len(parts) >= 3:
                ring_id = parts[2]
            
            # Create a more descriptive label that includes the ring ID
            label_base = f"Ring {ring_id}"

            # Load the statistics file
            df = pd.read_csv(stat_file)
            
            # Check for required columns
            required_columns = ['Frequency', 'Mean X Gain', 'Std X Gain', 'Mean X Phase (deg)', 'Std X Phase (deg)']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: File {os.path.basename(stat_file)} is missing required columns")
                print(f"Available columns: {df.columns.tolist()}")
                continue
            
            # Update frequency range
            min_freq = min(min_freq, df['Frequency'].min())
            max_freq = max(max_freq, df['Frequency'].max())
            
            # Convert gain to dB for plotting
            mean_gain_db = 20 * np.log10(df['Mean X Gain'])
            
            # Calculate dB bounds for standard deviation
            gain_upper_db = 20 * np.log10(df['Mean X Gain'] + df['Std X Gain'])
            gain_lower_db = 20 * np.log10(np.maximum(df['Mean X Gain'] - df['Std X Gain'], 1e-10))  # Prevent negative values
            
            # Plot magnitude
            ax1.plot(df['Frequency'], mean_gain_db, color=color, linestyle='-', 
                    label=label_base, linewidth=2)
            ax1.fill_between(df['Frequency'], gain_lower_db, gain_upper_db, 
                            color=color, alpha=0.2)
            
            # Plot phase
            ax2.plot(df['Frequency'], df['Mean X Phase (deg)'], color=color, linestyle='-',
                    label=label_base, linewidth=2)
            ax2.fill_between(df['Frequency'], 
                            df['Mean X Phase (deg)'] - df['Std X Phase (deg)'], 
                            df['Mean X Phase (deg)'] + df['Std X Phase (deg)'],
                            color=color, alpha=0.2)
            
            print(f"Processed: {os.path.basename(stat_file)} (Ring ID: {ring_id})")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(stat_file)}: {e}")
    
    # Set up the plot axes
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Mean Magnitude Response with Error Bands')
    ax1.grid(True, which="both", ls="-", alpha=0.7)
    ax1.legend(loc='upper right')
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Mean Phase Response with Error Bands')
    ax2.grid(True, which="both", ls="-", alpha=0.7)
    ax2.legend(loc='upper right')
    
    # Generate tick locations at specified interval
    # Round min/max to nearest interval for nice bounds
    start_tick = int(min_freq / interval) * interval
    if start_tick < min_freq:
        start_tick += interval
        
    end_tick = int(max_freq / interval) * interval
    if end_tick > max_freq:
        end_tick -= interval
        
    # Create frequency ticks
    freq_ticks = np.arange(start_tick, end_tick + interval, interval)
    
    # Apply to both axes (they share x-axis)
    ax2.set_xticks(freq_ticks)
    ax2.set_xticklabels([f"{x}" for x in freq_ticks])
    ax2.minorticks_off()  # Turn off minor ticks for cleaner look
    
    # Add information about the number of files compared
    fig.text(0.5, 0.01, f"Comparison of {len(stat_files)} statistical datasets", 
             ha='center', fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save the plot in a subfolder called "comparison" right under the base folder
    timestamp = time.strftime("%Y%m%d_%H%M")
    
    # Create the comparison output directory under base folder
    base_folder = 'Accelerometerplotter_JSON'
    output_dir = os.path.join(base_folder, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file path
    plot_path = os.path.join(output_dir, f"{output_name}_{timestamp}.png")
    
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"\nComparison plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        return None
    
    return plot_path

def offline_commands():
    # Get user input
    command = input("\nEnter command: ").strip().lower()
    if command.startswith("bode"):
        # Call the Bode plot function
        parts = command.split()
        if len(parts) > 1:
            # User specified a CSV file
            csv_file = ' '.join(parts[1:])
            
            # Normalize path separators to be consistent
            csv_file = csv_file.replace('\\', '/')
            
            # Handle relative paths
            if not os.path.isabs(csv_file):
                # Check if the path already contains the base directory
                if csv_file.lower().startswith('accelerometerplotter_json/'):
                    # Path already has the base directory, so use as is
                    pass
                else:
                    # File is in the filename-only format
                    if csv_file.lower().startswith('ring_'):
                        # Extract ring ID from the filename (format: ring_ID_...)
                        try:
                            # Parse out the ring ID from the filename
                            parts = csv_file.split('_')
                            if len(parts) >= 2:
                                ring_id = parts[1]  # Extract 's1' from 'ring_s1_...'
                                # Construct path with the ring directory
                                csv_file = f'Accelerometerplotter_JSON/ring_{ring_id}/{csv_file}'
                            else:
                                # Fallback if filename format is unexpected
                                csv_file = f'Accelerometerplotter_JSON/{csv_file}'
                        except Exception as e:
                            print(f"Error parsing ring ID: {e}")
                            csv_file = f'Accelerometerplotter_JSON/{csv_file}'
                    else:
                        # No ring_ prefix, just add base directory
                        csv_file = f'Accelerometerplotter_JSON/{csv_file}'
            
            print(f"Creating Bode plot from CSV file: {csv_file}")
                                # Check if file exists before proceeding
            if os.path.exists(csv_file):
                create_bode_plot(csv_file)
            else:
                print(f"Error: File not found: {csv_file}")
        else:
            # Use most recent CSV file
            print("Creating Bode plot from most recent sweep results...")
            create_bode_plot()
    elif command.startswith("combinedbode"):
        parts = command.split()
        if len(parts) > 1:
            # User specified a directory
            directory = ' '.join(parts[1:])
            
            # Normalize path separators to be consistent
            directory = directory.replace('\\', '/')
            
            # Check if the path is absolute or relative
            if not os.path.isabs(directory):
                # If it's a relative path, prepend the base directory
                if not directory.lower().startswith('accelerometerplotter_json/'):
                    directory = f'Accelerometerplotter_JSON/{directory}'
            
            print(f"Creating combined Bode plot from CSV files in directory: {directory}")
            
            # Check if the directory exists before proceeding
            if os.path.isdir(directory):
                # Pass the directory as directory_path parameter, not as the first argument
                create_combined_bode_plot(directory_path=directory)
            else:
                print(f"Error: Directory not found: {directory}")
        else:
            print("Error: Please specify a directory containing CSV files")
            print("Usage: combinedbode <directory>")
    elif command.startswith("stats"):
        # Call the statistics function
        parts = command.split()
        if len(parts) > 1:
            # User specified a directory
            directory = ' '.join(parts[1:])
            
            # Normalize path separators to be consistent
            directory = directory.replace('\\', '/')
            
            # Check if the path is absolute or relative
            if not os.path.isabs(directory):
                # If it's a relative path, prepend the base directory
                if not directory.lower().startswith('accelerometerplotter_json/'):
                    directory = f'Accelerometerplotter_JSON/{directory}'
            
            print(f"Calculating statistics from CSV files in directory: {directory}")
            
            # Check if the directory exists before proceeding
            if os.path.isdir(directory):
                # Pass the directory as directory_path parameter, not as the first argument
                calculate_sweep_statistics(directory_path=directory)
            else:
                print(f"Error: Directory not found: {directory}")
        else:
            print("Error: Please specify a directory containing CSV files")
            print("Usage: stats <directory>")
    elif command == "compare":
        compare_statistics_files()
    else:
        print(f"Unknown command: '{command}'. Type 'help' to see available commands.")

def main():
    if onlineMode:
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
    else:
        print("Running in offline mode. Enter commands manually.")
        while True:
            try:
                offline_commands()
            except KeyboardInterrupt:
                print("\nExiting offline mode.")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
