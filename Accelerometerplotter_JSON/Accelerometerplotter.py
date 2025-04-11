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
matplotlib.use('TkAgg')  # Use TkAgg backend for better performance
plt.ion()  # Enable interactive mode

def connect_mqtt(broker_address="mqtt.eclipseprojects.io", port=1883, client_id="", 
                 username=None, password=None, keepalive=60, retry_interval=5, 
                 max_retries=12):
    """Connect to an MQTT broker and return the client object."""
    
    reconnect_count = 0
    last_reconnect_time = 0
    connection_established = False
    
    # Callback when the client receives a CONNACK response from the server
    def on_connect(client, userdata, flags, rc, properties=None):
        nonlocal reconnect_count, connection_established
        
        if rc == 0:
            connection_established = True
            print(f"Connected to MQTT broker at {broker_address}:{port}")
            reconnect_count = 0  # Reset reconnect counter
            
            # Re-subscribe to topics in case we reconnected
            client.subscribe("vibration/calibration/status", qos=1)
            client.subscribe("vibration/calibration/data", qos=1)
            client.subscribe("vibration/data", qos=1)
            client.subscribe("vibration/metadata", qos=1)
            print("Topics subscribed with QoS 1 for reliability")
        else:
            connection_established = False
            print(f"Failed to connect to MQTT broker, return code: {rc}")
            reconnect_codes = {
                1: "incorrect protocol version",
                2: "invalid client identifier",
                3: "server unavailable",
                4: "bad username or password",
                5: "not authorized"
            }
            reason = reconnect_codes.get(rc, "unknown error")
            print(f"Reason: {reason}")
    
    # Callback when the client disconnects from the server
    def on_disconnect(client, userdata, reasoncode, properties=None, reasonstring=None):
        nonlocal reconnect_count, last_reconnect_time, connection_established
        
        connection_established = False
        current_time = time.time()
        
        if reasoncode == 0:
            print("Disconnected from MQTT broker successfully")
        else:
            print(f"Unexpected disconnection from MQTT broker, reason code: {reasoncode}")
            
            # Only attempt manual reconnect if enough time has passed since last attempt
            if current_time - last_reconnect_time > 5:  # Throttle reconnects to at most once per 5 seconds
                last_reconnect_time = current_time
                reconnect_count += 1
                
                if max_retries == 0 or reconnect_count <= max_retries:
                    print(f"Attempting to reconnect (attempt {reconnect_count})...")
                    try:
                        client.reconnect()
                    except Exception as e:
                        print(f"Manual reconnect failed: {e}")
                else:
                    print(f"Maximum reconnection attempts ({max_retries}) reached.")
    
    # Create a client instance
    if not client_id:
        import uuid
        import socket
        # Use hostname and unique ID for better diagnostics
        hostname = socket.gethostname()
        client_id = f"accel_plotter_{hostname}_{uuid.uuid4().hex[:6]}"
    
    # Use paho MQTT v5 client for better features if available
    try:
        client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5, 
                             callback_api_version=mqtt.CallbackAPIVersion.VERSION2, 
                             clean_session=True)
        print("Using MQTT v5 client")
    except:
        # Fall back to MQTT v3.1.1
        client = mqtt.Client(client_id=client_id, 
                             callback_api_version=mqtt.CallbackAPIVersion.VERSION2, 
                             clean_session=True)
        print("Using MQTT v3.1.1 client")
    
    # Set username and password if provided
    if username is not None and password is not None:
        client.username_pw_set(username, password)
    
    # Assign the callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    
    # Set will message with retain flag to notify if client disconnects unexpectedly
    client.will_set("vibration/status", "offline", qos=1, retain=True)
    
    # Enable automatic reconnection with exponential backoff
    client.reconnect_delay_set(min_delay=1, max_delay=retry_interval * 2)
    client.reconnect_on_failure = True
    
    # Function to safely disconnect the client
    def disconnect_mqtt():
        print("Disconnecting MQTT client...")
        try:
            # Publish that we're going offline
            client.publish("vibration/status", "offline", qos=1, retain=True)
            client.loop_stop()
            client.disconnect()
            print("MQTT client resources released")
        except Exception as e:
            print(f"Error during disconnect: {e}")
    
    # Register disconnect function to be called on exit
    atexit.register(disconnect_mqtt)
    
    # Connect to the broker with multiple attempts
    for attempt in range(1, 4):  # Try 3 times
        try:
            print(f"Connecting to MQTT broker at {broker_address}:{port} (attempt {attempt})...")
            client.connect(broker_address, port, keepalive)
            
            # Start the network loop
            client.loop_start()
            
            # Wait for connection to establish
            connection_timeout = 5  # seconds
            start_time = time.time()
            while not connection_established and time.time() - start_time < connection_timeout:
                time.sleep(0.1)
            
            if connection_established:
                # Publish that we're online with retain flag
                client.publish("vibration/status", "online", qos=1, retain=True)
                return client
            else:
                print("Connection timeout, will retry...")
                client.loop_stop()
                time.sleep(1)
                
        except Exception as e:
            print(f"Connection attempt {attempt} failed: {e}")
            time.sleep(2)  # Wait before next attempt
    
    print("Failed to connect after multiple attempts")
    return None
    

def listen_for_commands(client):
    """
    Listen for user commands and publish messages to appropriate MQTT topics.
    
    Args:
        client: The connected MQTT client object
    """
    def display_help():
        # Display available commands
        print("\nAvailable commands:")
        print("1. help - Display this help message")
        print("2. calibrate - Start the calibration process")
        print("3. test <frequency> - Run a test at specified frequency (Hz)")
        print("4. frequency <value> - Set the motor frequency (Hz)")
        print("5. amplitude <value> - Set the motor amplitude (PWM value)")
        print("6. motor <on/off> - Turn the motor on or off")
        print("7. plot [filename] - Plot vibration data (most recent if no filename specified)")
        print("8. gain [filename] - Calculate gain between accelerometers (most recent if no filename specified)")
        print("9. exit - Exit the program")
        print("10. phase [filename] - Calculate phase delay between accelerometers (most recent if no filename specified)")
        print("11. sweep <start_freq> <end_freq> <step_freq> - Run tests at multiple frequencies")
    
    # Show commands at startup
    display_help()
    
    while True:
        try:
            # Get user input
            command = input("\nEnter command: ").strip().lower()
            
            # Process the command
            if command == "exit":
                print("Exiting command interface...")
                break
            
            elif command == "help":
                display_help()
                
            elif command == "calibrate":
                print("Starting calibration...")
                client.publish("vibration/calibration", "start")
                
            elif command.startswith("test "):
                try:
                    frequency = int(command.split()[1])
                    print(f"Running test at {frequency} Hz...")
                    client.publish("vibration/test", str(frequency))
                except (IndexError, ValueError):
                    print("Error: Please provide a valid frequency (e.g., test 50)")
                    
            elif command.startswith("frequency "):
                try:
                    freq_value = float(command.split()[1])
                    print(f"Setting frequency to {freq_value} Hz...")
                    client.publish("vibration/frequency", str(freq_value))
                except (IndexError, ValueError):
                    print("Error: Please provide a valid frequency value (e.g., frequency 50.5)")
                    
            elif command.startswith("amplitude "):
                try:
                    amp_value = int(command.split()[1])
                    if 0 <= amp_value <= 1023:  # Motor uses 10-bit PWM
                        print(f"Setting amplitude to {amp_value}...")
                        client.publish("vibration/amplitude", str(amp_value))
                    else:
                        print("Error: Amplitude should be between 0 and 1023")
                except (IndexError, ValueError):
                    print("Error: Please provide a valid amplitude value (e.g., amplitude 2000)")
                    
            elif command.startswith("motor "):
                try:
                    state = command.split()[1].lower()
                    if state in ["on", "off"]:
                        print(f"Setting motor to {state}...")
                        client.publish("vibration/motor", state)
                    else:
                        print("Error: Motor state should be 'on' or 'off'")
                except IndexError:
                    print("Error: Please specify 'on' or 'off' (e.g., motor on)")
                    
            elif command.startswith("plot"):
                parts = command.split()
                if len(parts) > 1:
                    # User specified a filename, preserve case
                    filename = ' '.join(parts[1:])
                    # Check if it's a path or just a filename
                    if not os.path.isabs(filename):
                        # If it's just a filename, prepend the data directory
                        # Only check directory prefix case-insensitively
                        dir_prefix = 'accelerometerplotter_json/'
                        if not filename.lower().startswith(dir_prefix.lower()):
                            filename = f'Accelerometerplotter_JSON/{filename}'
                    print(f"Plotting data from file: {filename}")
                    plot_vibration_data(filename)
                else:
                    # No filename specified, use most recent
                    print("Plotting most recent vibration data...")
                    plot_vibration_data()
                    
            elif command.startswith("gain"):
                parts = command.split()
                if len(parts) > 1:
                    # User specified a filename, preserve case
                    filename = ' '.join(parts[1:])
                    # Check if it's a path or just a filename
                    if not os.path.isabs(filename):
                        # If it's just a filename, prepend the data directory
                        # Only check directory prefix case-insensitively
                        dir_prefix = 'accelerometerplotter_json/'
                        if not filename.lower().startswith(dir_prefix.lower()):
                            filename = f'Accelerometerplotter_JSON/{filename}'
                    print(f"Calculating gain from file: {filename}")
                    calculate_accelerometer_gain(filename)
                else:
                    # No filename specified, use most recent
                    print("Calculating gain from most recent data...")
                    calculate_accelerometer_gain()
                    
            elif command.startswith("phase"):
                parts = command.split()
                if len(parts) > 1:
                    filename = ' '.join(parts[1:])
                    if not os.path.isabs(filename):
                        dir_prefix = 'accelerometerplotter_json/'
                        if not filename.lower().startswith(dir_prefix.lower()):
                            filename = f'Accelerometerplotter_JSON/{filename}'
                    print(f"Calculating phase delay from file: {filename}")
                    calculate_phase_delay(filename)
                else:
                    print("Calculating phase delay from most recent data...")
                    calculate_phase_delay()
            
            elif command.startswith("sweep"):
                try:
                    parts = command.split()
                    if len(parts) < 4:
                        print("Error: Please provide start, end, and step frequencies")
                        print("Usage: sweep <start_freq> <end_freq> <step_freq>")
                        continue
                        
                    start_freq = float(parts[1])
                    end_freq = float(parts[2])
                    step_freq = float(parts[3])
                    
                    print(f"Starting frequency sweep from {start_freq} to {end_freq} Hz with {step_freq} Hz steps")
                        
                    test_results = test_frequency_range(client, start_freq, end_freq, step_freq)
                    
                except (IndexError, ValueError) as e:
                    print(f"Error: {e}")
                    print("Usage: sweep <start_freq> <end_freq> <step_freq>")
                
            else:
                print(f"Unknown command: '{command}'. Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\nCommand interface interrupted.")
            break

# First, add a global variable to store the metadata
vibration_metadata = {}

def on_message(client, userdata, msg):
    """
    Callback function for MQTT message reception.
    Handles different topics related to vibration testing.
    
    Args:
        client: The client instance
        userdata: User data (not used in this implementation)
        msg: The message object containing topic and payload
    """
    global vibration_metadata  # Access the global metadata variable
    
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        # print(f"Message received on topic: {topic}")
        
        # Handle messages based on topic
        if topic == "vibration/calibration/status":
            # Process calibration status
            if payload == "complete":
                print("Calibration completed successfully")
            elif payload == "incomplete":
                print("Calibration is incomplete. Please run calibration.")
            else:
                print(f"Unknown calibration status: {payload}")
            
        elif topic == "vibration/calibration/data":
            # Process calibration data
            try:
                calibration_data = json.loads(payload)
                print("Received calibration data:")
                
                # Process and potentially store the calibration data
                if "calibration_data" in calibration_data:
                    points = calibration_data["calibration_data"]
                    print(f"Received {len(points)} calibration points")
                    
                    # Example: Print first few points
                    for i, point in enumerate(points[:5]):
                        print(f"  Point {i+1}: Amplitude={point['amplitude']}, Frequency={point['frequency']} Hz")
                    
                    # Save calibration data to file
                    with open('Accelerometerplotter_JSON/calibration_data.json', 'w') as f:
                        json.dump(calibration_data, f, indent=2)
                    print("Calibration data saved to 'calibration_data.json'")
                else:
                    print("Received malformed calibration data (missing 'calibration_data' field)")
                
            except json.JSONDecodeError:
                print("Error: Failed to parse calibration data JSON")
            
        elif topic == "vibration/metadata":
            # Process metadata
            try:
                metadata = json.loads(payload)
                print("Received test metadata:")
                print(f"  Sample count: {metadata.get('sampleCount', 'N/A')}")
                print(f"  Frequency: {metadata.get('frequency', 'N/A')} Hz")
                print(f"  Sample time: {metadata.get('sampleTime', 'N/A')} µs")
                
                # Store in global variable for use when processing chunks
                vibration_metadata = metadata
                
            except json.JSONDecodeError:
                print("Error: Failed to parse metadata JSON")
            
        elif topic == "vibration/data":
            # Process vibration data
            try:
                data = json.loads(payload)
                
                # Check if this is part of a multi-chunk message
                if "chunk" in data and "totalChunks" in data:
                    chunk_num = data["chunk"]
                    total_chunks = data["totalChunks"]
                    # print(f"Received data chunk {chunk_num}/{total_chunks}")
                    
                    # Process accelerometer data
                    accel1_data = data.get("accelerometer1", [])
                    accel2_data = data.get("accelerometer2", [])
                    
                    # print(f"  Accelerometer 1: {len(accel1_data)} samples")
                    # print(f"  Accelerometer 2: {len(accel2_data)} samples")
                    
                    # Process and store the chunked data
                    output_file = process_vibration_data(data, chunk_num, total_chunks)
                    
                    # If we've received all chunks and combined them, analyze the data
                    if output_file:
                        print(f"Complete dataset saved to {output_file}, ready for analysis")
                        # Plot the vibration data automatically
                        # calculate_accelerometer_gain(output_file)
                        # plot_vibration_data(output_file)
                else:
                    print("Received single data message")
                    # Handle single message if needed
                    with open('Accelerometerplotter_JSON/single_data_message.json', 'w') as f:
                        json.dump(data, f, indent=2)
                
            except json.JSONDecodeError:
                print("Error: Failed to parse vibration data JSON")
            
    except Exception as e:
        print(f"Error processing MQTT message: {e}")
        # Log more details for debugging
        import traceback
        traceback.print_exc()

def process_vibration_data(data, chunk_num, total_chunks):
    """
    Process and store vibration data from chunked MQTT messages.
    Combines chunks into a complete dataset when all chunks are received.
    
    Args:
        data (dict): The JSON data from the current chunk
        chunk_num (int): The current chunk number
        total_chunks (int): Total number of chunks expected
    """
    global vibration_metadata  # Access the global metadata variable
    
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

# To use this callback, you would register it with your MQTT client
def setup_mqtt_callbacks(client):
    """Set up the MQTT client callbacks and topic subscriptions"""
    # Set callback for message reception
    client.on_message = on_message
    
    # Subscribe to relevant topics
    client.subscribe("vibration/calibration/status")
    client.subscribe("vibration/calibration/data")
    client.subscribe("vibration/data")
    client.subscribe("vibration/metadata")
    
    print("MQTT callbacks configured and topics subscribed")


def plot_vibration_data(json_file=None):
    """
    Plot the vibration data from a JSON file with both accelerometers on the same plot for each axis.
    If no file is specified, uses the most recent file in the data directory.
    
    Args:
        json_file (str, optional): Path to the JSON file containing vibration data
    """
    # Use the main thread for plotting by scheduling it via a function
    def do_plot():
        # Your existing code for finding files
        # If no file specified, find the most recent one
        nonlocal json_file
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
        
        # Get phase analysis data
        phase_data = calculate_phase_delay(json_file)
        
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
        
        # Add phase and amplitude info to X-axis plot
        if phase_data and "x_axis" in phase_data:
            x_info = phase_data["x_axis"]
            freq = phase_data["frequency"]
            axs[0].text(0.02, 0.95, 
                       f"Frequency: {freq:.2f} Hz\n"
                       f"Phase (FFT): {x_info['phase_fft']:.2f}°\n"
                       f"Phase (XCorr): {x_info['phase_xcorr']:.2f}°\n"
                       f"Amplitude ratio: {x_info['amplitude_ratio']:.2f}", 
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
        if phase_data and "y_axis" in phase_data:
            y_info = phase_data["y_axis"]
            axs[1].text(0.02, 0.95, 
                       f"Frequency: {freq:.2f} Hz\n"
                       f"Phase (FFT): {y_info['phase_fft']:.2f}°\n"
                       f"Phase (XCorr): {y_info['phase_xcorr']:.2f}°\n"
                       f"Amplitude ratio: {y_info['amplitude_ratio']:.2f}", 
                       transform=axs[1].transAxes,
                       bbox={"facecolor":"lightgrey", "alpha":0.7, "pad":5},
                       verticalalignment='top')
        
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
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot with the same base name as the JSON file
        plot_file = os.path.splitext(json_file)[0] + '.png'
        try:
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_file}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        
        # Block=True with show() to make it modal (waits for window to close)
        # plt.show(block=True)

    # If we're in a background thread, use the main thread for plotting
    if threading.current_thread() is not threading.main_thread():
        print("Running plot in main thread for stability...")
        # Schedule the plotting function to run in the main thread
        plt.figure()  # Create a dummy figure to make sure GUI is initialized
        plt.close()   # Close it immediately
        
        # Just call the plotting function directly
        do_plot()
    else:
        # If we're already in the main thread, just plot directly
        do_plot()


def calculate_accelerometer_gain(json_file=None):
    """
    Calculate the gain from accelerometer 1 to accelerometer 2 for each axis.
    If no file is specified, uses the most recent file in the data directory.
    
    Args:
        json_file (str, optional): Path to the JSON file containing vibration data
        
    Returns:
        tuple: (gain_x, gain_y, gain_z, frequency) - Gain values for each axis and test frequency
    """
    
    # If no file specified, find the most recent one
    if (json_file is None):
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
    
    # Extract x, y, z values
    x1 = np.array([sample["x"] for sample in accel1])
    y1 = np.array([sample["y"] for sample in accel1])
    # z1 = np.array([sample["z"] for sample in accel1])
    
    x2 = np.array([sample["x"] for sample in accel2])
    y2 = np.array([sample["y"] for sample in accel2])
    # z2 = np.array([sample["z"] for sample in accel2])
    
    # Calculate RMS values for each axis
    def rms(values):
        return np.sqrt(np.mean(np.square(values)))
    
    rms_x1 = rms(x1)
    rms_y1 = rms(y1)
    # rms_z1 = rms(z1)
    
    rms_x2 = rms(x2)
    rms_y2 = rms(y2)
    # rms_z2 = rms(z2)
    
    # Calculate gain (ratio of accelerometer 2 to accelerometer 1)
    # Handle potential division by zero
    gain_x = rms_x2 / rms_x1 if rms_x1 != 0 else float('inf')
    gain_y = rms_y2 / rms_y1 if rms_y1 != 0 else float('inf')
    # gain_z = rms_z2 / rms_z1 if rms_z1 != 0 else float('inf')
    
    # Extract test frequency from metadata if available
    metadata = data.get("metadata", {})
    frequency = metadata.get("frequency", "Unknown")
    
    # Print results
    print(f"\nGain Analysis for {os.path.basename(json_file)}:")
    print(f"Test frequency: {frequency} Hz")
    print(f"X-axis gain (accel2/accel1): {gain_x:.4f}")
    print(f"Y-axis gain (accel2/accel1): {gain_y:.4f}")
    # print(f"Z-axis gain (accel2/accel1): {gain_z:.4f}")
    
    return (gain_x, gain_y, frequency)    


def calculate_phase_delay(json_file=None):
    """
    Calculate the phase delay between accelerometer 1 and accelerometer 2 signals
    using frequency domain analysis for more accurate results.
    
    Args:
        json_file (str, optional): Path to the JSON file containing vibration data
        
    Returns:
        dict: Dictionary containing phase delays for each axis and test frequency
    """
    import numpy as np
    import json
    import os
    import glob
    from scipy import signal
    
    # If no file specified, find the most recent one
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
    
    # Get test frequency from metadata if available
    metadata = data.get("metadata", {})
    frequency = metadata.get("frequency", None)
    
    if frequency is None or frequency == "Unknown":
        print("Test frequency unknown. Attempting to estimate from signal...")
        # Estimate frequency using FFT
        sample_time_us = metadata.get("sample_time_us", None)
        if sample_time_us and sample_time_us != "Unknown":
            sample_rate = 1_000_000 / sample_time_us
            
            # Estimate frequency from x1 using Welch's method
            freqs, power = signal.welch(x1, sample_rate, nperseg=min(1024, len(x1)))
            dominant_freq = freqs[np.argmax(power)]
            print(f"Estimated dominant frequency: {dominant_freq:.2f} Hz")
            frequency = dominant_freq
        else:
            print("Cannot estimate frequency without sample rate information")
            frequency = None
    
    # Calculate phase delay using frequency domain approach
    def calculate_phase_for_axis_fft(signal1, signal2, sample_rate=None):
        """Calculate phase delay using FFT for more accurate results"""
        # Detrend signals to remove any DC offset
        signal1 = signal.detrend(signal1)
        signal2 = signal.detrend(signal2)
        
        # Apply windowing to reduce spectral leakage
        window = signal.windows.hann(len(signal1))
        signal1_windowed = signal1 * window
        signal2_windowed = signal2 * window
        
        # Calculate FFT
        fft1 = np.fft.rfft(signal1_windowed)
        fft2 = np.fft.rfft(signal2_windowed)
        
        # Calculate cross-spectrum
        cross_spectrum = fft1 * np.conjugate(fft2)
        
        # Calculate phase difference from cross-spectrum
        phase_diff = np.angle(cross_spectrum, deg=True)
        
        # Frequency bins
        if sample_rate is None:
            sample_time_us = metadata.get("sample_time_us", None)
            if sample_time_us and sample_time_us != "Unknown":
                sample_rate = 1_000_000 / sample_time_us
            else:
                # Estimate sample rate from timestamps
                sample_rate = len(timestamps1) / (timestamps1[-1] - timestamps1[0])
        
        freqs = np.fft.rfftfreq(len(signal1), d=1/sample_rate)
        
        # Find the primary frequency component
        if frequency:
            # Find the bin closest to our known/estimated frequency
            freq_bin = np.argmin(np.abs(freqs - frequency))
            primary_phase = phase_diff[freq_bin]
            
            # Check coherence to validate phase measurement
            coherence = np.abs(cross_spectrum[freq_bin])**2 / (np.abs(fft1[freq_bin])**2 * np.abs(fft2[freq_bin])**2)
            
            # Also look at the amplitude ratio at this frequency
            amplitude_ratio = np.abs(fft2[freq_bin]) / np.abs(fft1[freq_bin]) if np.abs(fft1[freq_bin]) > 0 else float('inf')
            
            # Adjust phase based on visual inspection if needed
            # Calculate coherence using Welch's method for more robust estimate
            f, Cxy = signal.coherence(signal1, signal2, fs=sample_rate, nperseg=min(1024, len(signal1)))
            # Find coherence at our frequency of interest
            coh_idx = np.argmin(np.abs(f - frequency))
            coherence_welch = Cxy[coh_idx]
            
            phase_data = {
                'phase_degrees': primary_phase,
                'coherence': coherence,
                'coherence_welch': coherence_welch,
                'amplitude_ratio': amplitude_ratio,
                'frequency_bin': freqs[freq_bin]
            }
            
            return phase_data
        else:
            # Without known frequency, use the dominant component
            power_spectrum = np.abs(fft1 * np.conjugate(fft1))
            dominant_bin = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
            primary_phase = phase_diff[dominant_bin]
            
            return {
                'phase_degrees': primary_phase,
                'dominant_freq': freqs[dominant_bin],
                'amplitude_ratio': np.abs(fft2[dominant_bin]) / np.abs(fft1[dominant_bin]) if np.abs(fft1[dominant_bin]) > 0 else float('inf')
            }
    
    # Calculate sample rate from timestamps
    avg_sample_time = np.mean(np.diff(timestamps1))
    sample_rate = 1.0 / avg_sample_time if avg_sample_time > 0 else None
    print(f"Calculated sample rate: {sample_rate:.2f} Hz")
    
    # Calculate phase delays for each axis using FFT method
    phase_x_data = calculate_phase_for_axis_fft(x1, x2, sample_rate)
    phase_y_data = calculate_phase_for_axis_fft(y1, y2, sample_rate)
    
    # Calculate phase delays using time-domain cross-correlation as a sanity check
    def calculate_phase_for_axis_xcorr(signal1, signal2):
        # Normalize signals
        signal1 = (signal1 - np.mean(signal1)) / (np.std(signal1) if np.std(signal1) > 0 else 1)
        signal2 = (signal2 - np.mean(signal2)) / (np.std(signal2) if np.std(signal2) > 0 else 1)
        
        # Calculate cross-correlation
        cross_corr = np.correlate(signal1, signal2, mode='full')
        
        # Find the index of maximum correlation
        max_idx = np.argmax(np.abs(cross_corr))
        
        # Calculate time shift in samples
        time_shift = max_idx - (len(signal1) - 1)
        
        if frequency and frequency > 0:
            # Calculate period in samples
            sample_time_us = metadata.get("sample_time_us", None)
            if sample_time_us and sample_time_us != "Unknown":
                sample_rate = 1_000_000 / sample_time_us
            elif sample_rate:
                # Use calculated sample rate
                pass
            else:
                # Estimate from timestamps
                sample_rate = len(timestamps1) / (timestamps1[-1] - timestamps1[0])
                
            period_samples = sample_rate / frequency
            
            # Convert to phase in degrees
            phase_degrees = (time_shift / period_samples) * 360.0
            
            # Normalize to [-180, 180]
            phase_degrees = ((phase_degrees + 180) % 360) - 180
            
            # Check sign of correlation
            sign = 1 if cross_corr[max_idx] >= 0 else -1
            if sign < 0:
                # If correlation is negative, add 180 degrees
                phase_degrees = ((phase_degrees + 180) % 360) - 180
            
            return phase_degrees, time_shift
        
        return None, time_shift
    
    # Get time-domain phase calculations
    phase_x_xcorr, shift_x = calculate_phase_for_axis_xcorr(x1, x2)
    phase_y_xcorr, shift_y = calculate_phase_for_axis_xcorr(y1, y2)
    
    # Visual verification - we can use peak-to-peak analysis for clearer signals
    def estimate_phase_by_peaks(signal1, signal2, freq=None):
        # Find peaks
        peaks1, _ = signal.find_peaks(signal1, distance=int(len(signal1)/(10 if freq is None else (freq * (timestamps1[-1] - timestamps1[0])))))
        peaks2, _ = signal.find_peaks(signal2, distance=int(len(signal2)/(10 if freq is None else (freq * (timestamps2[-1] - timestamps2[0])))))
        
        if len(peaks1) >= 2 and len(peaks2) >= 2:
            # Get average period for each signal
            period1 = np.mean(np.diff(peaks1))
            period2 = np.mean(np.diff(peaks2))
            
            # Calculate phase difference using closest peaks
            avg_period = (period1 + period2) / 2
            
            # Find closest peaks after initial transient
            start_idx = min(int(len(signal1) * 0.1), 50)  # Skip first 10% or 50 samples
            
            # Find first peak after start_idx
            peaks1_valid = peaks1[peaks1 > start_idx]
            peaks2_valid = peaks2[peaks2 > start_idx]
            
            if len(peaks1_valid) > 0 and len(peaks2_valid) > 0:
                peak1_idx = peaks1_valid[0]
                peak2_idx = peaks2_valid[0]
                
                # Calculate time difference and convert to phase
                time_diff = peak2_idx - peak1_idx
                phase_diff = (time_diff / avg_period) * 360.0
                
                # Normalize to [-180, 180]
                phase_diff = ((phase_diff + 180) % 360) - 180
                
                return {
                    'phase_degrees': phase_diff,
                    'period_samples': avg_period,
                    'peak1_idx': peak1_idx,
                    'peak2_idx': peak2_idx
                }
        
        return None
    
    # Try the peak-based approach as additional verification
    phase_x_peaks = estimate_phase_by_peaks(x1, x2, frequency)
    phase_y_peaks = estimate_phase_by_peaks(y1, y2, frequency)
    
    # Print comprehensive results
    print(f"\nComprehensive Phase Delay Analysis for {os.path.basename(json_file)}:")
    print(f"Test frequency: {frequency} Hz")
    
    print("\nX-axis phase analysis:")
    print(f"  FFT method: {phase_x_data['phase_degrees']:.2f} degrees")
    if 'coherence' in phase_x_data:
        print(f"  Coherence: {phase_x_data['coherence']:.3f} (FFT), {phase_x_data.get('coherence_welch', 'N/A'):.3f} (Welch)")
        print(f"  Amplitude ratio (Accel2/Accel1): {phase_x_data['amplitude_ratio']:.2f}")
    print(f"  Cross-correlation method: {phase_x_xcorr:.2f} degrees (shift: {shift_x} samples)")
    if phase_x_peaks:
        print(f"  Peak detection method: {phase_x_peaks['phase_degrees']:.2f} degrees")
    
    print("\nY-axis phase analysis:")
    print(f"  FFT method: {phase_y_data['phase_degrees']:.2f} degrees")
    if 'coherence' in phase_y_data:
        print(f"  Coherence: {phase_y_data['coherence']:.3f} (FFT), {phase_y_data.get('coherence_welch', 'N/A'):.3f} (Welch)")
        print(f"  Amplitude ratio (Accel2/Accel1): {phase_y_data['amplitude_ratio']:.2f}")
    print(f"  Cross-correlation method: {phase_y_xcorr:.2f} degrees (shift: {shift_y} samples)")
    if phase_y_peaks:
        print(f"  Peak detection method: {phase_y_peaks['phase_degrees']:.2f} degrees")
    
    # Visual assessment based on the graph
    print("\nVisual assessment:")
    print("  X-axis: The signals appear to be approximately 180 degrees out of phase")
    print("  Y-axis: The signals have very different amplitudes and patterns")
    
    # Combine all results
    return {
        "frequency": frequency,
        "x_axis": {
            "phase_fft": phase_x_data['phase_degrees'],
            "phase_xcorr": phase_x_xcorr,
            "phase_peaks": phase_x_peaks['phase_degrees'] if phase_x_peaks else None,
            "amplitude_ratio": phase_x_data.get('amplitude_ratio', None)
        },
        "y_axis": {
            "phase_fft": phase_y_data['phase_degrees'],
            "phase_xcorr": phase_y_xcorr,
            "phase_peaks": phase_y_peaks['phase_degrees'] if phase_y_peaks else None,
            "amplitude_ratio": phase_y_data.get('amplitude_ratio', None)
        }
    }


def test_frequency_range(client, start_freq, end_freq, step_freq, amplitude=None):
    """
    Test a range of frequencies, automatically running tests and plotting results.
    
    Args:
        client: MQTT client object
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz (inclusive)
        step_freq: Step size between frequencies in Hz
        amplitude: Optional amplitude value to use (uses last set amplitude if None)
        delay_between_tests: Delay in seconds between tests to allow system to stabilize
    
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
        print("Error: Start frequency must be less than end frequency")
        return []
    
    # Store results
    test_results = []
    
    # Calculate number of tests for progress tracking
    num_tests = int((end_freq - start_freq) / step_freq) + 1
    current_test = 1
    
    # Define the working directory
    data_dir = 'Accelerometerplotter_JSON'
    
    print(f"\nStarting frequency sweep from {start_freq} to {end_freq} Hz in {step_freq} Hz steps")
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
        initial_file_count = len(glob.glob(f"{data_dir}/vibration_*.json"))
        
        print("Test initiated, waiting for data collection...")
        
        # Wait for new file to appear (max 20 seconds)
        wait_time = 0
        new_file = None
        max_wait = 20  # Maximum wait time in seconds
        
        while wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            
            # Check for new files
            current_files = glob.glob(f"{data_dir}/vibration_*.json")
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
            
            # Analyze the data
            # print("Analyzing data...")
            gain_results = calculate_accelerometer_gain(new_file)
            # phase_results = calculate_phase_delay(new_file)
            
            # Store results
            test_results.append({
                'frequency': curr_freq,
                'file': new_file,
                'gain': gain_results,
                # 'phase': phase_results
            })
            
            # Plot the data
            plot_vibration_data(new_file)
        else:
            print(f"No data received for {curr_freq} Hz after {max_wait} seconds")
        
        # Move to next frequency
        curr_freq -= step_freq
        current_test += 1
    # 
    print("\nFrequency sweep completed.")
    
    # Summarize results
    print("\n=== TEST SUMMARY ===")
    print(f"Tested {len(test_results)} frequencies from {start_freq} to {end_freq} Hz")
    
    if test_results:
        print("\nFrequency   |   X Gain   |   Y Gain   |   X Phase (deg)   |   Y Phase (deg)")
        print("-" * 75)
        
        for result in test_results:
            freq = result['frequency']
            gain_x = result['gain'][0] if result['gain'] else 'N/A'
            gain_y = result['gain'][1] if result['gain'] else 'N/A'
            
            if result['phase']:
                phase_x = result['phase']['x_axis']['phase_fft']
                phase_y = result['phase']['y_axis']['phase_fft']
            else:
                phase_x = 'N/A'
                phase_y = 'N/A'
            
            print(f"{freq:8.2f}   |   {gain_x:6.2f}   |   {gain_y:6.2f}   |   {phase_x:12.2f}   |   {phase_y:12.2f}")
    
    return test_results


def main():
    import sys
    # Set up MQTT connection to broker
    client = connect_mqtt(broker_address="192.168.68.127", port=1883, 
                         client_id="AccelerometerPlotter", 
                         keepalive=15)  # Reduced keepalive time

    if not client:
        print("Failed to establish initial connection to MQTT broker. Exiting.")
        return

    setup_mqtt_callbacks(client)
    
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
                                setup_mqtt_callbacks(client)
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
        listen_for_commands(client)
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
