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
                        plot_vibration_data(output_file)
                        calculate_accelerometer_gain(output_file)
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
        
        # Rest of your plotting code...
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
        
        # Extract x, y, z values
        x1 = np.array([sample["x"] for sample in accel1])
        y1 = np.array([sample["y"] for sample in accel1])
        # z1 = np.array([sample["z"] for sample in accel1])
        
        x2 = np.array([sample["x"] for sample in accel2])
        y2 = np.array([sample["y"] for sample in accel2])
        # z2 = np.array([sample["z"] for sample in accel2])
        
        # Clear any existing plots
        plt.close('all')
        
        # Create figure and subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'Accelerometer Comparison - {os.path.basename(json_file)}', fontsize=16)
        
        # Rest of your plotting code...
        # Plot data for each axis
        axs[0].plot(timestamps1, x1, 'r-', label='Accelerometer 1', linewidth=1.5)
        axs[0].plot(timestamps2, x2, 'b-', label='Accelerometer 2', linewidth=1.5)
        axs[0].set_title('X-Axis')
        axs[0].set_ylabel('Acceleration')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Similar code for Y and Z axes...
        axs[1].plot(timestamps1, y1, 'r-', label='Accelerometer 1', linewidth=1.5)
        axs[1].plot(timestamps2, y2, 'b-', label='Accelerometer 2', linewidth=1.5)
        axs[1].set_title('Y-Axis')
        axs[1].set_ylabel('Acceleration')
        axs[1].set_xlabel('Time (seconds)')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        
        # axs[2].plot(timestamps1, z1, 'r-', label='Accelerometer 1', linewidth=1.5)
        # axs[2].plot(timestamps2, z2, 'b-', label='Accelerometer 2', linewidth=1.5)
        # axs[2].set_title('Z-Axis')
        # axs[2].set_xlabel('Time (seconds)')
        # axs[2].set_ylabel('Acceleration')
        # axs[2].grid(True, alpha=0.3)
        # axs[2].legend()
        
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
        
        # Block=True with show() to make it modal (waits for window to close)
        plt.show(block=True)

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
    Calculate the phase delay between accelerometer 1 and accelerometer 2 signals.
    If no file is specified, uses the most recent file in the data directory.
    
    Args:
        json_file (str, optional): Path to the JSON file containing vibration data
        
    Returns:
        dict: Dictionary containing phase delays for each axis in degrees and time
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
    
    # Extract metadata
    metadata = data.get("metadata", {})
    frequency = metadata.get("frequency", 0)
    sample_time_us = metadata.get("sample_time_us", 0)
    
    if sample_time_us == 0:
        print("Warning: Sample time is zero or missing in metadata")
        sample_time_s = 1.0  # Default to 1 second if metadata is invalid
    else:
        sample_time_s = sample_time_us / 1_000_000  # Convert to seconds
    
    # Extract x, y values 
    x1 = np.array([sample["x"] for sample in accel1])
    y1 = np.array([sample["y"] for sample in accel1])
    
    x2 = np.array([sample["x"] for sample in accel2])
    y2 = np.array([sample["y"] for sample in accel2])
    
    # Calculate phase delay using cross-correlation
    def calculate_delay(signal1, signal2):
        # Normalize signals to improve correlation accuracy
        sig1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) if np.std(signal1) != 0 else 1)
        sig2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) if np.std(signal2) != 0 else 1)
        
        # Calculate cross-correlation
        correlation = np.correlate(sig1_norm, sig2_norm, mode='full')
        
        # Find the index of maximum correlation
        max_corr_idx = np.argmax(correlation)
        
        # Calculate the delay in samples
        delay_samples = max_corr_idx - (len(signal1) - 1)
        
        # Convert to time delay
        time_delay = delay_samples * sample_time_s
        
        # Calculate phase delay in degrees if frequency is available
        if frequency > 0:
            # One full cycle at the given frequency
            cycle_time = 1.0 / frequency
            # Convert time delay to fraction of cycle
            phase_fraction = (time_delay / cycle_time) % 1.0
            # Convert to degrees (0-360)
            phase_degrees = phase_fraction * 360
        else:
            phase_degrees = None
            
        return {
            "delay_samples": delay_samples,
            "time_delay": time_delay,
            "phase_degrees": phase_degrees
        }
    
    # Calculate delays for each axis
    x_delay = calculate_delay(x1, x2)
    y_delay = calculate_delay(y1, y2)
    
    # Print results
    print(f"\nPhase Delay Analysis for {os.path.basename(json_file)}:")
    print(f"Test frequency: {frequency} Hz")
    
    print(f"\nX-axis:")
    print(f"  Time delay: {x_delay['time_delay']*1000:.2f} ms")
    if x_delay['phase_degrees'] is not None:
        print(f"  Phase delay: {x_delay['phase_degrees']:.2f}°")
    
    print(f"\nY-axis:")
    print(f"  Time delay: {y_delay['time_delay']*1000:.2f} ms")
    if y_delay['phase_degrees'] is not None:
        print(f"  Phase delay: {y_delay['phase_degrees']:.2f}°")
    
    # Return a dictionary with all results
    results = {
        "frequency": frequency,
        "x_axis": x_delay,
        "y_axis": y_delay,
    }
    
    return results


def main():
    import sys
    # Set up MQTT connection to broker
    client = connect_mqtt(broker_address="192.168.68.126", port=1883, 
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
