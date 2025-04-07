import paho.mqtt.client as mqtt
import time
import atexit
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

def connect_mqtt(broker_address="mqtt.eclipseprojects.io", port=1883, client_id="", 
                 username=None, password=None, keepalive=60, retry_interval=5, 
                 max_retries=12):
    """
    Connect to an MQTT broker and return the client object.
    
    Args:
        broker_address (str): The MQTT broker address
        port (int): The broker port
        client_id (str): The client ID (if empty, a random one will be generated)
        username (str, optional): Username for authentication
        password (str, optional): Password for authentication
        keepalive (int): Keepalive interval in seconds
        retry_interval (int): Seconds to wait between reconnection attempts
        max_retries (int): Maximum number of reconnection attempts, 0 for infinite
        
    Returns:
        mqtt.Client: Connected MQTT client object
    """
    
    reconnect_count = 0
    
    # Callback when the client receives a CONNACK response from the server
    def on_connect(client, userdata, flags, rc, properties=None):
        nonlocal reconnect_count
        if rc == 0:
            print(f"Connected to MQTT broker at {broker_address}:{port}")
            reconnect_count = 0  # Reset reconnect counter on successful connection
        else:
            print(f"Failed to connect to MQTT broker, return code: {rc}")
    
    # Callback when the client disconnects from the server
    def on_disconnect(client, userdata, reasoncode, properties, reasonstring=None):
        nonlocal reconnect_count
        if reasoncode == 0:
            print("Disconnected from MQTT broker successfully")
        else:
            print(f"Unexpected disconnection from MQTT broker, reason: {reasoncode}, {reasonstring}")
            # The automatic reconnection is handled by paho-mqtt if we enable it
    
    # Create a client instance
    client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    # Set username and password if provided
    if username is not None and password is not None:
        client.username_pw_set(username, password)
    
    # Assign the callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    
    # Enable automatic reconnection
    client.reconnect_delay_set(min_delay=1, max_delay=retry_interval)
    
    # Set the client to automatically reconnect
    client.reconnect_on_failure = True
    
    # Function to safely disconnect the client
    def disconnect_mqtt():
        print("Disconnecting MQTT client...")
        client.loop_stop()
        client.disconnect()
        print("MQTT client resources released")
    
    # Register disconnect function to be called on exit
    atexit.register(disconnect_mqtt)
    
    # Connect to the broker
    try:
        client.connect(broker_address, port, keepalive)
        # Start the loop to process network traffic
        client.loop_start()
        # Give some time for the connection to establish
        time.sleep(1)
        return client
    except Exception as e:
        print(f"Initial connection failed: {e}")
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
        print("8. exit - Exit the program")
    
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
                    # User specified a filename
                    filename = ' '.join(parts[1:])
                    # Check if it's a path or just a filename
                    if not os.path.isabs(filename):
                        # If it's just a filename, prepend the data directory
                        if not filename.startswith('Accelerometerplotter_JSON/'):
                            filename = f'Accelerometerplotter_JSON/{filename}'
                    print(f"Plotting data from file: {filename}")
                    plot_vibration_data(filename)
                else:
                    # No filename specified, use most recent
                    print("Plotting most recent vibration data...")
                    plot_vibration_data()
                
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
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    
    print(f"Message received on topic: {topic}")
    
    if topic == "vibration/calibration/status":
        # Handle calibration status updates
        if payload == "complete":
            print("Calibration completed successfully")
        elif payload == "incomplete":
            print("Calibration is incomplete. Please run calibration.")
        else:
            print(f"Unknown calibration status: {payload}")
            
    elif topic == "vibration/calibration/data":
        # Handle calibration data (frequency-amplitude mapping)
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
        # Store metadata for the upcoming vibration test
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
        # Handle vibration test data
        try:
            data = json.loads(payload)
            
            # Check if this is part of a multi-chunk message
            if "chunk" in data and "totalChunks" in data:
                chunk_num = data["chunk"]
                total_chunks = data["totalChunks"]
                print(f"Received data chunk {chunk_num}/{total_chunks}")
                
                # Process accelerometer data
                accel1_data = data.get("accelerometer1", [])
                accel2_data = data.get("accelerometer2", [])
                
                print(f"  Accelerometer 1: {len(accel1_data)} samples")
                print(f"  Accelerometer 2: {len(accel2_data)} samples")
                
                # Process and store the chunked data
                output_file = process_vibration_data(data, chunk_num, total_chunks)
                
                # If we've received all chunks and combined them, analyze the data
                if output_file:
                    print(f"Complete dataset saved to {output_file}, ready for analysis")
                    # Plot the vibration data automatically
                    plot_vibration_data(output_file)
            else:
                print("Received single data message")
                # Handle single message if needed
                with open('Accelerometerplotter_JSON/single_data_message.json', 'w') as f:
                    json.dump(data, f, indent=2)
                
        except json.JSONDecodeError:
            print("Error: Failed to parse vibration data JSON")
    
    # Add any other topics you need to handle

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
    
    print(f"Saved chunk {chunk_num}/{total_chunks} to temporary file")
    
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
        try:
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except (PermissionError, OSError) as e:
            print(f"Note: Could not remove temporary directory: {e}")
            print("This is not critical - processing completed successfully.")
            
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
    
    # If no file specified, find the most recent one
    if (json_file is None):
        data_dir = 'Accelerometerplotter_JSON'
        files = glob.glob(f"{data_dir}/vibration_data_*.json")
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
    
    # Extract x, y, z values
    x1 = np.array([sample["x"] for sample in accel1])
    y1 = np.array([sample["y"] for sample in accel1])
    z1 = np.array([sample["z"] for sample in accel1])
    
    x2 = np.array([sample["x"] for sample in accel2])
    y2 = np.array([sample["y"] for sample in accel2])
    z2 = np.array([sample["z"] for sample in accel2])
    
    # Create figure and subplots - one subplot per axis
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Accelerometer Comparison', fontsize=16)
    
    # Plot X axis data for both accelerometers
    axs[0].plot(timestamps1, x1, 'r-', label='Accelerometer 1', linewidth=1.5)
    axs[0].plot(timestamps2, x2, 'b-', label='Accelerometer 2', linewidth=1.5)
    axs[0].set_title('X-Axis')
    axs[0].set_ylabel('Acceleration')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # Plot Y axis data for both accelerometers
    axs[1].plot(timestamps1, y1, 'r-', label='Accelerometer 1', linewidth=1.5)
    axs[1].plot(timestamps2, y2, 'b-', label='Accelerometer 2', linewidth=1.5)
    axs[1].set_title('Y-Axis')
    axs[1].set_ylabel('Acceleration')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    # Plot Z axis data for both accelerometers
    axs[2].plot(timestamps1, z1, 'r-', label='Accelerometer 1', linewidth=1.5)
    axs[2].plot(timestamps2, z2, 'b-', label='Accelerometer 2', linewidth=1.5)
    axs[2].set_title('Z-Axis')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Acceleration')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()
    
    # Add metadata to the plot
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
    plt.show()
    
    # Return figure for further customization if needed
    return fig


def main():
    # Set up MQTT connection to broker
    client = connect_mqtt(broker_address="192.168.68.127", port=1883, client_id="AccelerometerPlotter")

    if not client:
        print("Failed to establish initial connection to MQTT broker. Exiting.")
        return

    setup_mqtt_callbacks(client)

    try:
        # Main program logic
         # Calibrate test rig
        # Save calibration values to a data structure with a column for motor input and a column for frequency
        # Repeat this for a range of frequencies:
            # Request test from ESP32
            # Parse incoming JSON data from ESP32
            # Save data to a file
            # Create a plot comparing the two accelerometers
        print("Program running. Press Ctrl+C to exit.")        
        # Start listening for user commands
        listen_for_commands(client)
            
    except KeyboardInterrupt:
        print("Program terminated by user")
    


if __name__ == "__main__":
    main()
