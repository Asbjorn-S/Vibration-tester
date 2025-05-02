import paho.mqtt.client as mqtt
import time
import json
import os
import atexit
import fft_analysis
import Accelerometerplotter


def connect_mqtt(broker_address="192.168.68.117", port=1883, client_id="", 
                 username=None, password=None, keepalive=15, retry_interval=2, 
                 max_retries=5):
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
    is_mqtt_v5 = False
    connect_properties = None
    
    try:
        client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5, 
                             callback_api_version=mqtt.CallbackAPIVersion.VERSION2, 
                             clean_session=True)
        print("Using MQTT v5 client")
        # Only create connect_properties for MQTT v5
        connect_properties = mqtt.Properties(mqtt.PacketTypes.CONNECT)
        connect_properties.SessionExpiryInterval = 300  # 5 minutes
        is_mqtt_v5 = True
    except:
        # Fall back to MQTT v3.1.1
        client = mqtt.Client(client_id=client_id, 
                             callback_api_version=mqtt.CallbackAPIVersion.VERSION2, 
                             clean_session=True)
        print("Using MQTT v3.1.1 client")
        connect_properties = None  # No properties for MQTT v3.1.1
    
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
            
            # Only use properties with MQTT v5
            if is_mqtt_v5:
                client.connect(broker_address, port, keepalive, properties=connect_properties)
            else:
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
        print("8. exit - Exit the program")
        print("9. phase [filename] - Calculate phase delay between accelerometers (most recent if no filename specified)")
        print("10. sweep <start_freq> <end_freq> <step_freq> <ring_id> - Run tests at multiple frequencies")
        print("11. bode [csv_file] - Create a Bode plot from sweep results (most recent if no file specified)")
        print("12. combinedbode <directory> - Create a combined Bode plot from selected CSV files in the directory")
        print("13. stats <directory> - Calculate statistics from CSV files in the directory")
        print("14. compare <directory> - Compare statistics from CSV files in the directory")
        print("15. exit - Exit the command interface")
    
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
                    # Use QoS 2 for critical test commands to ensure delivery
                    msg_info = client.publish("vibration/test", str(frequency), qos=2)
                    if not msg_info.is_published():
                        msg_info.wait_for_publish(timeout=5)
                        if msg_info.is_published():
                            print("Test command delivered successfully")
                        else:
                            print("Warning: Test command delivery could not be confirmed")
                except (IndexError, ValueError) as e:
                    print(f"Error: {e}")
                    
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
                    Accelerometerplotter.plot_vibration_data(filename)
                else:
                    # No filename specified, use most recent
                    print("Plotting most recent vibration data...")
                    Accelerometerplotter.plot_vibration_data()
                    
            elif command.startswith("phase"):
                parts = command.split()
                if len(parts) > 1:
                    filename = ' '.join(parts[1:])
                    print(f"Calculating phase delay from file: {filename}")
                    fft_analysis.analyze(filename, doPlot=True)
                else:
                    print("Calculating phase delay from most recent data...")
                    fft_analysis.analyze(doPlot=True)
            
            elif command.startswith("sweep"):
                try:
                    parts = command.split()
                    if len(parts) == 2:
                        start_freq = 350.0
                        end_freq = 175.0
                        step_freq = 25.0
                        ring_id = str(parts[1])
                    elif len(parts) == 5:
                        start_freq = float(parts[1])
                        end_freq = float(parts[2])
                        step_freq = float(parts[3])
                        ring_id = str(parts[4])
                    else:
                        print("Error: Please provide start, end, and step frequencies")
                        print("Usage: sweep [start_freq] [end_freq] [step_freq] <ring_id>")
                        continue
                    
                    
                    print(f"Starting frequency sweep on ring {ring_id}from {start_freq} to {end_freq} Hz with {step_freq} Hz steps")
                        
                    test_results = Accelerometerplotter.test_frequency_range(client, start_freq, end_freq, step_freq, ring_id)
                    
                except (IndexError, ValueError) as e:
                    print(f"Error: {e}")
                    print("Usage: sweep <start_freq> <end_freq> <step_freq> <ring_id>")
            
            elif command.startswith("bode"):
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
                                        csv_file = f'Accelerometerplotter_JSON/rings/ring_{ring_id}/{csv_file}'
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
                        Accelerometerplotter.create_bode_plot(csv_file)
                    else:
                        print(f"Error: File not found: {csv_file}")
                else:
                    # Use most recent CSV file
                    print("Creating Bode plot from most recent sweep results...")
                    Accelerometerplotter.create_bode_plot()
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
                            directory = f'Accelerometerplotter_JSON/rings/ring_{directory}'
                    
                    print(f"Creating combined Bode plot from CSV files in directory: {directory}")
                    
                    # Check if the directory exists before proceeding
                    if os.path.isdir(directory):
                        # Pass the directory as directory_path parameter, not as the first argument
                        Accelerometerplotter.create_combined_bode_plot(directory_path=directory)
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
                            directory = f'Accelerometerplotter_JSON/rings/ring_{directory}'

                    print(f"Calculating statistics from CSV files in directory: {directory}")

                    # Check if the directory exists before proceeding
                    if os.path.isdir(directory):
                        # Pass the directory as directory_path parameter, not as the first argument
                        Accelerometerplotter.calculate_sweep_statistics(directory_path=directory)
                    else:
                        print(f"Error: Directory not found: {directory}")
                else:
                    print("Error: Please specify a directory containing CSV files")
                    print("Usage: stats <directory>")
            elif command.startswith("compare"):
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

                    print(f"Comparing statistics from CSV files in directory: {directory}")

                    # Check if the directory exists before proceeding
                    if os.path.isdir(directory):
                        # Pass the directory as directory_path parameter, not as the first argument
                        Accelerometerplotter.compare_statistics_files(directory_path=directory)
                    else:
                        print(f"Error: Directory not found: {directory}")
                else:
                    print("Error: Please specify a directory containing CSV files")
                    print("Usage: compare <directory>")

            else:
                print(f"Unknown command: '{command}'. Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\nCommand interface interrupted.")
            break

# First, add a global variable to store the metadata
vibration_metadata = {}
received_metadata = False

def on_message(client, userdata, msg):
    """
    Callback function for MQTT message reception.
    Handles different topics related to vibration testing.
    
    Args:
        client: The client instance
        userdata: User data (not used in this implementation)
        msg: The message object containing topic and payload
    """
    global vibration_metadata, received_metadata  # Access the global metadata variable
    
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
                    
                    # Example: Print all points
                    for i, point in enumerate(points):
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
                # print("Received test metadata:")
                # print(f"  Sample count: {metadata.get('sampleCount', 'N/A')}")
                # print(f"  Frequency: {metadata.get('frequency', 'N/A')} Hz")
                # print(f"  Sample time: {metadata.get('sampleTime', 'N/A')} µs")
                
                # Store in global variable for use when processing chunks
                vibration_metadata = metadata
                received_metadata = True  # Set this flag when metadata is received
                
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
                    output_file = Accelerometerplotter.process_vibration_data(data, chunk_num, total_chunks, vibration_metadata)

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


if __name__ == "__main__":
    Accelerometerplotter.main()