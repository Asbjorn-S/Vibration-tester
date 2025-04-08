import json
from statistics import mean


def calculate_accelerometer_averages(file_path):
    """
    Calculate average values for each axis of each accelerometer in a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary containing averages for each accelerometer's x, y, and z axes
    """
    try:
        # Load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        results = {}
        
        # Process each accelerometer in the data
        for key, values in data.items():
            # Skip metadata or any non-list entries
            if not isinstance(values, list):
                continue
                
            # Initialize lists to store values for each axis
            x_values = []
            y_values = []
            z_values = []
            
            # Extract values for each axis
            for entry in values:
                x_values.append(entry["x"])
                y_values.append(entry["y"])
                z_values.append(entry["z"])
            
            # Calculate averages
            results[key] = {
                "x_avg": mean(x_values),
                "y_avg": mean(y_values),
                "z_avg": mean(z_values)
            }
        
        return results
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        return None
    except KeyError as e:
        print(f"Error: JSON structure is missing expected key {e}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def main():
    # Hardcode the file path directly here
    file_path = "C:/Users/asbjo/Documents/PlatformIO/Projects/Vibration tester/Accelerometerplotter_JSON/vibration_2025-04-08_12-43-40_unknownHz.json"
    
    averages = calculate_accelerometer_averages(file_path)
    
    if averages:
        print("Average Values:")
        for accelerometer, values in averages.items():
            print(f"\n{accelerometer}:")
            print(f"  X-axis average: {values['x_avg']:.6f}")
            print(f"  Y-axis average: {values['y_avg']:.6f}")
            print(f"  Z-axis average: {values['z_avg']:.6f}")


if __name__ == "__main__":
    main()