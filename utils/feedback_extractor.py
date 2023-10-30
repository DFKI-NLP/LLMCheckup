import json
import datetime
import argparse


def subtract_hours_from_timestamp(timestamp, hours):
    updated_datetime = timestamp - datetime.timedelta(hours=hours)
    return updated_datetime

def extract_json_data_between_timestamps(json_data, start_timestamp, end_timestamp):
    start_datetime = datetime.datetime.strptime(start_timestamp, "%d-%m-%Y, %H:%M:%S")
    end_datetime = datetime.datetime.strptime(end_timestamp, "%d-%m-%Y, %H:%M:%S")
    # Subtract two hours from the start and end timestamps
    print(type(start_datetime))
    start_datetime = subtract_hours_from_timestamp(start_datetime, 2)
    end_datetime = subtract_hours_from_timestamp(end_datetime, 2)
    print(type(start_datetime))

    extracted_data = []

    for entry in json_data:
        entry_datetime = datetime.datetime.strptime(entry['timestamp'], "%d-%m-%Y, %H:%M:%S")
        if start_datetime <= entry_datetime <= end_datetime:
            extracted_data.append(entry)

    return extracted_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_timestamp', type=str, required=True)
    parser.add_argument('--end_timestamp', type=str, required=True)
    parser.add_argument('--username', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.input_file, 'r') as file:
        json_data = json.load(file)
    extracted_data = extract_json_data_between_timestamps(json_data, args.start_timestamp, args.end_timestamp)     
    
    import os
    if not os.path.exists(args.dataset_name):
        os.makedirs(args.dataset_name)
    output_file = args.dataset_name +'/'+args.username +'_'+'feedback.json'
    with open(output_file, 'w') as file:
        json.dump(extracted_data, file, indent=4)

    print(f"Extracted data saved to '{output_file}'.")