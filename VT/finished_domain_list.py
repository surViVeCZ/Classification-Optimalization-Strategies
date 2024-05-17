import argparse
import csv
from pathlib import Path

def filter_domains(mode, input_path, output_folder):
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"The file at {input_path} was not found.")
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    domains = []
    
    with open(input_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if (mode == 'malign' and row['Verdict'].strip().lower() == 'malign') or \
               (mode == 'benign' and row['Verdict'].strip().lower() != 'malign'):
                domains.append(row['Domain'])
                
    base_name = input_path.split('_')[-1].split('.')[0]
    output_file_name = f"{base_name}_finished_list.txt"
    output_file_path = Path(output_folder) / output_file_name
    
    with open(output_file_path, 'w') as file:
        for domain in domains:
            file.write(domain + '\n')

    return output_file_path

# Setup the argument parser
parser = argparse.ArgumentParser(description='Filter domain names based on verdict and save them to a file.')
parser.add_argument('--mode', type=str, choices=['malign', 'benign'], help='Mode to filter the domains: malign/benign', required=True)
parser.add_argument('--input', type=str, help='Path to the input CSV file', required=True)
parser.add_argument('--output', type=str, help='Path to the output folder', required=True)

args = parser.parse_args()

# Replace the placeholder paths and uncomment the following line to execute the function with real paths
filtered_domains_path = filter_domains(args.mode, args.input, args.output)
print(f"Filtered domains saved to {filtered_domains_path}")
