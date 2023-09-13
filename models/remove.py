import csv

# Define the input and output file names
input_file = 'C:\\Users\\quchenfu\\Downloads\\ml-stuttering-events-dataset\\SEP-28k_labels_old.csv'
output_file = 'C:\\Users\\quchenfu\\Downloads\\ml-stuttering-events-dataset\\SEP-28k_labels.csv'

# List of strings to remove from the lines
strings_to_remove = ['StutteringIsCool', 'StrongVoices']

# Open the input and output CSV files
with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Iterate through each row in the input CSV
    for row in reader:
        # Check if any of the strings to remove are in the row
        if not any(s in ' '.join(row) for s in strings_to_remove):
            # If none of the strings are found, write the row to the output CSV
            writer.writerow(row)

print(f"Filtered data written to {output_file}")
