import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_data(csv_file_path: str):
    """
    This code plots the precision-recall curve based on data from a .csv file,
    where precision is on the x-axis and recall is on the y-axis.
    It is not so important right now what precision and recall means.
    
    :param csv_file_path: The CSV file containing the data to plot.
    """
    # Load data
    results = []
    with open(csv_file_path, 'r') as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        headers = next(csv_reader)  # Skip the header row
        
        # Find column indices (case insensitive)
        precision_index = -1
        recall_index = -1
        for i, header in enumerate(headers):
            if header.lower() == 'precision':
                precision_index = i
            elif header.lower() == 'recall':
                recall_index = i
                
        if precision_index == -1 or recall_index == -1:
            raise ValueError("CSV file must contain 'precision' and 'recall' columns")
            
        for row in csv_reader:
            # Skip empty rows and ensure row has enough data
            if len(row) > max(precision_index, recall_index):
                try:
                    # Convert row values to float
                    precision_val = float(row[precision_index])
                    recall_val = float(row[recall_index])
                    results.append([precision_val, recall_val])
                except (ValueError, IndexError):
                    # Skip rows with non-numeric data or missing values
                    print(f"Skipping invalid row: {row}")
    
    if not results:
        raise ValueError("No valid data rows found in the CSV file")
        
    results = np.array(results)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    # Use column 0 for precision and column 1 for recall in our results array
    plt.plot(results[:, 0], results[:, 1])
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

# Create a sample data file for testing
f = open("data_file.csv", "w")
w = csv.writer(f)
w.writerow(["precision", "recall"])
w.writerows([[0.013,0.951],
             [0.376,0.851],
             [0.441,0.839],
             [0.570,0.758],
             [0.635,0.674],
             [0.721,0.604],
             [0.837,0.531],
             [0.868,0.453],
             [0.962,0.348],
             [0.982,0.273],
             [1.0,0.0]])
f.close()

# Test the fixed function
plot_data("data_file.csv")