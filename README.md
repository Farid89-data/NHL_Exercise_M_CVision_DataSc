
# Debugging Exercises

This repository contains a series of Python debugging exercises for the NHL Stenden Master's program in Computer Vision & Data Science (2025-2026).

## Exercise 1: Fixed id_to_fruit Function

### Problem Description
The original function was attempting to access elements from a set by index, which is not reliable because sets in Python are unordered collections.

### Solution
The function was modified to accept a `List[str]` instead of a `Set[str]`, allowing for direct indexing into the collection. The implementation was also simplified to use Python's built-in indexing rather than manually iterating through the collection.

### Key Insights
- Sets in Python are unordered collections, meaning you cannot reliably access elements by index
- For operations that require ordered access, lists or tuples should be used instead
- The fixed implementation is more efficient and clearer, using direct indexing rather than iteration

### Usage
```python
fruits_list = ["apple", "orange", "melon", "kiwi", "strawberry"]
name1 = id_to_fruit(1, fruits_list)  # Returns "orange"
name3 = id_to_fruit(3, fruits_list)  # Returns "kiwi"
name4 = id_to_fruit(4, fruits_list)  # Returns "strawberry"
```
### Output
```python
name1 = orange
name3 = kiwi
name4 = strawberry
```

## Exercise 2: Fixed Coordinate Swapping Function

### Problem Description
The original function was attempting to swap x and y coordinates in a numpy array of bounding box coordinates, but was using an invalid approach to simultaneous assignment with numpy array slices.

### Solution
The function was modified to:
1. Create a copy of the input array to avoid modifying the original data
2. Perform each coordinate swap individually rather than attempting to do them all at once

### Key Insights
- In Python, the simultaneous assignment `a, b = b, a` does not work as expected when `a` and `b` are numpy array slices
- When modifying array values based on other values in the same array, it's safer to create a copy first
- Breaking down complex operations into simpler steps can avoid unexpected behaviors

### Usage
```python
coords = np.array([
    [10, 5, 15, 6, 0],
    [11, 3, 13, 8, 0],
    [5, 3, 13, 6, 1]
])
swapped_coords = swap(coords)
# Result: swapped x and y coordinates
# [[5, 10, 6, 15, 0],
#  [3, 11, 8, 13, 0],
#  [3, 5, 6, 13, 1]]
```
### Output
```python
Original coordinates:
[[10  5 15  6  0]
 [11  3 13  8  0]
 [ 5  3 13  6  1]
 [ 4  4 13  6  1]
 [ 6  5 13 16  1]]

Swapped coordinates:
[[ 5 10  6 15  0]
 [ 3 11  8 13  0]
 [ 3  5  6 13  1]
 [ 4  4  6 13  1]
 [ 5  6 16 13  1]]
```

## Exercise 3: Fixed Precision-Recall Curve Plotting

### Problem Description
The original function was attempting to plot a precision-recall curve from CSV data, but had several issues:
1. The axes were swapped compared to the function description
2. The CSV header row was being treated as data
3. The plot had incorrect axis ranges and orientation

### Solution
The function was modified to:
1. Skip the header row when reading the CSV file
2. Convert data to float explicitly
3. Correctly plot precision on the x-axis and recall on the y-axis
4. Set appropriate axis limits (0 to 1.05 for both axes)
5. Add a title and grid for better readability

### Key Insights
- When reading data from CSV files, headers should be handled properly
- Data type conversion is important for numerical operations
- Appropriate axis limits and labels are essential for meaningful data visualization
- Precision-recall curves typically have both axes ranging from 0 to 1

### Usage
```python
# Create a CSV file with precision-recall data
# Format: "precision,recall" with a header row
plot_data("path_to_your_precision_recall_data.csv")
```
### Output

![image](https://github.com/user-attachments/assets/05809ba2-5ddc-4b07-a757-d6af816e5c0a)

## Exercise 4: Fixed GAN Implementation for MNIST Digit Generation

### Problem Description
The original GAN implementation had two bugs:
1. A structural bug that appeared when changing the batch size from 32 to 64, causing a ValueError due to inconsistent tensor dimensions
2. A cosmetic bug related to tensor reshaping that affected the visual quality of the generated outputs

### Solution
The implementation was fixed by:
1. Correctly handling variable batch sizes throughout the training process
2. Ensuring consistent tensor reshaping, especially when passing data between the Generator and Discriminator
3. Improving the training loop to handle the last batch of each epoch correctly

### Key Insights
- GANs are sensitive to tensor dimensions, and inconsistencies can easily cause training failures
- Dynamic batch sizing is necessary to handle the last batch in each epoch, which may be smaller than the specified batch size
- Proper reshaping is crucial when working with image data in GANs
- Consistent handling of data shapes through the generator-discriminator pipeline is essential

### Usage
```python
# Train with default parameters
train_gan()

# Train with custom parameters
train_gan(batch_size=64, num_epochs=100, device="cuda:0")
```
### Output

![image_2025-02-26_13-32-46](https://github.com/user-attachments/assets/d37944fe-55fc-4daa-92b0-6b908caf61c5)

### Installation and Usage
Clone the repository and install required dependencies:
Clone the repository and install required dependencies:
```pyton
git clone https://github.com/Farid89-data/NHL_Exercise_M_CVision_DataSc.git
cd NHL_Exercise_M_CVision_DataSc
pip install -r requirements.txt
```
