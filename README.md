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

