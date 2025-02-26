### Exercise 1
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
