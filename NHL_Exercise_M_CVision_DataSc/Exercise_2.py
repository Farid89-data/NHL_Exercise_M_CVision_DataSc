import numpy as np

def swap(coords: np.ndarray):
    """
    This method will flip the x and y coordinates in the coords array.
    
    :param coords: A numpy array of bounding box coordinates with shape [n,5] in format:
        [[x11, y11, x12, y12, classid1],
         [x21, y21, x22, y22, classid2],
         ...
         [xn1, yn1, xn2, yn2, classidn]]
         
    :return: The new numpy array where the x and y coordinates are flipped.
    """
    # Create a copy to avoid modifying the original array
    new_coords = coords.copy()
    
    # Swap x1 with y1 and x2 with y2 for each row
    new_coords[:, 0] = coords[:, 1]  # x1 becomes y1
    new_coords[:, 1] = coords[:, 0]  # y1 becomes x1
    new_coords[:, 2] = coords[:, 3]  # x2 becomes y2
    new_coords[:, 3] = coords[:, 2]  # y2 becomes x2
    
    return new_coords

# Test the fixed function
coords = np.array([[10, 5, 15, 6, 0],
                   [11, 3, 13, 8, 0],
                   [5, 3, 13, 6, 1],
                   [4, 4, 13, 6, 1],
                   [6, 5, 13, 16, 1]])

swapped_coords = swap(coords)
print("Original coordinates:")
print(coords)
print("\nSwapped coordinates:")
print(swapped_coords)