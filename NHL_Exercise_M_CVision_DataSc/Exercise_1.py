from typing import List

def id_to_fruit(fruit_id: int, fruits: List[str]) -> str:
    """
    This method returns the fruit name by getting the string at a specific index of the list.
    
    :param fruit_id: The id of the fruit to get
    :param fruits: The list of fruits to choose the id from
    :return: The string corresponding to the index `fruit_id`
    """
    if 0 <= fruit_id < len(fruits):
        return fruits[fruit_id]
    raise RuntimeError(f"Fruit with id {fruit_id} does not exist")

# Test the fixed function
fruits_list = ["apple", "orange", "melon", "kiwi", "strawberry"]
name1 = id_to_fruit(1, fruits_list)
name3 = id_to_fruit(3, fruits_list)
name4 = id_to_fruit(4, fruits_list)

print(f"name1 = {name1}")  # Should output: name1 = orange
print(f"name3 = {name3}")  # Should output: name3 = kiwi
print(f"name4 = {name4}")  # Should output: name4 = strawberry
