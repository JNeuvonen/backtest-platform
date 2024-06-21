from typing import List


def rm_items_from_list(original_list: List, items_to_remove: List):
    """
    Removes specified items from the original list.

    Parameters:
    original_list (list): The list from which to remove items.
    items_to_remove (list or set): The items to remove from the original list.

    Returns:
    list: A new list with the specified items removed.
    """
    return [item for item in original_list if item not in items_to_remove]
