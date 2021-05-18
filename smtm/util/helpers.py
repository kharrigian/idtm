
"""
Common helper functions
"""

## Imports
import os

def flatten(items):
    """

    """
    items_flat = [j for i in items for j in i]
    return items_flat

def make_directory(directory,
                   remove_existing=False):
    """

    """
    ## Remove Existing if Desired
    if os.path.exists(directory) and remove_existing:
        _ = os.system("rm -rf {}".format(directory))
    ## Create
    if not os.path.exists(directory):
        _ = os.makedirs(directory)
