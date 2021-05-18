
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

def chunks(l,
           n):
    """
    Yield successive n-sized chunks from l.
    Args:
        l (list): List of objects
        n (int): Chunksize
    
    Yields:
        Chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]