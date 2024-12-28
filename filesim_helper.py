import os
import re
import sys
import pickle
import numpy as np

# Return the set of tuples from a given word list
def kshingles(data:list, k:int = 3) -> set:
    shingles = {tuple(data[i:i+k]) for i in range(len(data)-k-1)}   # set comprehension
    return shingles

# Get the list of words in a file
def get_words(fname:str) -> list:
    if(not os.path.exists(fname)):
        print(f"Error: file, \"{fname}\" does not exist.", file=sys.stderr)
        sys.exit()
    with open(fname, "r") as fd:
        raw_data = fd.read()
    raw_data = re.sub('[0-9]', '', raw_data)    # Investigate some better regex that doesn't grab empty strings
    data = re.split(r'\W+', raw_data.lower())
    return data[:-1]

# Add some value to a dict if it isn't already there, otherwise increment it
def add_to_dict(key_name:str, dictionary:dict, on_creation:int=1) -> None:
    if (key_name not in dictionary):
        dictionary[key_name] = on_creation
    else:
        dictionary[key_name] += 1

def read_pickle(fname:str):
    if(not os.path.exists(fname)):
        print(f"Error: file, \"{fname}\" does not exist.", file=sys.stderr)
        sys.exit()
    with open(fname, "rb") as fd:
        data = pickle.load(fd)
    return data

# Make shingle binary array
def charfunc(imp_shingles:dict, fname:str) -> np.array:
    n = len(imp_shingles)
    if(n == 0):
        print("Error: Shingle dictionary is empty.", file=sys.stderr)
        sys.exit()
    shtype = type(list(imp_shingles.keys())[0])
    if(shtype != tuple):
        print(f"Error: Shingles dictionary keys in type {shtype} when they should be of type {tuple}.", file=sys.stderr)
        sys.exit()

    # Open shingles of given file
    words = get_words(fname) 
    shins = kshingles(words, k=3)

    # The output array, 1 = shingle in file, 0 = not
    shin_bin = np.zeros(n, dtype=np.uint8)

    for shin in shins:
        ind = imp_shingles.get(shin)
        if(ind != None):
            shin_bin[ind] = 1

    return shin_bin