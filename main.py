#!/usr/bin/env python3

# Input:
#   - A directory
# Output:
#   - All connected component groups

# Can be ran with:
# python main.py .\TestDirs\smallAB
# minVal <= 2

import os
import sys
import numpy as np
import filesim_helper as fsh
from sympy import primerange
from random import choice, randrange

# Obtain file names from a given directory
def get_fnames(files_dir:str, justNames:bool=False) -> list:
    dir_path = os.path.abspath(files_dir)
    file_names = [os.path.join(dir_path, x) for x in os.listdir(files_dir)]
    file_paths = [x for x in file_names if os.path.isfile(x)]
    if(len(file_paths) <= 0):
        print(f"No files found in \"{files_dir}\".")
        sys.exit()
    if justNames:
        return [os.path.basename(x) for x in file_names]
    return file_names

# Build the charfunc matrix from important shingles and a list of file paths
def all_charfunc(imp_shingles:dict, file_paths:list) -> np.array:
    # Build base matrix
    n = len(imp_shingles)
    mat = np.zeros((n,len(file_paths)), dtype=np.uint8)

    # Loop through all files in given directory
    for i, fname in enumerate(file_paths):
        mat[:,i] = fsh.charfunc(imp_shingles, fname)

    return mat

# lambda function generator
def randfun(a:int, b:int, n:int) -> function:
    return lambda x: (a*x+b) % n

# Move stuff into this function when more settled
def minhash(mat:np.array, num_minhashes:int, max_rows:int) -> np.array:
    n = mat.shape[0]            # number of shingles
    n_files = mat.shape[1]      # number of files
    oddprimes = np.array(list(primerange(2, n)))

    minhash_mat = np.empty((num_minhashes, n_files), dtype=np.uint32)

    for k in range(num_minhashes):
        arr = np.empty((max_rows-1, n_files), dtype=np.uint32)
        # Generate hash function using the prime numbers
        fun = randfun(choice(oddprimes),randrange(n), n)
        # Use minhash on up to max_rows rows
        for i in range(max_rows-1):
            j = fun(i)      # Permuted index
            arr[i,:] = mat[j,:]

        arr = np.vstack((np.zeros(n_files), arr))
        minhash_mat[k,:] = np.argmax(arr, axis=0)

    return minhash_mat

# Given a minhash matrix construct an adjacency matrix of the files
#  with edges where there is a vote value of at least reqVotes
def sim_vote(hashmat:np.array, reqVotes:int, blocks:int, rows_per_block:int) -> np.array:
    if (blocks*rows_per_block != hashmat.shape[0]):
        print(f"Error: sim_vote(4), blocks*rows_per_block should be equal to hashmat rows.\n"
              f" You had {blocks} blocks and {rows_per_block} rows per block. Hash matrix had {hashmat.shape[0]} rows",
               file=sys.stderr)
        sys.exit()
    
    n_files = hashmat.shape[1]                  #number of files
    adjmat = np.zeros((n_files, n_files))
    hashmat = np.reshape(hashmat, (blocks, rows_per_block, n_files))

    # Loop through the blocks
    for b_ind in range(blocks):
        sim_dict = {}
        # Loop through each column of the block adding the cols to a dictionary with tuple keys
        for col in range(n_files):
            key = tuple(hashmat[b_ind, :, col])
            if (0 in key):
                # Don't count files that have no similarity
                continue
            if (sim_dict.get(key) != None):
                # Add a vote for an edge on each other vertex with a matching tuple
                for x in sim_dict[key]:
                    adjmat[x,col] += 1
                sim_dict[key].append(col)
            else:
                # Create a key, value pair with ind in a list
                sim_dict[key] = [col]

    #Only return the adjmat where the value >= reqVotes
    adjmat = adjmat >= reqVotes             # Convert matrix to Trues and Falses
    return adjmat.astype(np.uint8)          # Convert to 0s and 1s and return it

# Turn an edge list into an adjacency list that can be used to find strongly connected components
def adjmat2adjlist(adjmat:np.array) -> list:
    n = adjmat.shape[0]
    adjlist = [{
                'vis'    : False,
                'scan'   : False,
                'nbr'    : list(np.nonzero(adjmat[i,:])[0])
            } for i in range(n)]

    return adjlist

def find_not_vis(verts:list, n:int) -> int:    # find an unvisited vertex
    for i in range(n):
        if not verts[i]['vis']:
            return i
    return None

def visit(k:int, verts:list, c_dict:dict, c_ind:int) -> None:
    if not verts[k]['vis']:                 # if it hasn't been visited
        # print(k, end=' ')
        if (c_dict.get(c_ind) != None):
            c_dict[c_ind].append(k) # append k to existing list in dict
        else:
            c_dict[c_ind] = [k]     # add a list with k in it to the dict
        verts[k]['vis'] = True              # mark it visited
        scan(k, verts, c_dict, c_ind)

def scan(k:int, verts:list, c_dict:dict, c_ind:int) -> None:
    verts[k]['scan'] = True
    for i in verts[k]['nbr']:                   # for each neighbor of scanned vertex
        if not verts[i]['vis']:                 # if it hasn't been visited
            visit(i, verts, c_dict, c_ind)

# Create a dictionary of important shingles. Keep only the shingles that appear at least minVal times
def imp_shins(file_paths:list, minVal:int = 4) -> dict:
    shin_freq = dict()      # Shingle Frequency
    # Loop through the files retrieving all of our shingles
    for fname in file_paths:
        words = fsh.get_words(fname)
        for shin in fsh.kshingles(words, k=3):
            fsh.add_to_dict(shin, shin_freq)

    ordered_shin = dict()
    i = 0
    for k in sorted(shin_freq.keys()):
        if(shin_freq[k] >= minVal):
            fsh.add_to_dict(k, ordered_shin, i)
            i += 1

    print(len(ordered_shin), 'shingles')
    # for k in ordered_shin.keys():
    #     print(k)
    return ordered_shin


if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print(f"Usage: {sys.argv[0]} <file-directory> <OPTIONAL:num-minhashes> <OPTIONAL:blocks> <OPTIONAL:rows-per-block>", file=sys.stderr)
        sys.exit()

    files_dir = sys.argv[1]
    if(not os.path.isdir(files_dir)):
        print(f"\"{files_dir}\" is not a directory or cannot be found.", file=sys.stderr)
        sys.exit()    

    # Default values
    num_minhashes = 96
    blocks = 32
    rows_per_block = 3
    votes = 1
    max_rows = 500

    if(len(sys.argv) > 2):
        num_minhashes = int(sys.argv[2])
    if(len(sys.argv) > 3):
        blocks = int(sys.argv[3])
    if(len(sys.argv) > 4):
        rows_per_block = int(sys.argv[4])

    file_paths = get_fnames(files_dir)                      # Get all the file paths
    imp_shingles = imp_shins(file_paths, minVal=2)          # Find all the important shingles that appear atleast minVal times
    mat = all_charfunc(imp_shingles, file_paths)            # Apply the characteristic function to all files to make a matrix
    mat = minhash(mat, num_minhashes, max_rows)             # Minhash the matrix
    sim_mat = sim_vote(mat, votes, blocks, rows_per_block)  # Obtain the adjacency matrix of similar documents

    # Turn the adjacency matrix into an adjacency list that can be used with strong
    n = len(file_paths)
    fnames = get_fnames(files_dir, justNames=True)
    verts = adjmat2adjlist(sim_mat)

    # Find the strongly connected components:
    components = {}
    component_num = 0
    while (i := find_not_vis(verts, n)) != None:
        visit(i, verts, components, component_num)
        component_num += 1
    
    sizeGT2 = 0     # Count of groups of size > 2
    for key in components.keys():
        if len(components[key]) > 2:
            sizeGT2 += 1
            print(f"\nGroup of size {len(components[key])}:")
            print([fnames[x] for x in components[key]])

    print(f"\n{sizeGT2} groups of size > 2 found.")
    print(f"{component_num-sizeGT2} groups of size <= 2 found.")
