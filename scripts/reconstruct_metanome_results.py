import sys
import os
import argparse

def getCols(columndict):
    return [c["columnIdentifier"] for c in columndict]

def extractFields(dat):
    cols = getCols(dat['columnCombination']['columnIdentifiers'])
    tablename = dat['columnCombination']['columnIdentifiers'][0]['tableIdentifier'] + '.'
    pk = dat['statisticMap']['PrimaryKey']['value'].replace(tablename, '').split(", ")
    if 'ForeignKey' in dat['statisticMap']:
        fk = dat['statisticMap']['ForeignKey']['value'].replace(tablename, '').split(", ")
    else:
        fk = []

    return (cols, pk, fk)


parser = argparse.ArgumentParser(description='parse metanome result.')
parser.add_argument('datafile', type=str, 
                    help='data file')
parser.add_argument('--keys', type=str,
                    help='comma separated list of pks')


args = parser.parse_args()
filename = args.datafile

def tokey(s):
    return ":".join(s)

def revertkey(s):
    return s.split(':')

import json

data = []

fk_map = dict()
attr_map = dict()
ref_count = dict()
with open(filename) as f:
    for line in f:
        dat = json.loads(line)
        (cols, pk, fk) = extractFields(dat)
        pk.sort()
        fk.sort()
        attr =  list(set(cols) - set(fk) - set(pk))
        attr.sort()
        pk_key = tokey(pk)
        fk_map[pk_key] = fk
        attr_map[pk_key] = attr
        ref_count[pk_key] = 0

keys = args.keys.split(',')
keys = [f"'{x}'" for x in keys]

visited = dict()
def traverse(pk, root=False):
    if not root and pk in keys:
        return (0, [])

    if not pk in fk_map:
        if pk != '':
            print(f"Could not find fk: {pk}")
        return (0, [])

    visited[pk] = 1
    (numnodes, cols) = traverse(tokey(fk_map[pk]))
    return (numnodes+1, fk_map[pk] + attr_map[pk] + cols)




#print(fk_map.keys())
#print(keys)
tbls = dict()
for k in keys:
    (v, tbls[k]) = traverse(k, root=True)
    print(f"{k} {v} {tbls[k]}\n")
    

print("Unvisited:")
for k in fk_map:
    if k not in visited:
        print(f"{k} {visited.get(k,0)} {fk_map[k]} {attr_map[k]}")

# count num references for each pk

for (pk_key,fk) in fk_map.items():
    ref_count[pk_key] += 1
    
#for (pk_key, cnt) in ref_count.items():
#    print(f"{pk_key}: {cnt}")



