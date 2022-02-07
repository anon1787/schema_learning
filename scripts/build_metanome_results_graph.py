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
parser.add_argument('--display', type=str, default='pk,attr',
                    help='comma separated list of [pk, attr, fk]')

args = parser.parse_args()
filename = args.datafile
fields = [x.strip() for x in args.display.split(",")]

import json

data = []
tbls = []
with open(filename) as f:
    for line in f:
        dat = json.loads(line)
        (cols, pk, fk) = extractFields(dat)
        pk.sort()
        fk.sort()
        attr =  list(set(cols) - set(fk) - set(pk))
        attr.sort()
        allcols = pk + fk + attr
        allcols.sort()
        allcols = set(allcols)
        tbls.append(allcols)


for t in tbls:
    print(t)
    for t2 in tbls:
        common = t.intersection(t2)
        if len(common) > 0:
            print(f"\t{common}:\n\t\t {t2}")
