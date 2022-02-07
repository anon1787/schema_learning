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
                    help='comma separated list of [pk, attr, fk] or all')

parser.add_argument('--sample', type=int, default='1000',
                    help='number of tables to display')

args = parser.parse_args()
filename = args.datafile
fields = [x.strip() for x in args.display.split(",")]

import json

data = []
with open(filename) as f:
    lines = [line for line in f]

import numpy.random

if args.sample < len(lines):
    lines = numpy.random.choice(lines, args.sample)
    
for line in lines:
        dat = json.loads(line)
        (cols, pk, fk) = extractFields(dat)
        pk.sort()
        fk.sort()
        attr =  list(set(cols) - set(fk) - set(pk))
        attr.sort()
        allcols = list(set(pk + fk + attr))        
        allcols.sort()
        s = ""
        if 'pk' in fields:
            s += f"pk: {pk!s: <80}"

        if 'fk' in fields:
            s += f"fk: {fk!s: <80}"

        if 'attr' in fields:
            s += f"attr: {attr!s: <80}"

        if 'all' in fields:
            allcols_str = "\n".join(allcols)
            s = f"{allcols_str}\n"

#        if 'all' in fields:
#            s = f"{allcols}\n"

        print(s)

