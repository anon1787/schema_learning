#!python

import pandas as pd
import argparse
import random

parser = argparse.ArgumentParser(description='noise arguments.')
parser.add_argument('file', type=str, 
                    help='filename')
parser.add_argument('--nrows', type=int, default=100000000,
                    help='limit number of rows')
parser.add_argument('--noise', dest='noise_percent', action='store', type=float,
                    default=0.0, 
                    help='percent of entries (default: 0%)')

parser.add_argument('--noisetype', dest='noise_type', action='store', type=str,
                    default="replace", 
                    help='type of noise (replace or swap)')

parser.add_argument('--outfile', type=str, default=None,
                    help='limit nuoutput file')

args = parser.parse_args()
#print(args.noise_percent)
#print(args.file)

dat = pd.read_csv(args.file, nrows=args.nrows, dtype='str')
n, d = dat.shape

nentries = int(n*d * args.noise_percent / 100)

for x in range(nentries):
    i = random.randint(0,n-1)
    j = random.randint(0,d-1)
    if args.noise_type == 'replace':
        v = random.getrandbits(32)
        dat.iloc[i,j]= -v
    else:
        i2 = random.randint(0,n-1)
        dat.iloc[i,j], dat.iloc[i2,j] = dat.iloc[i2,j], dat.iloc[i,j]


#print(nentries)


if args.outfile is None:
    import os
    basefile = os.path.splitext(args.file)[0]
    args.outfile = f"{basefile}_noise_{args.noise_percent}percent_type_{args.noise_type}.csv"
print(args.outfile)

dat.to_csv(args.outfile, index=False)




