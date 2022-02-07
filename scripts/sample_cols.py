#!python

import pandas as pd
import argparse
import numpy as np
import numpy.random


parser = argparse.ArgumentParser(description='sample rows and columns to csv to reduce the size.')
parser.add_argument('file', type=str, 
                    help='filename')
parser.add_argument('--nrows', type=int, default=100000000,
                    help='limit number of rows')
parser.add_argument('--ncols', type=int, default=64,
                    help='limit number of cols')


parser.add_argument('--outfile', type=str, default=None,
                    help='limit nuoutput file')

args = parser.parse_args()
#print(args.noise_percent)
#print(args.file)

dat = pd.read_csv(args.file, nrows=args.nrows, dtype='str')
n, d = dat.shape

args.nrows = min(n, args.nrows)
args.ncols = min(d, args.ncols)

i = np.random.choice(n, args.nrows, replace=False)
j = np.random.choice(d, args.ncols, replace=False)
print(dat.iloc[i,j].shape)

#print(nentries)


if args.outfile is None:
    import os
    basefile = os.path.splitext(args.file)[0]
    args.outfile = f"{basefile}_nrow_{args.nrows}_ncol_{args.ncols}.csv"
print(f"output: {args.outfile}")
dat.to_csv(args.outfile, index=False)




