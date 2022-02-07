#!python
# Replace XXXX in directories at the bottom with the appropriate directory 

import sys
import os
import argparse

parser = argparse.ArgumentParser(description='generate dataset.')
parser.add_argument('datafile', type=str, 
                    help='data file')
parser.add_argument('--alpha', type=float, default=0.94,
                    help='rle discounting parameter (range: rle 0 - 1 no rle)')
parser.add_argument('--beta', type=float, default=0.0,
                    help='penalty on new tables/fk (>= 0)')
parser.add_argument('--tau', type=float, default=0.0,
                    help='penalty allowing for weak dependence (>= 0)')
parser.add_argument('--gamma', type=float, default=0.0,
                    help='penalty on creating a table without a primary key (>= 0)')
parser.add_argument('--fk_mult', type=float, default=1.0,
                    help='cost multiplier for fk cost (> 0)')
parser.add_argument('--limit', type=int, default=127,
                    help='max number of columns in exhaustive search')
parser.add_argument('--column_order_file', type=str, default=None,
                    help='file specifying column order')
parser.add_argument('--name', type=str, default='',
                    help='tag output file with name')


parser.add_argument('--timeout', type=float, default=3600.0,
                    help='timeout in secs (default 36000 (10 hr)')

args = parser.parse_args()
filename = args.datafile

abs_file = os.path.abspath(filename)
datadir = os.path.dirname(abs_file)
cwd = os.path.abspath(os.getcwd())

basefile = os.path.splitext(os.path.basename(filename))[0]

if args.column_order_file is not None:
    col_order = f"--column-order-file {args.column_order_file}"
    filename_extra = f"_orderedcols{args.name}"
else:
    col_order = ""
    filename_extra ="{args.name}"

os.chdir('/home/XXXXX/research/autobi/schema_learning')
cmd = f"/usr/bin/time -v cargo run --release '{abs_file}' --alpha {args.alpha} --beta {args.beta} --tau {args.tau} --gamma {args.gamma} --timeout {args.timeout} --limit {args.limit} --fk-mult {args.fk_mult} {col_order} 2>&1 | tee /home/XXXX/research/autobi/results/{basefile}_results_alpha_{args.alpha}_beta_{args.beta}_tau_{args.tau}_gamma_{args.gamma}_fkmult_{args.fk_mult}_limit_{args.limit}{filename_extra}.txt"
os.system(cmd)
