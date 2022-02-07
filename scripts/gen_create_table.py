#!python

import argparse
import os


parser = argparse.ArgumentParser(description='wrap a plain sql into a create table and copy the output to csv in the docker container')
parser.add_argument('sqlfile', type=str, 
                    help='sql file to run')
parser.add_argument('--outtable', type=str, 
                    help='output table/csv filename')

args = parser.parse_args()

cmd = f"echo CREATE TABLE {args.outtable} AS > tmp.sql"
cmd2 = f"cat {args.sqlfile} >> tmp.sql"
cmd3 = f"echo ; >> tmp.sql"
cmd4 = f"echo COPY {args.outtable} TO '/home/{args.outtable}.csv' DELIMITER ',' CSV HEADER >> tmp.sql";

print(cmd)
print(cmd2)
print(cmd3)
print(cmd4)

os.system(cmd)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)


