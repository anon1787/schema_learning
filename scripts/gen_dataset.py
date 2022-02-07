#!python

import argparse
import os


parser = argparse.ArgumentParser(description='generate dataset.')
parser.add_argument('sqlfile', type=str, 
                    help='sql file to run')

args = parser.parse_args()

copycmd = f"sudo docker cp {args.sqlfile} 4ebd06fbdef3:/sqlfile.sql"
dbcmd = f"sudo docker exec -it 4ebd06fbdef3 psql musicbrainz_db -U musicbrainz -f sqlfile.sql"

print(copycmd)
os.system(copycmd)

print(dbcmd)
os.system(dbcmd)
