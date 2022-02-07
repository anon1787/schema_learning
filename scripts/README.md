# Scripts

## Usage
For all the python scripts, you need to activate the python environment

conda activate py38

1. gen_create_table.py sqlfile --outtable outtable
   - locally create tmp.sql which will wrap a sql query in a ddl statement creating outtable + a command to copy the table to a csv of the same name in /home/
2. gen_dataset.py sqlfile
  - (in docker) run the sqlfile in the docker db
3. fetch_file_from_docker.sh filename
  - fetch the file from the docker container and put in in the cwd
4. inject_noise.py filename --noise \%entries_with_noise --noise_type (swap|replace)
  - (locally) read in a csv and inject noise. 
5. sample_cols.py file --nrows nr --ncols nc
   - sample a set of rows/columns to reduce the size of the dataset
6. run_on_data_file.py datafile
   - run schema_learning on datafile and store corresponding result in result subdirectory


