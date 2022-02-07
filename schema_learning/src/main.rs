mod entropy;
mod optimizer;
mod schema;
mod table;
mod utils;

use std::io::Write;
use std::rc::Rc;
use std::time::Instant;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use crate::optimizer::{preprocess_table, Optimizer, print_output};
use crate::schema::{count_tables, print_schema};
use crate::table::load_table;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(about = "Infer snowflake schema (stable)")]
pub struct Opts {
    #[structopt(long, help = "Limit on the number of columns modeled")]
    limit: Option<usize>,
    #[structopt(long, help = "Enable CSV output")]
    csv: bool,
    #[structopt(long, help = "Quiet output")]
    quiet: bool,
    #[structopt(long, help = "Timeout in seconds")]
    timeout: Option<f64>,
    #[structopt(long, help = "rle alpha value [0 (rle) - 1 (no rle)]", default_value="0.94")]
    alpha: f64,
    #[structopt(long, help = "constant penalty beta value (>= 0). larger discourages new tables", default_value="1.0")]
    beta: f64,
    #[structopt(long, help = "mutual information penalty (>= 0). should be small. larger discourages new tables", default_value="0.0")]
    tau: f64,
    #[structopt(long, help = "penalize tables without a primary key", default_value="0.0")]
    gamma: f64,
    #[structopt(long, help = "reduce cost of creating table with primary key", default_value="0.0")]
    neg_gamma: f64,
    #[structopt(long, help = "filename containing column oder")]
    column_order_file: Option<String>,
    #[structopt(long, help = "fk cost multiplier (> 0)", default_value="1.0")]
    fk_mult: f64,
    #[structopt(
        name = "FILE",
        help = "Input file name (if missing will read from stdin)"
    )]
    filename: Option<String>,
}

//use crate::schema::Schema;
//use crate::table::Table;

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn load_column_order(column_order_file: &Option<String>) -> Option<Vec<String>> {
    match column_order_file {
        Some(filename) => {
	    let mut vec = Vec::new();
	    if let Ok(lines) = read_lines(filename) {
		// Consumes the iterator, returns an (Optional) String
		for line in lines {
		    if let Ok(col) = line {
			println!("{}", col);
			vec.push(col);
		    }
		}
	    }
	    Some(vec)
        }
        None => None
    }
}

fn main() {
    let opts = &Opts::from_args();

    if opts.csv {
        match &opts.filename {
            Some(x) => print!("\"{}\",", x),
            None => print!("\"stdin\""),
        }
    } else {
        match &opts.filename {
            Some(x) => println!("{}", x),
            None => println!("stdin"),
        }
    }

    let original_table = load_table(&opts.filename);

    let original_table = match original_table {
        Ok(x) => Rc::new(x),
        Err(err) => {
            print!("Error loading CSV: {}", err);
            return;
        }
    };

    let now = Instant::now();

    let ncols = original_table.cols.len();

    let column_order = load_column_order(&opts.column_order_file);
    let table = preprocess_table(&original_table, &column_order);
    let ncols = std::cmp::min(ncols, table.cols.len());
    let limit_ncols = opts.limit.unwrap_or(ncols);

    if opts.csv {
        print!(
            "{},{},{},",
            original_table.rows,
            original_table.cols.len(),
            limit_ncols
        );
        let _ = std::io::stdout().flush();
    } else {
        println!("Cols: {} Rows: {}", table.cols.len(), table.rows);
    }

    if ncols > 127 {
        if opts.csv {
            println!("\"Too many columns to compute snowflake (limit 127)\"");
        } else {
            println!("Too many columns to compute snowflake (limit 127)");
        }
        return;
    }

    let update_interval = 60.0*10.0; // 10 mins
    let timeout = match opts.timeout {
                Some(x) => x,
                _ => 3600.0*12.0, // 12 hours
    };


    let elapsed = {
        let mut opt = Optimizer::new(&table, update_interval, timeout, opts.alpha, opts.beta, opts.tau, opts.gamma, opts.fk_mult, opts.neg_gamma);
        let (schema, score) = opt.eval(limit_ncols);

        let elapsed = now.elapsed().as_secs_f64();

        if opts.csv {
            match opts.timeout {
                Some(x) if elapsed > x => println!(
                    "\"Timeout\",{:.5},{},{:.2}",
                    elapsed,
                    count_tables(&schema),
                    score
                ),
                _ => println!(
                    "\"OK\",{:.5},{},{:.2}",
                    elapsed,
                    count_tables(&schema),
                    score
                ),
            };
        } else {
	  if !opts.quiet {
	  	  print_output(Rc::clone(&table), Rc::clone(&schema), &opt, score);
	  }		
        }

        elapsed
    };

    if !(opts.csv || opts.quiet) {
        let total_elapsed = now.elapsed().as_secs_f64();
        println!(
            "Total elapsed: {:.2}s ({:.2}s freeing memory)",
            total_elapsed,
            total_elapsed - elapsed
        );
        println!("");
    }
}
