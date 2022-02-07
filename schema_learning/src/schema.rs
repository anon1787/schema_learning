use std::rc::Rc;
use ansi_term::Style;

use crate::table::{ColumnMask, Table};

#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Index(usize),
    Nested(Rc<Schema>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SColumn {
    pub column: Column,
    pub fk_cost: f64,       // Cost, per-row, of storing the foreign key for this column
    pub pk_cost: f64,       // Total cost of storing the dictionary or pk-side table for this column
    pub rows: u32,        // Total number of rows in the dictonary or pk-side table for this column
}

pub type Schema = Vec<SColumn>;

pub fn get_schema_cols(schema: Rc<Schema>) -> ColumnMask {
    let mut cols = ColumnMask::default();
    for x in schema.iter() {
	cols = cols.union(&get_cols(&x.column));
    }
    cols
}

pub fn get_schema_numcols(schema: Rc<Schema>) -> u32 {
    let cols = get_schema_cols(schema);
    cols.count_ones()
}

pub fn get_schema_table_cols(schema: Rc<Schema>) -> ColumnMask {
    let mut cols = ColumnMask::default();
    for x in schema.iter() {
	let added = match x.column {
            Column::Index(i) => ColumnMask::one(i),
	    _ => ColumnMask::default(),
	};
	cols = cols.union(&added);
    }
    cols
}

pub fn get_cols(c: &Column) -> ColumnMask {
    match c {
        Column::Index(i) => ColumnMask::one(*i),
        Column::Nested(v) => {
            let mut x = ColumnMask::default();
            for i in &**v {
                x = x.union(&get_cols(&i.column));
            }
            x
        }
    }
}

pub fn count_tables(schema: &[SColumn]) -> usize {
    let t: usize = schema
        .iter()
        .map(|x| match &x.column {
            Column::Index(_) => 0,
            Column::Nested(v) => count_tables(&v),
        })
        .sum();
    t + 1
}

fn print_schema_impl(t: &Table, c: &Schema, indent: usize) {
    for c in c {
        match &c.column {
            Column::Index(i) => {
                for _ in 0..indent {
                    print!(" ");
                }
                println!("{} {} {:.8} {:.8}", t.names[*i], Style::new().italic().dimmed().paint(c.rows.to_string()), c.fk_cost, c.pk_cost)
            }
            _ => (),
        }
    }
    for c in c {
        match &c.column {
            Column::Nested(v) => {
                for _ in 0..indent {
                    print!(" ");
                }
                let title = format!("Table: {} rows, fk cost {:.8}, pk_cost {:.8}", c.rows, c.fk_cost, c.pk_cost);
                let text = Style::new().underline().bold().paint(title);
                println!("  {}", text);
                print_schema_impl(&t, &*v, indent + 2)
            }
            _ => (),
        }
    }
}

pub fn print_schema(t: &Table, c: &Schema) {
    let title = format!("Table: {} rows", t.rows);
    let text = Style::new().underline().bold().paint(title);
    println!("{}", text);
    
    print_schema_impl(t, c, 0);
}
