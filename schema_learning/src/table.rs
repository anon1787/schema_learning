use itertools::Itertools;
use hashbrown::HashMap;
use hashbrown::hash_map::Entry;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::rc::Rc;
use bitintr::{Pdep};

// Dictionary-encoded column representation (with up to ~4B entries)
#[derive(Default)]
pub struct Column<T> {
    pub indexes: Rc<Vec<u32>>,
    pub dictionary: Rc<Vec<T>>,
}

impl<T> Clone for Column<T> {
    fn clone(&self) -> Self {
        Column::<T> {
            indexes: self.indexes.clone(),
            dictionary: self.dictionary.clone(),
        }
    }
}

// Table with named, weighted, dictionary-encoded columns
pub struct Table {
    pub names: Vec<String>,
    pub weights: Vec<f64>,
    pub cols: Vec<Column<String>>,
    pub rows: u32,
}

impl Table {
    // Select a subset of the columns without changing names
    /*pub fn select_cols(&self, indices: &[usize]) -> Rc<Table> {
        let names = indices.iter().map(|i| self.names[*i].clone()).collect_vec();
        let cols = indices.iter().map(|i| self.cols[*i].clone()).collect_vec();

        Rc::new(Table {
            names,
            cols,
            rows: self.rows,
        })
    }*/

    // Select a subset of the columns with new names
    pub fn select_cols_with_names(&self, indices: &[(usize, String, f64)]) -> Rc<Table> {
        let names = indices.iter().map(|(_, n, _)| n.clone()).collect_vec();
        let weights = indices.iter().map(|(_, _, w)| w.clone()).collect_vec();
        let cols = indices
            .iter()
            .map(|(i, _, _)| self.cols[*i].clone())
            .collect_vec();

        Rc::new(Table {
            names,
            weights,
            cols,
            rows: self.rows,
        })
    }
}

fn load_from_csv(reader: &mut dyn BufRead, delimiter: u8) -> Result<Table, Box<dyn Error>> {
    let mut names: Vec<String> = Vec::new();
    let mut cols: Vec<(Vec<u32>, Vec<String>)> = Vec::new();
    let mut idicts: Vec<HashMap<String, usize>> = Vec::new();

    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_reader(reader);

    {
        let headers = rdr.headers()?;
        for r in headers.iter() {
            names.push(r.to_string());
            cols.push((Vec::default(), Vec::default()));
            idicts.push(HashMap::new());
        }
    }

    for result in rdr.records() {
        let record = result?;

        for (i, r) in record.iter().enumerate() {
            let len = cols[i].1.len();
            let idx = *idicts[i].entry(r.to_string()).or_insert(len);
            cols[i].0.push(idx as u32);
            if idx == len {
                cols[i].1.push(r.to_string());
            }
        }
    }

    let rows = cols[0].0.len() as u32;

    let cols = cols
        .into_iter()
        .map(|c| Column::<String> {
            indexes: Rc::new(c.0),
            dictionary: Rc::new(c.1),
        })
        .collect_vec();

    let weights = vec![1.0; cols.len()];

    Ok(Table { names, weights, cols, rows })
}

pub fn load_table(filename: &Option<String>) -> Result<Table, Box<dyn Error>> {
    match filename {
        Some(filename) => {
            let file = File::open(filename)?;
            if filename.ends_with("tsv") {
                load_from_csv(&mut BufReader::new(file), b'\t')
            } else {
                load_from_csv(&mut BufReader::new(file), b',')
            }
        }
        None => load_from_csv(&mut BufReader::new(io::stdin()), b','),
    }
}

// Combines dictionary-encoded columns together into tuples with a combined dictionary.
// Uses a passed in combiner function to generate the dictionary entries.
pub struct Combiner<S, T> {
    combine_fn: Box<dyn Fn(u32, &T, &S) -> T>,
    // Reusable storage so we don't have to reallocate a hashmap over and over.
    hash: HashMap<(u32, u32), u32>
}

impl<S, T> Combiner<S, T> {
    pub fn new(combine_fn: Box<dyn Fn(u32, &T, &S) -> T>) -> Combiner<S, T> {
        Combiner {
            combine_fn,
            hash: HashMap::new(),
        }
    }

    pub fn combine(&mut self, a: &Column<T>, b: &Column<S>) -> Column<T> {
        let mut dictionary =
            Vec::with_capacity(std::cmp::max(a.dictionary.len(), b.dictionary.len()));
        let mut indexes = Vec::with_capacity(std::cmp::max(a.indexes.len(), b.indexes.len()));

        for (i, (x, y)) in a.indexes.iter().zip(b.indexes.iter()).enumerate() {
            let group = match self.hash.entry((*x, *y)) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    let result = dictionary.len() as u32;
                    dictionary.push((self.combine_fn)(i as u32, &a.dictionary[*x as usize], &b.dictionary[*y as usize]));
                    e.insert(result);
                    result
                }
            };

            indexes.push(group);
        }

        self.hash.clear();

        Column::<T> {
            indexes: Rc::new(indexes),
            dictionary: Rc::new(dictionary),
        }
    }
}

// A set of columns stored as a bit mask
#[derive(Default, PartialEq, Eq, Copy, Clone, Hash, PartialOrd, Ord)]
pub struct ColumnMask {
    v: u128
}

impl ColumnMask {
    pub fn empty() -> ColumnMask {
        ColumnMask{ v: 0_u128 }
    }

    pub fn ones(v: usize) -> ColumnMask {
        ColumnMask{ v: (1_u128 << v) - 1 }
    }

    #[inline]
    pub fn one(v: usize) -> ColumnMask {
        ColumnMask{ v: 1_u128 << v }
    }

    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.v.count_ones()
    }

    #[inline]
    pub fn last(&self) -> usize {
        (127 - self.v.leading_zeros()) as usize
    }

    #[inline]
    pub fn first(&self) -> usize {
        self.v.trailing_zeros() as usize
    }

    #[inline]
    pub fn all_but_last(&self) -> ColumnMask {
        ColumnMask{v: self.v & !(ColumnMask::one(self.last()).v)}
    }

    #[inline]
    pub fn contains(&self, b: &ColumnMask) -> bool {
        (self.v | b.v) == self.v
    }

    #[inline]
    pub fn union(&self, b: &ColumnMask) -> ColumnMask {
        ColumnMask{ v: self.v | b.v }
    }

    #[inline]
    pub fn diff(&self, b: &ColumnMask) -> ColumnMask {
        ColumnMask{ v: self.v & !b.v }
    }

    pub fn split(&self, split: &ColumnMask, depth: u32) -> (ColumnMask, ColumnMask, ColumnMask) {
        let mask = ColumnMask::ones(depth as usize);

        let a = split.v & mask.v;
        let b = !split.v & mask.v;
        let c = !mask.v;

        let cols_low = self.v as u64;
        let cols_high = (self.v >> 64) as u64;
        let low_cnt = cols_low.count_ones();

        // now spread these across the cols
        let a = ((a as u64).pdep(cols_low) as u128) + ((((a >> low_cnt) as u64).pdep(cols_high) as u128) << 64);
        let b = ((b as u64).pdep(cols_low) as u128) + ((((b >> low_cnt) as u64).pdep(cols_high) as u128) << 64);
        let c = ((c as u64).pdep(cols_low) as u128) + ((((c >> low_cnt) as u64).pdep(cols_high) as u128) << 64);

        (ColumnMask{v:a},ColumnMask{v:b},ColumnMask{v:c})
    }

    pub fn iter_ones(&self) -> Box<dyn Iterator<Item=usize>> {
        let v = self.v;
        Box::new((0_usize..128)
            .filter(move |x| (v >> x) & 1 == 1))
    }
}
