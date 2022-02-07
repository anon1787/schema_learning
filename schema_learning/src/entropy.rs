use hashable_rc::HashableRc;
use itertools::Itertools;
use hashbrown::HashMap;
use std::rc::Rc;
use std::time::Instant;

use crate::table::{Column, ColumnMask, Combiner, Table};

/*
    Compute entropy of a subset of columns in a table.

    Caches and reuses partial results.
    Finds and tracks functional dependencies to simplify entropy computation (since adding a functionally dependent column to a subset
        won't change the subset's entropy).

    Status: fairly stable
            Entropy computation is often the bottleneck for schema inference. More performance tuning or parallelization could be useful.
*/

pub fn entropy(r: &[u32], bins: usize) -> f64 {
    let mut v = vec![0_u32; bins];

    for b in r {
        v[*b as usize] += 1;
    }

    let mut e = 0.0;
    for p in v {
        if p > 0 {
            let p = (p as f64) / (r.len() as f64);
            e = e - p * p.log2();
        }
    }

    e
}

pub struct Entropy {
    pub rows: u32,
    pub distinct: Vec<Column<u32>>,
    combiner: Combiner<u32, u32>,

    log_table: Vec<f64>, // Precomputed logs so we don't have to repeatedly evaluate

    // Cache grouping results (unique rows)
    group_cache: HashMap<ColumnMask, Column<u32>>,

    // Cache entropy results (unique rows + entropy evaluated relative to the input rows)
    pub entropy_cache: HashMap<(HashableRc<Vec<u32>>, ColumnMask), (Rc<Vec<u32>>, f64)>,

    // Cache known keys (combinations of columns that uniquely determine a given cardinality)
    // Used to help compute FDs and to skip grouping on keys (where the result is the same as the input).
    key_cache: HashMap<usize, Vec<ColumnMask>>,

    // Cache discovered FDs to simplify unique row calculations.
    // If A -> C, then the unique rows of {A,B,C} are the same as the unique rows of {A,B}.
    fd_cache: HashMap<ColumnMask, ColumnMask>,

    pub entropy_calls: usize,
    pub entropy_time: f64,
    pub summarize_time: f64,
    pub group_time: f64,
}

impl Entropy {
    pub fn new(t: &Rc<Table>) -> Entropy {
        // build up vectors with the first occurrence of each unique value in each column.
        let distinct = t
            .cols
            .iter()
            .map(|c| {
                let mut distinct = vec![c.indexes.len() as u32; c.dictionary.len()];
                for (i, v) in c.indexes.iter().enumerate() {
                    if (i as u32) < distinct[*v as usize] {
                        distinct[*v as usize] = i as u32;
                    }
                }
                Column::<u32> {
                    indexes: c.indexes.clone(),
                    dictionary: Rc::new(distinct),
                }
            })
            .collect_vec();

        let combine_fn = |i: u32, _: &u32, _: &u32| i;

        Entropy {
            rows: t.rows,
            distinct: distinct,
            combiner: Combiner::new(Box::new(combine_fn)),

            log_table: (0..t.rows + 1).map(|x| (x as f64).log2()).collect_vec(),

            entropy_cache: HashMap::new(),
            group_cache: HashMap::new(),
            key_cache: HashMap::new(),
            fd_cache: HashMap::new(),

            entropy_calls: 0,
            entropy_time: 0.0,
            summarize_time: 0.0,
            group_time: 0.0,
        }
    }

    pub fn group(&mut self, c: &ColumnMask) -> Column<u32> {
        let now = Instant::now();
        if let Some(x) = self.group_cache.get(&c) {
            self.group_time += now.elapsed().as_secs_f64();
            return x.clone();
        }

        let result = match c.count_ones() {
            0 => Column::<u32> {
                indexes: Rc::new(vec![0; self.rows as usize]),
                dictionary: Rc::new(vec![0]),
            },
            1 => self.distinct[c.last()].clone(),
            _ => {
                // Get the distinct rows for all but the first column in the set
                let t = self.group(&c.all_but_last());

                // Get the distinct rows for the first column in the set
                let h = &self.distinct[c.last()];

                // If one set of distinct rows has only one element or if one is completely unique,
                // then the result is trivial, otherwise combine them to get the new set of unique
                // rows.
                if t.dictionary.len() == 1 || h.dictionary.len() == self.rows as usize {
                    h.clone()
                } else if h.dictionary.len() == 1 || t.dictionary.len() == self.rows as usize {
                    t
                } else {
                    self.combiner.combine(&t, &h)
                }
            }
        };

        self.group_cache.insert(c.clone(), result.clone());
        self.group_time += now.elapsed().as_secs_f64();
        result
    }

    fn summarize(&mut self, r: &[u32], gb: &Column<u32>) -> f64 {
        let now = Instant::now();

        let mut v = vec![0_u32; gb.dictionary.len()];
        for i in r {
            v[gb.indexes[*i as usize] as usize] += 1;
        }

        let n = r.len() as f64;
        let n_log = n.log2();
        let e: f64 = v
            .iter()
            .map(|p| (*p as f64) / n * (self.log_table[*p as usize] - n_log))
            .sum();

        self.summarize_time += now.elapsed().as_secs_f64();

        -e
    }

    pub fn eval(&mut self, r: &Rc<Vec<u32>>, c: &ColumnMask) -> (Rc<Vec<u32>>, f64) {
        let now = Instant::now();

        let key = (HashableRc::new(r.clone()), c.clone());
        if let Some(x) = self.entropy_cache.get(&key) {
            self.entropy_time += now.elapsed().as_secs_f64();
            return x.clone();
        }

        let c = self.expand_column_set(c);

        // If the passed in column set is already a key of the rows, then no need to
        // recompute, we already know the result.
        if let Some(x) = self.key_cache.get(&r.len()) {
            if x.iter().any(|x| c.contains(x)) {
                let result = (r.clone(), self.log_table[r.len()]);
                self.entropy_cache.insert(key, result.clone());
                self.entropy_time += now.elapsed().as_secs_f64();
                return result;
            }
        }

        let d = self.simplify_column_set(&c);

        let result = if d != c {
            let key = (HashableRc::new(r.clone()), d.clone());
            if let Some(x) = self.entropy_cache.get(&key) {
                x.clone()
            } else {
                let result = self.eval_impl(r, &d);
                self.entropy_cache.insert(key, result.clone());
                result
            }
        } else {
            self.eval_impl(r, &c)
        };

        self.entropy_cache.insert(key, result.clone());
        self.entropy_time += now.elapsed().as_secs_f64();
        result
    }

    fn eval_impl(&mut self, r: &Rc<Vec<u32>>, c: &ColumnMask) -> (Rc<Vec<u32>>, f64) {
        self.entropy_calls += 1;

        let gb = self.group(c);
        let result = self.summarize(r, &gb);

        // The grouping columns are a key of the resulting table.
        // Use that fact to update the key and FD information.
        let d = self.expand_column_set(c);
        let e = self
            .key_cache
            .entry(gb.dictionary.len())
            .or_insert(Vec::new());

        // use the key information to update FDs
        let mut inserted = false;
        for x in e.iter() {
            if (x != &d) && d.contains(x) {
                let f = self.fd_cache.entry(x.clone()).or_insert(ColumnMask::default());
                *f = d.union(f);
            }
            if c.contains(x) {
                inserted = true;
            }
        }

        if !inserted {
            e.push(c.clone());
        }

        (gb.dictionary, result)
    }

    fn expand_column_set(&self, c: &ColumnMask) -> ColumnMask {
        self.fd_cache
            .iter()
            .fold(c.clone(), |x, (k, v)| if x.contains(k) { x.union(v) } else { x })
    }

    // Use functional dependencies to simplify the column set.
    // TODO: Use a trie or similar to make it faster to find subsets?
    // If the passed in column set is a superset of a key, we can remove all dependents to simplify the column set without changing the resulting entropy.
    fn simplify_column_set(&self, c: &ColumnMask) -> ColumnMask {
        self.fd_cache
            .iter()
            .fold(c.clone(), |x, (k, v)| if x.contains(k) { x.diff(v).union(k) } else { x })
    }
}
