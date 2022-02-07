use std::cmp::Ordering;
use std::rc::Rc;
use std::time::Instant;

use hashable_rc::HashableRc;
use itertools::{chain, Itertools};
use std::collections::BinaryHeap;
use hashbrown::HashMap;

use crate::entropy::Entropy;
use crate::schema::{get_cols, Column, SColumn, Schema, get_schema_cols, get_schema_numcols, print_schema};
use crate::table::{ColumnMask, Table};
use crate::utils::fp_cmp;

use float_cmp::{F64Margin, approx_eq};
//use crate::float-cmp::{ApproxEq,approx_eq, F32Margin};

/*
    Infer snowflake schemas over the columns in the data set

    This finds subsets of columns (subtables) that are likely to be attributes of the same level of detail.
    It also forms hierarchical relationships between the subsets.

    Status: fairly well tested and stable
            the optimization is exponential, so will slow down dramatically as the number of columns increases
            can take a long time even for smaller numbers of columns if the data set isn't a snowflake
                (cases when the data set needs to be unpivoted first, in particular, seem to be costly)
            currently, won't ever create a level of detail (subtable) from a single column, so some hierarchies may be incomplete.
*/

// Node tracks the branch-and-bound state for the optimization search.
struct Node {
    split: ColumnMask,
    depth: u32,
    lower_score: f64,
    upper_score: f64,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        match fp_cmp(other.lower_score, self.lower_score) { // XXX should I use upper or lower?
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            _ => (&self.split, &self.depth).cmp(&(&other.split, &other.depth)),
        }
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        (&self.split, &self.depth) == (&other.split, &other.depth)
    }
}

impl Eq for Node {}

// Optimizer implements the branch-and-bound search for the schema learning.
pub struct Optimizer {
    pub entropy: Entropy,
    t: Rc<Table>,
    timeout: f64,
    update_interval: f64,
    update_time: f64,

    // alpha softens the cost of storing a column. You might think about it as a hack that is attempting to account
    // for something like run length compression. The value here hasn't been tuned, but it seems to kind of work to
    // discourage the algorithm from pulling out weakly correlated fields together.
    alpha: f64,
    beta: f64,
    tau: f64,
    gamma: f64,
    fk_mult: f64,
    neg_gamma: f64,

    global_upper_score: f64,
    global_best_schema: Rc<Schema>,

    timedout: bool,

    pub cache: HashMap<(HashableRc<Vec<u32>>, ColumnMask), (Rc<Schema>, f64, f64)>,

    empty_result: Rc<Schema>,

    pub calls: usize,
    pub nodes_visited: usize,
    pub cache_hits: usize,

    pub timer: Instant,
}

impl Optimizer {
    pub fn new(t: &Rc<Table>, update_interval: f64, timeout: f64, alpha: f64, beta: f64, tau: f64, gamma: f64, fk_mult: f64, neg_gamma: f64) -> Optimizer {
        Optimizer {
            entropy: Entropy::new(&t),
            t: t.clone(),
	    update_interval: update_interval,
	    update_time: update_interval,
            timeout: timeout,
            alpha: alpha,
            beta: beta,
	    tau: tau,
	    gamma: gamma,
	    neg_gamma: neg_gamma,
	    fk_mult: fk_mult,
	    global_upper_score: std::f64::INFINITY,
            global_best_schema: Rc::new(Vec::new()),
            cache: HashMap::new(),
            empty_result: Rc::new(Vec::new()),
            calls: 0,
            nodes_visited: 0,
            cache_hits: 0,
	    timedout: false,
            timer: Instant::now(),
        }
    }

    fn rle(&self, a: f64, b: f64) -> f64 {
        (self.alpha as f64* (a as f64).ln() + (1.0 - self.alpha) as f64 * (b as f64).ln()).exp() as f64
    }

    pub fn eval(&mut self, exhaustive_ncols: usize) -> (Rc<Schema>, f64) {
        println!("eval:");
	let n = self.entropy.rows;
        let rows = Rc::new((0..n).collect_vec());
	let ncols = self.entropy.distinct.len();
	let exhaustive_ncols = std::cmp::min(ncols, exhaustive_ncols);
        let (schema, score, _upper_score) = self.opt(&rows, &ColumnMask::ones(exhaustive_ncols), 0, std::f64::INFINITY);

	println!("final global best schema: ");
	print_schema(&self.t.clone(), &self.global_best_schema.clone());
        let schema_ncols = get_schema_cols(schema.clone()).count_ones() as usize;

	if schema_ncols < self.entropy.distinct.len() {
	    let base_schema = if schema_ncols < exhaustive_ncols { self.global_best_schema.clone() } else {schema.clone()}; // XXX should I always get the global one?
	    let base_schema_ncols = get_schema_cols(base_schema.clone()).count_ones() as usize;
//	    return (self.global_best_schema.clone(), self.global_upper_score);

//	    let score = self.score_schema_from_scratch(schema.clone(), n); // need to rescore it from scratch in case it got timed out

	    let (greedy_schema, greedy_score) = self.greedy_allocate(&rows, base_schema_ncols..ncols, base_schema.clone());
	    let recalc_score = self.score_schema_from_scratch(greedy_schema.clone(), n);
	    assert!(self.timedout || approx_eq!(f64, greedy_score, recalc_score, F64Margin { ulps: 10, epsilon: 10.0 }), "greedy score {} != {} recalc", greedy_score, recalc_score);

//	    print_schema(&self.t.clone(), &greedy_schema.clone());
	    return (greedy_schema, greedy_score);
	} else {
	    return (schema, score);
	}
    }

    // Note: This is only good for single column evaluations with rle param alpha != 1 since the effective n depends on the column's distinct count
    pub fn eval_score(n: f64, cols: &[SColumn]) -> f64 {
        let linear: f64 = cols.iter().map(|x| x.fk_cost).sum();
        let constant: f64 = cols.iter().map(|x| x.pk_cost).sum();
        linear * n + constant
    }

    pub fn eval_score_with_rle(&mut self, n: f64, cols: &[SColumn]) -> f64 {
        let linear: f64 = cols.iter().map(|x| self.rle(n, x.rows as f64) * x.fk_cost).sum();
        let constant: f64 = cols.iter().map(|x| x.pk_cost).sum();
        linear + constant
    }

    fn score_schema(&self, schema: Rc<Schema>, top_table_n: u32) -> f64 {
	let n = top_table_n; //self.t.rows;

        // iterate over top level only since all the subtable costs are already rolled up to the PK cost 
	let cost = schema.iter().map(|x| match x.column {
	    Column::Index(_) => x.pk_cost + x.fk_cost * self.rle(n as f64, x.rows as f64),
	    Column::Nested(_) => x.pk_cost + x.fk_cost * self.rle(n as f64, x.rows as f64),
	}).sum();

	let cost2 = self.score_schema_from_scratch(schema.clone(), n);
        print_schema(&self.t.clone(), &schema.clone());
        assert!(approx_eq!(f64, cost, cost2, F64Margin { ulps: 2, epsilon: 2.0 }), "recalc score {} != {} score (from scratch)", cost, cost2);

	cost
    }

    fn get_max_cardinality_column(&self, schema: Rc<Schema>) -> (usize, u32, f64) {
	let mut max_idx = 0;
	let mut max_n = 0;
	let mut e = 0.0;

	for scol in schema.iter() {
	    let (col_idx, col_n) = match scol.column {
		Column::Index(z) => (z, scol.rows),
		_ => (0, 0)};

	    if col_n > max_n {
		max_n = col_n;
		max_idx = col_idx;
		e = scol.fk_cost;
	    }
	}

	(max_idx, max_n, e)
    }

    fn pk_len(&self, schema: Rc<Schema>, table_n: u32) -> u32 {
	let (_idx, max_cardinality, _e) = self.get_max_cardinality_column(schema.clone());
//	println!("max n {} table_n {}", max_cardinality, table_n);
	let pk_len = if max_cardinality == table_n {1} else {(*schema).len() as u32};
	
	pk_len
    }

    fn get_gamma_penalty(&self, schema: Rc<Schema>, table_n: u32) -> (f64, f64) {
	let (_idx, max_cardinality, e) = self.get_max_cardinality_column(schema.clone());
	let gamma_penalty = if max_cardinality == table_n {
	    if true || schema.len() > 1 {
   	       (0.0, -self.neg_gamma * e * table_n as f64)
	    } else {
	       (0.0, 0.0)
	    }

	} else {
	    (self.gamma, 0.0)
	};

	gamma_penalty
    }   

    fn score_schema_from_scratch(&self, schema: Rc<Schema>, table_n: u32) -> f64 {
	let mut cost = 0.0;

        // boilerplate code to traverse the schema
	for scol in schema.iter() {	    
	    let m = scol.rows;
	    let eff_n = self.rle(table_n as f64, m as f64);
	    let delta = match &scol.column {
		Column::Index(_) => scol.pk_cost + scol.fk_cost * eff_n,
		Column::Nested(child_schema) => {
		    let (gamma_fk_penalty, gamma_pk_boost) = self.get_gamma_penalty((*child_schema).clone(), scol.rows);

		    self.score_schema_from_scratch((*child_schema).clone(), scol.rows) + 
		                  eff_n * scol.fk_cost + 
		                  self.beta * m as f64 + 0.0*gamma_pk_boost
		}, // penalty on number of distinct values in the foreign key
	    };
	    cost += delta;
//	    println!("acc cost: {} delta {}  orig_cost {} pk_cost {} fk_cost {} eff_n {} table_n {} m {}", cost, delta, scol.pk_cost + eff_n * scol.fk_cost, scol.pk_cost, scol.fk_cost, eff_n, table_n, m);
	}

	cost
    }

    // deprecated
    fn eval_col_cost(&self, scol: &SColumn, table_n: u32) -> f64 {
	let m = scol.rows;
	let eff_n = self.rle(table_n as f64, m as f64);

	let cost: f64 = match &scol.column {
	    Column::Index(_) => scol.fk_cost * eff_n,
	    Column::Nested(_child_schema) => {
		scol.pk_cost + 
		    eff_n * (scol.fk_cost + self.tau) + // weak correlation and penalty on if pk exists
		    self.beta * m as f64 // penalty on number of distinct values in the foreign key
	    },
	};

	cost
    }

    fn make_scolumn_from_idx(&mut self, r: &Rc<Vec<u32>>, col_idx: usize) -> SColumn {
	let (r2, e) = self.entropy.eval(&r, &ColumnMask::one(col_idx));
	let downweight = if r.len() == r2.len() {self.neg_gamma} else {0.0};
	
	let m = r2.len() as f64;
	let unif_entropy = m.log2() * self.t.weights[col_idx];

	let new_scol = SColumn {
	    column: Column::Index(col_idx),
	    fk_cost: e * self.t.weights[col_idx],
	    pk_cost: -downweight * unif_entropy * r2.len() as f64, 
	    rows: r2.len() as u32};

	new_scol
    }

    // turn a schema into a nested table scolumn with the appropriate costs
    fn make_scolumn_from_schema(&mut self, r: &Rc<Vec<u32>>, schema: Rc<Schema>) -> SColumn {
	let (r2, e) = self.entropy.eval(&r, &get_schema_cols(schema.clone()));
	let n_rows_schema = r2.len() as u32;
//	let pk_len = self.pk_len(schema.clone(), n_rows_schema);
//        println!("nrows {} pk_len {}", n_rows_schema, pk_len);
	let (gamma_fk_penalty, gamma_pk_boost) = self.get_gamma_penalty(schema.clone(), n_rows_schema);
	
	let fk_cost = e + gamma_fk_penalty + self.tau;
	let pk_cost = self.score_schema(schema.clone(), n_rows_schema) + self.beta * n_rows_schema as f64 + 0.0*gamma_pk_boost;

	let new_scol = SColumn {
                column: Column::Nested(schema.clone()),
                fk_cost: fk_cost,
                pk_cost: pk_cost,
                rows: n_rows_schema};

	new_scol
    }
	
    fn greedy_allocate(&mut self, r: &Rc<Vec<u32>>, cols: std::ops::Range<usize>, schema: Rc<Schema>) -> (Rc<Schema>, f64) {
	let mut new_schema = schema.clone();
        let n = r.len() as u32;
        let orig_score = self.score_schema_from_scratch(schema.clone(), n);
	let mut new_score = orig_score;

	for c in cols {
	    println!("Greedy allocate {} {}", c, self.t.names[c]);
	    let (new_greedy_schema, new_greedy_score, _, _) = self.greedy_allocate_col(r, c, new_schema, true);
            new_schema = new_greedy_schema.clone();
	    new_score = new_greedy_score;
            let recalc_score = self.score_schema_from_scratch(new_schema.clone(), n);
//	    print_schema(&self.t.clone(), &new_schema.clone());
	     // timeout shouldn't matter here
            assert!(self.timedout || approx_eq!(f64, new_score, recalc_score, F64Margin { ulps: 10, epsilon: 10.0 }), "greedy score {} != {} recalc, orig {}", new_score, recalc_score, orig_score);
	}

        println!("Done greedy. cost {}", new_score);
	print_schema(&self.t.clone(), &new_schema.clone());
//        println!("Done greedy1. cost {}", new_score);
        let recalc_score = self.score_schema_from_scratch(new_schema.clone(), n);
//        println!("Done greedy2. cost {}", new_score);
// XXX there was an extremely weird bug where new_score = 0...it seems to be a compiler problem.
        assert!(self.timedout || approx_eq!(f64, new_score, recalc_score, F64Margin { ulps: 10, epsilon: 10.0 }), "greedy score {} != {} recalc", new_score, recalc_score);

	(new_schema, new_score)
    }



    fn greedy_allocate_col(&mut self, r: &Rc<Vec<u32>>, col_idx: usize, schema: Rc<Schema>, is_top_level: bool) -> (Rc<Schema>, f64, f64, u32) {
	let schema_cols = get_schema_cols(schema.clone());	
	let new_schema_cols = schema_cols.union(&ColumnMask::one(col_idx));

	// get the new "table" 
//	let (r2, e) = self.entropy.eval(&r, &new_schema_cols);
//	let table_n = if is_top_level {r.len() as u32} else {r2.len() as u32};
	let table_n = r.len() as u32;
//	let r_active = if is_top_level {r.clone()} else {r2.clone()};

	// allocation to top level
	let new_scol = self.make_scolumn_from_idx(&r, col_idx);
	let new_schema = schema.iter().map(|c| { 
		match &c.column {
		    Column::Index(idx) => self.make_scolumn_from_idx(&r, *idx),
		    Column::Nested(child_schema) => self.make_scolumn_from_schema(&r, (*child_schema).clone()),
		}}).collect_vec();
	let mut best_schema = Rc::new(chain(new_schema.clone(), 
					    vec![new_scol]).collect_vec());
	let mut best_cost = self.eval_score_with_rle(table_n as f64, &best_schema);

	// allocate to a child table
	for i in 0..schema.len() {
	    let scol = schema[i].clone();

	    if let Column::Nested(child_schema) = &scol.column {
		let child_schema_cols = get_schema_cols((*child_schema).clone());	
		let new_child_schema_cols = child_schema_cols.union(&ColumnMask::one(col_idx));
		let (r2, e2) = self.entropy.eval(&r, &new_child_schema_cols);
		let new_child_n = r2.len() as u32;
		let (new_child_schema, new_child_cost, _new_child_e, _new_child_n) = self.greedy_allocate_col(&r2, col_idx, (*child_schema).clone(), false);

//		let pk_len = self.pk_len(new_child_schema.clone(), new_child_n);
//		let has_no_pk = if pk_len == 1 {0.0} else {1.0};
		let (gamma_fk_penalty, gamma_pk_boost) = self.get_gamma_penalty(new_child_schema.clone(), new_child_n);		
		let mut candidate_schema_contents = (*schema).clone();

		// replace nested table in the cloned candidate
		candidate_schema_contents[i] = SColumn {
		    column: Column::Nested(new_child_schema.clone()), 
		    fk_cost: e2 //new_child_e 
			+ gamma_fk_penalty + self.tau,
		    pk_cost: new_child_cost + self.beta * new_child_n as f64 + 0.0*gamma_pk_boost,
		    rows: r2.len() as u32, //new_child_n,
		}; 

		let candidate_schema = Rc::new(candidate_schema_contents);
		let candidate_cost = self.eval_score_with_rle(table_n as f64, &candidate_schema);
		println!("candidate: child_n {} old n {} table_n {} cost {} old best {}", new_child_n, scol.rows, table_n, candidate_cost, best_cost);
		if candidate_cost < best_cost {
		    best_cost = candidate_cost;
		    best_schema = candidate_schema;
		    println!("allocated to child: n {} cost {} old n {} parent_n {}", new_child_n, best_cost, scol.rows, table_n);
		}
	    }
	}

	(best_schema, best_cost, 0.0 /*  wrong, now unused? XXX */ , table_n )
    }


    // returns the best schema and its score if it beats must_beat_score and the current global optimal score
    // otherwise returns the empty schema and a lower bound on the score given r and the column mask
    fn opt(&mut self, r: &Rc<Vec<u32>>, cols: &ColumnMask, indent: usize, must_beat_score: f64) -> (Rc<Schema>, f64, f64) {
        let max_depth = cols.count_ones() as u32; // number of columns that must be allocated

//	println!("opt cols max_depth {} indent {} cols {:?} ", max_depth, indent, cols.iter_ones().collect_vec());

        let key = (HashableRc::new(r.clone()), cols.clone());

	// state variables keeping track of "best" schema, upper bound and lower bound
        let (mut best_schema, mut best_upper_score, mut best_lower_score, mut best_depth) = (self.empty_result.clone(), std::f64::INFINITY, 0.0 as f64, 0 as u32);
	let mut best_non_greedy_upper_score = std::f64::INFINITY;
	let mut highest_seen_depth = 0;

	// XXX would need to change this if cache bounds
        if let Some(x) = self.cache.get(&key) {
            self.cache_hits += 1;
	    if x.0 != self.empty_result {		
		return x.clone();
	    } else if x.1 > must_beat_score {
	        println!("prune opt {} {} {:?}", x.1, must_beat_score, cols.iter_ones().collect_vec());
	        return (self.empty_result.clone(), x.1, x.2);
	    } else {
		best_lower_score = x.1;
		best_upper_score = x.2;
	    }
	    // need to continue if there is a potential candidate that beats the must_beat_score
        }
        self.calls += 1;

        if max_depth == 0 {
            return (self.empty_result.clone(), 0.0 as f64, 0.0 as f64);
        }



	// start off with trivial allocation of 1 column
	let (trivial_schema, trivial_lower_score, trivial_upper_score) = self.build_opt(cols, &ColumnMask::one(0), 1, r, std::f64::INFINITY, indent);
//	assert!(trivial_lower_score == trivial_upper_score, "trivial mismatch {} != {}, col {}", trivial_lower_score, trivial_upper_score, cols.first());

        let mut queue = BinaryHeap::new();         
        if max_depth == 1 {
	    return (trivial_schema, trivial_lower_score, trivial_upper_score);
	} else {
            queue.push(Node {
		split: ColumnMask::one(0),
		depth: 1,
		lower_score: trivial_lower_score,
		upper_score: trivial_upper_score,
            });
	}

	// temporary variables for cleaner control flow
	let (mut this_schema, mut this_upper_score, mut this_lower_score);// = (self.empty_result.clone(), std::f64::INFINITY, 0.0 as f64);

        while let Some(el) = queue.pop() {
            if el.lower_score > best_upper_score { continue; } // skip 

	    highest_seen_depth = std::cmp::max(highest_seen_depth+1, el.depth);

	    // check time and collect stats
	    let elapsed_time = self.timer.elapsed().as_secs_f64();
	    if elapsed_time > self.timeout {
	       println!("Timed out. max_depth {} cols {:?} ", max_depth, cols.iter_ones().collect_vec());
	       self.timedout = true;
	       break;
	    } 
	    if elapsed_time > self.update_time && indent == 0 {
	       println!("update: {} {}", self.update_time, self.update_interval);
	       self.update_time = self.update_time + self.update_interval;
	       print_output(self.t.clone(), best_schema.clone(), self, best_upper_score);
	    }	    
            self.nodes_visited += 1;

	    // build allocations putting current column (i.e. column at depth position) into left or right side
            let (depth, l_split, r_split) = (el.depth + 1, el.split.clone(), el.split.union(&ColumnMask::one(el.depth as usize)));

            // build the two alternatives and add to queue if necessary
            // TODO: clean up repetitive code here.
            let (left, l_lower_score, l_upper_score) = self.build_opt(cols, &l_split, depth, r,f64::min(best_upper_score, self.global_upper_score), indent);
            let (right, r_lower_score, r_upper_score) = self.build_opt(cols, &r_split, depth, r, f64::min(best_upper_score, self.global_upper_score), indent);

	    assert!(self.timedout || right == self.empty_result || get_schema_numcols(right.clone()) == depth, "bad right depth: numcols {} depth {}", get_schema_numcols(right.clone()), depth);
	    assert!(self.timedout || left == self.empty_result || get_schema_numcols(left.clone()) == depth, "bad left depth: numcols {} depth {}", get_schema_numcols(left.clone()), depth);
	    assert!(depth <= max_depth, "depth {} > max_depth {}", depth, max_depth);

	    if false && depth == max_depth {
		assert!(l_lower_score == l_upper_score || l_upper_score == f64::INFINITY, "l bounds: {} {} must beat {}", l_lower_score, l_upper_score, must_beat_score);
		assert!(r_lower_score == r_upper_score || r_upper_score == f64::INFINITY, "r bounds: {} {} must beat {} ", r_lower_score, r_upper_score, must_beat_score);
	    }

	    // choose the better split
	    if l_upper_score < r_upper_score {
		this_schema = left.clone();
		this_upper_score = l_upper_score;
		this_lower_score = l_lower_score;
	    } else {
		this_schema = right.clone();
		this_upper_score = r_upper_score;
		this_lower_score = r_lower_score;
	    };	    

                if false && indent <= 2 { // top level
			    println!("l: depth {} maxdepth {} upper bound {} lower {} gap {} l_split{:?} time (min) {}", depth, max_depth, l_upper_score, l_lower_score, l_upper_score - l_lower_score, 
				l_split.iter_ones().collect_vec(),
				     (elapsed_time / 60.0) as f32);
			    print_schema(&self.t.clone(), &left.clone());
			    println!("r: depth {} maxdepth {} upper bound {} lower {} gap {} r_split{:?} time (min) {}", depth, max_depth, r_upper_score, r_lower_score, r_upper_score - r_lower_score, 
				     r_split.iter_ones().collect_vec(),
				     (elapsed_time / 60.0) as f32);
			    print_schema(&self.t.clone(), &right.clone());
		}

            if this_upper_score < best_non_greedy_upper_score ||
               (approx_eq!(f64, this_upper_score, best_non_greedy_upper_score, F64Margin { ulps: 10, epsilon: 10.0 }) && depth > best_depth)
            { // XXX need to fix this

		let ncols = max_depth as usize;
		let schema_ncols = depth as usize;
		let n = self.t.rows;
		if indent == 0 && depth > 15 && depth < max_depth {
		    let (greedy_schema, greedy_score) = self.greedy_allocate(&r, schema_ncols..ncols, this_schema.clone());
		    let recalc_score = self.score_schema_from_scratch(greedy_schema.clone(), n);
		    // xxx timedout shouldn't matter here
		    assert!(self.timedout || approx_eq!(f64, greedy_score, recalc_score, F64Margin { ulps: 10, epsilon: 10.0 }), "greedy score {} != {} recalc", greedy_score, recalc_score);

		    if recalc_score < self.global_upper_score {
			println!("new global upper bound {} old {}", recalc_score, self.global_upper_score);
			print_schema(&self.t.clone(), &greedy_schema.clone());
			println!("From schema:");
			print_schema(&self.t.clone(), &this_schema.clone());
		    }

		    println!("greedy update: greedy upper {} old upper {} old global upper {} this lower {} greedy gap {}", recalc_score, this_upper_score, 
			     self.global_upper_score, this_lower_score, recalc_score - this_lower_score);
		    println!("depth {} max_depth {} n {} n.r {} time (min) {}", schema_ncols, ncols, n, r.len(), (elapsed_time / 60.0) as f32);
		    if recalc_score < self.global_upper_score {
			self.global_upper_score = recalc_score;
			self.global_best_schema = greedy_schema.clone(); // XXX does this cause a reference counter that isn't decremented?
			best_upper_score = recalc_score; 
		    }
		}
   		if indent == 0 && this_upper_score < self.global_upper_score && depth == max_depth {
		    self.global_upper_score = this_upper_score;
		    self.global_best_schema = this_schema.clone(); // XXX does this cause a reference counter that isn't decremented?
		    println!("new global best through exhaustive search {} ", this_upper_score);
		    print_schema(&self.t.clone(), &this_schema.clone());
		    
		}

		assert!(self.timedout || get_schema_numcols(this_schema.clone()) == depth, "bad schema: depth {} ncols {} lower {} upper {} curr_best upper {}", 
			depth, get_schema_numcols(this_schema.clone()), this_lower_score, this_upper_score, best_upper_score);
		best_non_greedy_upper_score = this_upper_score;

	    }

            if this_upper_score < best_upper_score ||
               (approx_eq!(f64, this_upper_score, best_upper_score, F64Margin { ulps: 10, epsilon: 10.0 }) && depth > best_depth)
            { // XXX need to fix this

		// XXX greedy placeholder

                if indent <= 0 { // top level
		    let n = self.t.rows;
		    let recalc_score = self.score_schema(this_schema.clone(), n);
		    if l_upper_score == this_upper_score {
			
			    println!("l update indent {}: depth {} maxdepth {} upper bound {} lower {} gap {} score {} l_split{:?} time (min) {}", indent, depth, max_depth, l_upper_score, l_lower_score, l_upper_score - l_lower_score, recalc_score,
				l_split.iter_ones().collect_vec(),
				     (elapsed_time / 60.0) as f32);

			    print_schema(&self.t.clone(), &left.clone());
		    } else {
			    println!("r update indent {}: depth {} maxdepth {} upper bound {} lower {} gap {} score {} r_split{:?} time (min) {}", indent, depth, max_depth, r_upper_score, r_lower_score, r_upper_score - r_lower_score, recalc_score,
				     r_split.iter_ones().collect_vec(),
				     (elapsed_time / 60.0) as f32);
			    print_schema(&self.t.clone(), &right.clone());
		    }
                    println!(
			"Optimizer: {} calls, {} cache hits, {} nodes visited",
			self.calls, self.cache_hits, self.nodes_visited
                    );

		}

	        // update current best schema using the best upper bound
                best_schema = this_schema.clone();
                best_upper_score = this_upper_score;
		best_lower_score = this_lower_score;
		best_depth = depth;
	        best_non_greedy_upper_score = this_upper_score;
	   }

            // keep track of candidate splits on the first depth columns. These form the basis for splits on depth+1 columns						 
	    let constraint = f64::min(f64::min(best_upper_score, self.global_upper_score), must_beat_score);
            if l_lower_score < constraint || approx_eq!(f64, l_lower_score, constraint, F64Margin { ulps: 10, epsilon: 10.0 }) {
                // dt: (old) only save the schema and score as the best candidate if all columns are assigned (depth = max_depth)
		// dt: (new) always save the upper bound
                if depth < max_depth {
                    queue.push(Node {
                        split: l_split,
                        depth: depth,
                        lower_score: l_lower_score,
			upper_score: l_upper_score,
                    });
                }
            }

	    if false && indent == 2 {
		println!("indent {} depth {} max_depth {} constraint {} l_lower_score {} r_lower_score {} best_upper_score {} self.global_upper_score {}  must_beat_score {}", indent, depth, max_depth, constraint, l_lower_score, r_lower_score, best_upper_score, self.global_upper_score, must_beat_score);
	    }

            if r_lower_score < constraint || approx_eq!(f64, r_lower_score, constraint, F64Margin { ulps: 10, epsilon: 10.0 }) {
                if depth < max_depth {
                    queue.push(Node {
                        split: r_split,
                        depth: depth,
                        lower_score: r_lower_score,
			upper_score: r_upper_score,
                    });
                }
            }



        }
							
        if !self.timedout {
	    // XXX is it ok to aadd to the cache if I don't do better than the must_beat_score? Guessing not
	    // what about the global greedy score?
//	    if best_lower_score < must_beat_score && best_lower_score == best_upper_score{ // && depth == max_depth {
	    if self.isOptimalResult(best_schema.clone(), best_lower_score, best_upper_score) {
		assert!(get_schema_numcols(best_schema.clone()) == max_depth, "bad schema: depth {} ncols {}", max_depth, get_schema_numcols(best_schema.clone()));
		if !(best_lower_score == best_upper_score && best_lower_score != std::f64::INFINITY) && !(best_schema == self.empty_result) {
		    println!("cache insertion is not optimal result lower {} upper {} cols {:?}", best_lower_score, best_upper_score, cols.iter_ones().collect_vec());
		    print_schema(&self.t.clone(), &best_schema.clone());
		}

		self.cache.insert(key, (best_schema.clone(), best_lower_score, best_upper_score));
	    } else {
		self.cache.insert(key, (self.empty_result.clone(), best_lower_score, best_upper_score));
	    }

	}

	if indent == 0 {
	    print_schema(&self.t.clone(), &best_schema.clone());
	    println!("best_upper_score: {}", best_upper_score);
//	    let recalc_score = self.score_schema(best_schema.clone(), (*r).len() as u32);
	    // assert!(timedout || approx_eq!(f64, recalc_score, best_upper_score, F64Margin { ulps: 4, epsilon: 4.0 }), "recalc score {} != {} best_upper_score", recalc_score, best_upper_score);
	}

	if !self.isOptimalResult(best_schema.clone(), best_lower_score, best_upper_score) && !(best_schema == self.empty_result) {
	    let recalc_score = self.score_schema_from_scratch(best_schema.clone(), r.len() as u32); 
	    println!("return is not optimal result lower {} upper {} recalc {} depth {} max_depth {} high depth {} indent {}", best_lower_score, best_upper_score, recalc_score, best_depth, max_depth, highest_seen_depth, indent);
	    print_schema(&self.t.clone(), &best_schema.clone());
	}
//	println!("opt cols max_depth {} indent {} cols {:?} best {} {} {}", max_depth, indent, cols.iter_ones().collect_vec(), best_lower_score, best_upper_score, best_depth);
        (best_schema, best_lower_score, best_upper_score)
    }

///////////////////////////////////////////////////////////////////////

    fn build_a_opt(
        &mut self,
	cols: &ColumnMask,
        a: &ColumnMask,
        r: &Rc<Vec<u32>>,
        must_beat_score: f64,
        indent: usize,
    ) -> (Rc<Schema>, f64, f64) {
        // Pull out a (if none of the early outs succeed)
        let (r2, e) = self.entropy.eval(&r, &a);
        let (n, m) = (r.len() as f64, r2.len() as f64);


	let (a_schema, a_lower_score, a_upper_score) = if a.count_ones() == 1 {
	        let downweight = if n == m {self.neg_gamma} else {0.0};

                let schema = Rc::new(vec![
                    SColumn {
                        column: Column::Index(a.last()),
                        fk_cost: e * self.t.weights[a.last()],
                        pk_cost: -downweight * e * self.t.weights[a.last()] * r2.len() as f64, //self.beta*m, // single columns do not get a penalty since they are stored in all schema
                        rows: r2.len() as u32}]);
                let score = Optimizer::eval_score(self.rle(n, m), &schema);

	        let unif_entropy_cost = (r2.len() as f64).log2() * (self.t.weights[a.last()]);
	        let nested_downweight = self.neg_gamma;
                let child_schema = Rc::new(vec![
                    SColumn {
                        column: Column::Index(a.last()),
                        fk_cost: unif_entropy_cost,
                        pk_cost: -nested_downweight * unif_entropy_cost, // XXX should I allow this?
                        rows: r2.len() as u32}]);
	       
	        let gamma_pk_boost = -0.0*self.neg_gamma * unif_entropy_cost * r2.len() as f64;

	        let nested_schema = Rc::new(vec![
                    SColumn {
                        column: Column::Nested(child_schema.clone()),
                        fk_cost: e,
                        pk_cost: Optimizer::eval_score(self.rle(m, m), &child_schema) + self.beta*m + 0.0*gamma_pk_boost,
                        rows: r2.len() as u32}]);

                let nested_score = Optimizer::eval_score(self.rle(n, m), &nested_schema);
	        
	        if score < nested_score {
                    (schema, score, score)
		} else {
		    (nested_schema, nested_score, nested_score)
		}
	} else if n == m && cols.count_ones() == a.count_ones() {
	    // make sure I don't  end up in an infinite recursion
	    // Do not call opt if r=r2 and a = a union b union c

	    return (self.empty_result.clone(), std::f64::INFINITY, std::f64::INFINITY)
        } else {
                let (schema, lower_score, upper_score) = self.opt(&r2, &a, indent + 1, must_beat_score);
	        if lower_score > must_beat_score && !approx_eq!(f64, lower_score, must_beat_score, F64Margin { ulps: 4, epsilon: 4.0 }) {
//   	            println!("pruned left lower {} upper {} beat {}", lower_score, upper_score, must_beat_score);
	            return (self.empty_result.clone(), lower_score, std::f64::INFINITY)
	        } else {
	            
		    assert!(self.timedout || lower_score == upper_score, "a opt lower {} !- {} upper a {:?}", lower_score, upper_score, a.iter_ones().collect_vec());
//   	            let has_no_pk = if max_cardinality != m {1.0} else {0.0};
		    let (gamma_fk_penalty, gamma_pk_boost) = self.get_gamma_penalty(schema.clone(), m as u32);
                    let schema = Rc::new(vec![SColumn {
                        column: Column::Nested(schema),
                        fk_cost: self.fk_mult * (e + self.tau + gamma_fk_penalty),
                        pk_cost: lower_score +  self.beta*m + 0.0*gamma_pk_boost, // add log(table size) penalty
                        rows: r2.len() as u32}]);
		
                    let score = Optimizer::eval_score(self.rle(n, m), &schema);
	            //println!("return a schema {}", score);
		    //print_schema(&self.t.clone(), &schema.clone());
                    (schema, score, score)
    	        }
        };

	(a_schema, a_lower_score, a_upper_score)
    }

    fn build_c_bounds(&mut self,
		      c: &ColumnMask,
		      r: &Rc<Vec<u32>>,		      
     ) -> (f64, f64) {

        let c_entropies: Vec<(f64,f64)> = c.iter_ones()
            .map(|x| {
                let (r2, e) = self.entropy.eval(&r, &ColumnMask::one(x));
                (r2.len() as f64, e * self.t.weights[x])
            }).collect();

	// slightly tighten lower bound. if everything is in its own table, the entropy is log2(cardinality)
        //let c_lower_score: f64 = c_entropies.iter().map(|(m, e)| {m * e}).sum();
//        let c_lower_score: f64 = c_entropies.iter().map(|(m, _)| {m * m.log2() * (1.0 - self.neg_gamma)}).sum() ;
        let c_lower_score: f64 = 0.0;

	// self.t.rows implies assignment to the central table in the snowflake. This maybe can be tightened to nrows of r.
	let nrows = (*r).len(); //self.t.rows;
        let c_upper_score: f64 = f64::max(c_entropies.iter().map(|(m, e)| {self.rle(nrows as f64, *m) * e}).sum(), c_lower_score);
 
	(c_lower_score, c_upper_score)
    }

				     
    // TODO: does passing in best here help at all?
    fn build_opt(
        &mut self,
        cols: &ColumnMask,
        split: &ColumnMask,
        depth: u32,
        r: &Rc<Vec<u32>>,
        must_beat_score: f64,
        indent: usize,
    ) -> (Rc<Schema>, f64, f64) {
        // Split cols into three groups:
        //  (a) fields pulled out into the current table,
        //  (b) fields put in other tables in the recursive call, and
        //  (c) fields not yet decided.
        let(a,b,c) = cols.split(split, depth);

//	println!("cols {:?} split {:?}", cols.iter_ones().collect_vec(), split.iter_ones().collect_vec());
//	println!("a {:?} b {:?} c {:?}", a.iter_ones().collect_vec(), b.iter_ones().collect_vec(), c.iter_ones().collect_vec());


	assert!(a.count_ones() > 0, "Build opt called on empty split: depth {}, cols a {:?} b {:?} c {:?}", depth, 
		a.iter_ones().collect_vec(), b.iter_ones().collect_vec(), c.iter_ones().collect_vec());

	let (a_schema, a_lower_score, a_upper_score, b_schema, b_lower_score, b_upper_score, c_lower_score, c_upper_score) =
	if a.count_ones() < b.count_ones() {
//	    println!("choose a {} |a| {} over b {} depth {}", a.last(), a.count_ones(), b.last(), depth);

	//placeholder
	    let (a_schema, a_lower_score, a_upper_score) = self.build_a_opt(cols, &a, r, must_beat_score, indent);
	    if a_lower_score == f64::INFINITY {
		return (self.empty_result.clone(), f64::INFINITY, f64::INFINITY);
	    }
	    
	    //  ensure it's strictly larger to avoid bad pruning due to FP errors
	    //  This should only really matter when a search at a greater depth is prematurely pruned and a shallow depth is returned by opt
	    //  A slightly better solution might be to keep track of depth and only require strict inequality when the current depth is different from the saved one
            if a_lower_score > must_beat_score && !approx_eq!(f64, a_lower_score, must_beat_score, F64Margin { ulps: 4, epsilon: 4.0 }) {
		return (self.empty_result.clone(), a_lower_score, std::f64::INFINITY);
            }
	    
            let (c_lower_score, c_upper_score) = self.build_c_bounds(&c, r);
	    
            if a_lower_score + c_lower_score > must_beat_score {
		return (self.empty_result.clone(), a_lower_score + c_lower_score, std::f64::INFINITY);
            }
	    
            let must_beat_for_b = must_beat_score - a_lower_score - c_lower_score + (r.len() as f64)*self.gamma; // XXX gamma fudge?
	    //        let must_beat_for_b = std::f64::INFINITY;
            let (b_schema, b_lower_score, b_upper_score) = self.opt(r, &b, indent + 1, must_beat_for_b);
	    (a_schema, a_lower_score, a_upper_score, b_schema, b_lower_score, b_upper_score, c_lower_score, c_upper_score)
	} else {
//	    println!("not choose a {} over b {} depth {}", a.last(), b.last(), depth);
            let (b_schema, b_lower_score, b_upper_score) = self.opt(r, &b, indent + 1, must_beat_score);
            if b_lower_score > must_beat_score {
                return (self.empty_result.clone(), b_lower_score, std::f64::INFINITY);
            }

	    let (c_lower_score, c_upper_score) = self.build_c_bounds(&c, r);
            if b_lower_score + c_lower_score > must_beat_score {
                return (self.empty_result.clone(), b_lower_score + c_lower_score, std::f64::INFINITY);
            }

            let must_beat_for_a = must_beat_score - b_lower_score - c_lower_score + (r.len() as f64)*self.gamma; // XXX gamma fudge?
	    let (a_schema, a_lower_score, a_upper_score) = self.build_a_opt(cols, &a, r, must_beat_for_a, indent);
            if a_lower_score == std::f64::INFINITY {
		return (self.empty_result.clone(), std::f64::INFINITY, std::f64::INFINITY);
	    }
	    (a_schema, a_lower_score, a_upper_score, b_schema, b_lower_score, b_upper_score, c_lower_score, c_upper_score)
	};


	if indent ==0 && false{
	    println!("a {:?} b {:?} c {:?}", a.iter_ones().collect_vec(), b.iter_ones().collect_vec(), c.iter_ones().collect_vec());
	    println!("aschema");
	    print_schema(&self.t.clone(), &a_schema.clone());
	    println!("bschema ");
	    print_schema(&self.t.clone(), &b_schema.clone());
	}
	// dt: combine schema for this and schema for everything else into actual schema
	// dt: score is the minimizer for columns under consideration (a + b) plus c_score??
        let (return_lower_score, return_upper_score) = (
            a_lower_score + b_lower_score + c_lower_score, 
            a_lower_score + b_lower_score + c_upper_score, // dt: replace with upper bound that is an assignment of 'c' columns to top level table
        );

        // if it is a real optimum, then return a schema
	let return_schema = if self.isOptimalResult(a_schema.clone(), a_lower_score, a_upper_score) && self.isOptimalResult(b_schema.clone(), b_lower_score, b_upper_score) {
	    assert!(self.timedout || a.count_ones() == get_schema_numcols(a_schema.clone()), "a depth {} ncols {} score {}", a.count_ones(), get_schema_numcols(a_schema.clone()), a_lower_score);
	    assert!(self.timedout || b.count_ones() == get_schema_numcols(b_schema.clone()), "b depth {} ncols {} score {}", b.count_ones(), get_schema_numcols(b_schema.clone()), b_lower_score);
	    Rc::new(chain((*a_schema).clone(), (*b_schema).clone()).collect_vec())
	} else {
	    self.empty_result.clone()
	};

        (return_schema, return_lower_score, return_upper_score)
    }

    pub fn isOptimalResult(&mut self, schema: Rc<Schema>, lower: f64, upper: f64) -> bool{
	lower == upper && lower != std::f64::INFINITY && (schema != self.empty_result || upper == 0.0)
    }
}

// Preprocess schema: remove constant fields, combine 1-to-1 fields,
// sort columns from highest entropy to lowest. This makes the optimization search much faster.
pub fn preprocess_table(t: &Rc<Table>, column_order: &Option<Vec<String>>) -> Rc<Table> {
    let mut entropy: Entropy = Entropy::new(&t);
    let rows = Rc::new((0..t.rows).collect_vec());

    let col_lengths = t.cols.iter().map(|i| i.dictionary.len()).collect_vec();    

    // bin together columns of the same length to look for 1-to-1 fields.
    let mut col_bins: HashMap<usize, Vec<(usize, String, f64)>> = HashMap::new();
    for (idx, len) in col_lengths.iter().enumerate() {
        if *len > 1 && t.weights[idx] > 0.0 {
            if col_bins.contains_key(len) {
                let mut added = false;
                for x in col_bins.get_mut(len).unwrap() {
                    let combined_len = entropy.eval(&rows, &(ColumnMask::one(idx).union(&ColumnMask::one(x.0)))).0.len();
                    if combined_len == *len {
                        x.1 = format!("{}, {}", x.1, t.names[idx]);
                        x.2 += t.weights[idx];
                        added = true;
                        break;
                    }
                }
                if !added {
                    col_bins
                        .get_mut(len)
                        .unwrap()
                        .push((idx, t.names[idx].clone(), 1.0));
                }
            } else {
                col_bins.insert(*len, vec![(idx, t.names[idx].clone(), 1.0)]);
            }
        }
    }

    let mut new_cols = col_bins
        .iter()
        .map(|(_, v)| v.clone())
        .flatten()
        .collect_vec();

    // Build fake entropies based on column order
    let mut column_entropies: HashMap<String, f64> = HashMap::new();

    if let Some(v) =  column_order {
	for (i, col) in v.iter().enumerate() {
	    // enter a reversed ordering as the entropy, so early items in the ordering get high "entropy"
	    column_entropies.insert((*col).clone(), 128.0 - i as f64); 
	}
    }
    
    let entropy = new_cols
        .iter()
//        .map(|x| x.2 * crate::entropy::entropy(&t.cols[x.0].indexes, t.cols[x.0].dictionary.len()))
        .map(|x| match column_order {
	    Some(v) => {
		let e = *column_entropies.get(&t.names[x.0]).unwrap();
//		println!("get {} {} {} {}", x.0, t.names[x.0], v[x.0], e);
		e
	    },
	    None => crate::entropy::entropy(&t.cols[x.0].indexes, t.cols[x.0].dictionary.len()) * x.2
	})
        .collect_vec();

    let mut index = (0..new_cols.len()).collect_vec();
    index.sort_by(|a, b| fp_cmp(entropy[*b], entropy[*a]));

    for (i,c) in (&index).iter().enumerate() {
        let c2 = new_cols[*c].0;
       println!("{} {} {} {} {} {}", i, c2, new_cols[*c].1, entropy[*c],
//                  new_cols[*c].2 *
                    crate::entropy::entropy(&t.cols[c2].indexes,
					    t.cols[c2].dictionary.len()), 
		t.cols[c2].dictionary.len());
    }

    new_cols = index.into_iter().map(|i| new_cols[i].clone()).collect_vec();

//    for (a,b,c) in new_cols.iter() { //table.names.iter() {
//       println!("{} {} {}", a, b, c);
//    }

    t.select_cols_with_names(&new_cols)
}

/***********************************************************************/


pub fn print_output(table: Rc<Table>, schema: Rc<Schema>, opt: &Optimizer, score: f64) {
            println!("Final results:");
            print_schema(&table, &schema);

                println!("=> {:.2}", score);

                println!(
                    "Optimizer: {} calls, {} cache hits, {} nodes visited",
                    opt.calls, opt.cache_hits, opt.nodes_visited
                );
                println!(
                    "Entropy: {} calls, elapsed: {:.2}s ({:.2}s group, {:.2}s summarize, {:.2}s cache)",
                    opt.entropy.entropy_calls,
                    opt.entropy.entropy_time,
                    opt.entropy.group_time,
                    opt.entropy.summarize_time,
                    opt.entropy.entropy_time - opt.entropy.group_time - opt.entropy.summarize_time,
                );
                println!(
                    "Cache entries: {} optimizer, {} entropy",
                    opt.cache.len(),
                    opt.entropy.entropy_cache.len()
                );
           


}
