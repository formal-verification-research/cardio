use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use bitvec::prelude::*;
use log::*;
use num::{Rational32, Rational64, Zero, pow::Pow};
use ref_ops::RefAdd;
use sprs::CsMat;
use vector_map::VecMap;

use crate::{checker::ExplicitModelContext, labels::Labels};

/// A trait representing what we need for a matrix entry
pub trait CheckableNumber:
	num::Num
	+ num::Signed
	+ Clone
	+ Copy
	+ Default
	+ Sized
	+ std::iter::Sum
	+ std::cmp::PartialOrd
	+ Div
	+ DivAssign
	+ Add
	+ AddAssign
	+ Mul
	+ MulAssign
	+ Sub
	+ SubAssign
	+ Zero
	+ Pow<i32, Output = Self>
	+ std::fmt::Display
	+ num::FromPrimitive
	+ num::ToPrimitive
	+ std::marker::Send
	+ std::marker::Sync
	+ sprs::MulAcc
	+ RefAdd
{
}

impl CheckableNumber for f64 {}
impl CheckableNumber for f32 {}
impl CheckableNumber for Rational64 {}
impl CheckableNumber for Rational32 {}

/// (Re)-uniformizes a uniformized matrix given that it has already been uniformized
pub fn reuniformize_ma(
	matrix: &mut sprs::CsMat<f64>,
	// steps: &mut Vec<(usize, f64)>,
	old_unif_rate: f64,
	new_unif_rate: f64,
	_deadlock: &BitVec,
) {
	if old_unif_rate == new_unif_rate {
		// No need to re-uniformize
		return;
	} else if old_unif_rate > new_unif_rate {
		panic!("Cannot re-uniformize to smaller uniformization rate!");
	}
	let _diff = new_unif_rate - old_unif_rate;
	let ratio = old_unif_rate / new_unif_rate;

	// Iterate over columns
	for (col_idx, mut col) in matrix.outer_iterator_mut().enumerate() {
		for (row_idx, val_ref) in col.iter_mut() {
			if row_idx == col_idx {
				(*val_ref) -= 1.0;
				(*val_ref) *= ratio;
				(*val_ref) += 1.0;
			} else {
				(*val_ref) *= ratio;
			}
		}
	}
}

/// A trait that represents any type of sparse matrix construction.
pub trait SprsMatBuilder {
	/// Get the value (if it exists) at row `row` and column `col`. If it does not exist, this
	/// function will return None.
	fn get_value(&self, row: usize, col: usize) -> Option<f64>;
	/// Inserts an element. Depending on the struct implementing this trait, this function may
	/// either replace existing values, or just panic if a value already exists.
	fn insert(&mut self, row: usize, col: usize, entry: f64);
	/// Inserts an entire row. Again, depending on the struct implementing the trait, this function
	/// may replace the entire row, or panic if the row is not none.
	fn insert_row<I>(&mut self, row: usize, elements: I)
	where
		I: ExactSizeIterator<Item = (usize, f64)>,
	{
		for (col, entry) in elements {
			self.insert(row, col, entry);
		}
	}
	/// The number of rows
	fn row_count(&self) -> usize;
	/// The sum of a row.
	fn row_sum(&self, row: usize) -> Option<f64>;
	/// A vector of row sums
	fn row_sum_vec(&self) -> Vec<f64> {
		(0..self.row_count())
			.map(|i| {
				if let Some(sum) = self.row_sum(i) {
					sum
				} else {
					0.0
				}
			})
			.collect()
	}
	/// Creates the sparse matrix from the data.
	fn to_sparse_matrix(&mut self) -> sprs::CsMat<f64>;
	/// Creates an infantesimile generator matrix.
	fn to_emb_dtmc(&mut self) -> (f64, sprs::CsMat<f64>);
	/// Creates a uniformized matrix, suitable for model checking.
	fn to_inf_matrix(&mut self) -> (f64, sprs::CsMat<f64>);
	/// Creates an explicit model context, given a labelling structure
	fn to_model_context(&mut self, labels: &Labels, discrete_time: bool) -> ExplicitModelContext {
		let (epoch, mat) = if discrete_time {
			(1.0, self.to_sparse_matrix())
		} else {
			self.to_unif_matrix()
		};
		ExplicitModelContext::new(discrete_time, labels, &mat, epoch)
	}

	fn to_unif_matrix(&mut self) -> (f64, sprs::CsMat<f64>);

	fn get_deadlocks(&self) -> BitVec {
		let n = self.row_count();
		let mut deadlocks = BitVec::with_capacity(n);
		deadlocks.resize(n, false);

		for i in 0..n {
			if let Some(sm) = self.row_sum(i) {
				if sm == 0.0 {
					deadlocks.set(i, true);
				} else {
					deadlocks.set(i, false);
				}
			} else {
				deadlocks.set(i, true);
			}
		}
		deadlocks
	}
}

/// A sparse matrix builder that allows for random access and updating and is optimized for VAS and
/// CRNs. I.e., models that have only a few abstract transitions, but potentially large numbers of
/// states.
pub struct OptimalSprsMatBuilder {
	/// The actual data storage. We use a `VecMap` here since most models we've encountered have
	/// only 5-30 reactions. Therefore it will be more efficient than dealing with the overhead of
	/// a `HashMap` or a `BTreeMap` for every single element.
	data: Vec<Option<VecMap<usize, f64>>>,
	/// The number of elements (not the number of reserved elements) in the builder
	length: usize,
	/// The highest state index
	state_count: usize,
	/// During state insertion, this will be set as the maximal y (destination) values from attempted
	/// insertions (x a.k.a., source must be resized immediately).
	queued_new_size: usize,
	/// The number of abstract transitions in the model serves as a hint for the amount of
	/// elements needed to pre-allocate in each row. In fact, it is an upper bound for it since
	/// the only way the number could be different is if a transition is not enabled at a state.
	abstract_transition_count: Option<usize>,
}

impl OptimalSprsMatBuilder {
	/// The number of nonzero entries in the sparse matrix
	pub fn num_entries(&self) -> usize {
		// Sum the counts of elements for any non-empty row
		self.data
			.iter()
			.filter_map(|row_opt| {
				if let Some(row_val) = row_opt {
					Some(row_val.len())
				} else {
					None
				}
			})
			.sum()
	}

	/// Creates a new OptimalSparseMatrix for CTMC model checking
	pub fn new() -> Self {
		// by default just allocate with a capacity of 5000
		Self::with_row_capacity(5000, None)
	}

	/// Creates a new OptimalSparseMatrix for CTMC model checking with a known capacity
	pub fn with_row_capacity(capacity: usize, abstract_transition_count: Option<usize>) -> Self {
		Self {
			data: Vec::with_capacity(capacity),
			queued_new_size: 0,
			length: 0,
			state_count: 0,
			abstract_transition_count,
		}
	}

	/// Resizes the underlying datastructure by using Rust's efficient `resize_with`. This function
	/// does NOT shrink the datastructure--only increasing the size if necessary.
	pub fn resize(&mut self, size: usize) {
		if size >= self.data.len() {
			self.data.resize_with(size + 1, Default::default)
		}
	}

	/// Applies the queued new size to the datastructure size
	pub fn apply_queued_size(&mut self) {
		if self.state_count == 0 {
			return;
		}
		self.resize(self.queued_new_size);
	}

	/// Gets the number of elements in the matrix
	pub fn len(&self) -> usize {
		self.length
	}

	/// Gets the epoch time, i.e., the maximal row sum.
	pub fn epoch(&self) -> f64 {
		let rows = self.data.len();
		(0..rows)
			.filter_map(|row| self.row_sum(row))
			.reduce(|epoch, mid| if epoch < mid { mid } else { epoch })
			.unwrap_or(f64::zero())
	}
}

impl SprsMatBuilder for OptimalSprsMatBuilder {
	/// Gets the value at a particular row and column
	fn get_value(&self, row: usize, col: usize) -> Option<f64> {
		if row >= self.data.len() {
			None
		} else if let Some(row_val) = &self.data[row] {
			if let Some(element_val) = row_val.get(&col) {
				Some(*element_val)
			} else {
				None
			}
		} else {
			None
		}
	}

	/// Inserts or replaces a value at the position `row, col` in the matrix.
	fn insert(&mut self, row: usize, col: usize, entry: f64) {
		// No need to add if the entry is zero
		if entry == 0.0 {
			return;
		}
		self.state_count = self.state_count.max(row + 1).max(col + 1);
		// Call resize regardless since it will do nothing if we have enough capacity
		self.resize(row);
		// Since we only insert a transition if we've encountered that state, we can assume
		// that the column corresponds to an existing state index. Thus, update the internal
		// queued new size.
		self.queued_new_size = self.queued_new_size.max(col);
		// If this row hasn't been encountered before, we have to create an entry for it
		if self.data[row].is_none() {
			if let Some(trans_count) = self.abstract_transition_count {
				// We can allocate with capacity
				self.data[row] = Some(VecMap::with_capacity(trans_count));
			} else {
				// Use the default allocator
				self.data[row] = Some(VecMap::new());
			}
		}
		// We can unwrap because we've just guaranteed that the element is non-zero
		let row_values = &mut self.data[row].as_mut().unwrap();
		if let Some(_old_value) = row_values.insert(col, entry) {
			warn!(
				"Overwrite inserting value into matrix at ({}, {}): {}",
				row, col, entry
			);
		} else {
			self.length += 1;
		}
	}

	/// Inserts an entire row in the sparse matrix.
	/// Will replace that row if it exists. This is useful if a state that was, say,
	/// previously redirected to the artificial absorbing state is now being expanded. This WILL
	/// NOT WARN if you are overwriting existing data.
	fn insert_row<I>(&mut self, row: usize, elements: I)
	where
		I: ExactSizeIterator<Item = (usize, f64)>,
	{
		// As before, resize.
		self.resize(row);
		// Create a new VecMap with the capacity of the length of `elements`
		self.data[row] = Some(VecMap::with_capacity(elements.len()));
		let row_values = &mut self.data[row].as_mut().unwrap();
		for (col, entry) in elements {
			if let Some(_old_value) = row_values.insert(col, entry) {
				warn!("Overwrite inserting row into matrix at row {}", row);
			} else {
				self.length += 1;
			}
		}
	}

	fn row_count(&self) -> usize {
		self.state_count
	}

	/// Gets the sum of a row. Useful for uniformization in CTMCs.
	fn row_sum(&self, row: usize) -> Option<f64> {
		if row < self.data.len() {
			if let Some(row_val) = &self.data[row] {
				let sm = row_val.iter().map(|(_k, v)| *v).sum();
				Some(sm)
			} else {
				None
			}
		} else {
			None
		}
	}

	/// I attempted to optimize this function by pre-allocating the size for each triplet vector
	fn to_sparse_matrix(&mut self) -> sprs::CsMat<f64> {
		self.apply_queued_size();
		let state_count = self.state_count.max(self.data.len());
		let mut rows = Vec::<usize>::with_capacity(self.len());
		let mut cols = Vec::<usize>::with_capacity(self.len());
		let mut values = Vec::<f64>::with_capacity(self.len());
		for (row, col_option) in self.data.iter().enumerate() {
			if let Some(col_data) = col_option {
				for (col, value) in col_data.iter() {
					rows.push(row);
					cols.push(*col);
					values.push(*value);
				}
			}
		}
		CsMat::new_csc((state_count, state_count), rows, cols, values)
	}

	fn to_emb_dtmc(&mut self) -> (f64, sprs::CsMat<f64>) {
		self.apply_queued_size();
		let state_count = self.state_count.max(self.data.len());
		let row_cnt = self.data.len();
		let mut rows = Vec::<usize>::with_capacity(state_count + row_cnt);
		let mut cols = Vec::<usize>::with_capacity(state_count + row_cnt);
		let mut values = Vec::<f64>::with_capacity(state_count + row_cnt);
		let mut epoch = f64::zero();
		for (row, col_option) in self.data.iter().enumerate() {
			let row_sum = self.row_sum(row);
			if let Some(col_data) = col_option {
				for (col, value) in col_data.iter() {
					rows.push(row);
					cols.push(*col);
					values.push(*value);
				}
			}
			// Add the negative diagonal entry
			if let Some(row_sum) = row_sum {
				rows.push(row);
				cols.push(row);
				values.push(row_sum);
				// Update the epoch
				if row == 0 || epoch > row_sum {
					epoch = row_sum;
				}
			}
		}
		(
			epoch,
			CsMat::new_csc((state_count, state_count), rows, cols, values),
		)
	}

	fn to_inf_matrix(&mut self) -> (f64, sprs::CsMat<f64>) {
		self.apply_queued_size();
		// We have to do it this way since the `Sub` trait isn't implemented for sparse matrices.
		let mut epoch = self.epoch();
		if epoch == 0.0 {
			epoch = 1.0;
		}
		let state_count = self.state_count;

		// TODO: remove this when we get direct CsMat construction to work
		let mut m = sprs::TriMat::new((state_count, state_count));
		for (row, col_option) in self.data.iter().enumerate() {
			let row_sum = self.row_sum(row);
			if let Some(col_data) = col_option {
				for (col, value) in col_data.iter() {
					m.add_triplet(row, *col, *value / epoch);
				}
			}
			// Add the negative diagonal entry
			if let Some(row_sum) = row_sum {
				m.add_triplet(row, row, 1.0 - row_sum / epoch);
			}
		}
		(epoch, m.to_csc())
	}

	fn to_unif_matrix(&mut self) -> (f64, sprs::CsMat<f64>) {
		self.apply_queued_size();
		// Like in to_inf_matrix, we have to do it this way since `Sub` isn't implemented for
		// sprs::CsMat
		let mut epoch = self.epoch();
		if epoch == 0.0 {
			epoch = 1.0;
		}
		info!("Uniformizing matrix with rate {epoch}");
		let state_count = self.state_count;

		// TODO: Also remove this when direct CsMat construction works
		let mut m = sprs::TriMat::new((state_count, state_count));
		for (row, col_option) in self.data.iter().take(state_count).enumerate() {
			let exit_rate_opt = self.row_sum(row);
			let mut had_self_loop: bool = false;
			if let Some(col_data) = col_option {
				for (col, value) in col_data.iter() {
					if *col == row {
						if let Some(exit_rate) = exit_rate_opt {
							// If we've added a self-loop it looks like this.
							m.add_triplet(row, *col, (*value - exit_rate) / epoch + 1.0);
							had_self_loop = true;
						} else {
							error!("Could not get exit rate for state {row}");
							panic!("Could not get exit rate for state {row}");
						}
					} else {
						m.add_triplet(row, *col, *value / epoch);
					}
				}
			}
			if !had_self_loop {
				// If the state never had a self-loop we have to manually add the
				// remaining probability mass on the diagonal.
				if let Some(exit_rate) = exit_rate_opt {
					m.add_triplet(row, row, 1.0 - exit_rate / epoch);
				} else {
					m.add_triplet(row, row, 1.0);
				}
			}
		}
		(epoch, m.to_csc())
	}
}

type ExplicitSprsMatBuilder = sprs::TriMat<f64>;

impl SprsMatBuilder for ExplicitSprsMatBuilder {
	/// See documentation for `TriMat::add_triplet`
	fn insert(&mut self, row: usize, col: usize, entry: f64) {
		self.add_triplet(row, col, entry);
	}

	/// Only returns the first time the triplet was added
	fn get_value(&self, row: usize, col: usize) -> Option<f64> {
		let idxes = self.find_locations(row, col);
		if idxes.len() == 0 {
			None
		} else {
			let idx = idxes[0].0;
			Some(self.data()[idx])
		}
	}

	fn row_count(&self) -> usize {
		self.rows()
	}

	/// Highly unoptimized
	fn row_sum(&self, row: usize) -> Option<f64> {
		if row < self.rows() {
			let sm = (0..self.cols())
				.filter_map(|col| self.get_value(row, col))
				.sum();
			Some(sm)
		} else {
			None
		}
	}

	/// See documentation for `TriMat::to_csc()`
	fn to_sparse_matrix(&mut self) -> sprs::CsMat<f64> {
		self.to_csc()
	}

	fn to_inf_matrix(&mut self) -> (f64, sprs::CsMat<f64>) {
		let state_count = self.rows().max(self.cols());
		// Compute row sums and epoch (max row sum).
		let mut row_sums = vec![0.0f64; state_count];
		for i in 0..self.data().len() {
			let row = self.row_inds()[i];
			let val = self.data()[i];
			if row < state_count {
				row_sums[row] += val;
			}
		}
		let mut epoch = row_sums.iter().copied().fold(0.0f64, f64::max);
		if epoch == 0.0 {
			epoch = 1.0;
		}
		let mut m = sprs::TriMat::new((state_count, state_count));
		for i in 0..self.data().len() {
			let row = self.row_inds()[i];
			let col = self.col_inds()[i];
			let val = self.data()[i];
			m.add_triplet(row, col, val / epoch);
		}
		for row in 0..state_count {
			m.add_triplet(row, row, 1.0 - row_sums[row] / epoch);
		}
		(epoch, m.to_csc())
	}

	fn to_emb_dtmc(&mut self) -> (f64, sprs::CsMat<f64>) {
		unimplemented!();
	}

	fn to_unif_matrix(&mut self) -> (f64, sprs::CsMat<f64>) {
		let state_count = self.rows().max(self.cols());
		// Aggregate duplicate triplets per (row, col).
		let mut row_sums = vec![0.0f64; state_count];
		let mut entries: Vec<std::collections::BTreeMap<usize, f64>> =
			vec![std::collections::BTreeMap::new(); state_count];
		for i in 0..self.data().len() {
			let row = self.row_inds()[i];
			let col = self.col_inds()[i];
			let val = self.data()[i];
			if row < state_count && col < state_count {
				*entries[row].entry(col).or_insert(0.0) += val;
				row_sums[row] += val;
			}
		}
		let mut epoch = row_sums.iter().copied().fold(0.0f64, f64::max);
		if epoch == 0.0 {
			epoch = 1.0;
		}
		let mut m = sprs::TriMat::new((state_count, state_count));
		for row in 0..state_count {
			let mut had_self_loop = false;
			for (col, value) in &entries[row] {
				if *col == row {
					m.add_triplet(row, *col, (*value - row_sums[row]) / epoch + 1.0);
					had_self_loop = true;
				} else {
					m.add_triplet(row, *col, *value / epoch);
				}
			}
			if !had_self_loop {
				m.add_triplet(row, row, 1.0 - row_sums[row] / epoch);
			}
		}
		(epoch, m.to_csc())
	}
}

#[cfg(test)]
mod matrix_tests {
	use num::ToPrimitive;

	use super::*;

	#[test]
	fn length_test() {
		let mut mat_builder = OptimalSprsMatBuilder::new();
		let mut num_inserted: usize = 0;
		for (row, col) in (0..=150).zip(0..150) {
			let entry = (row + 1).to_f64().unwrap() / col.to_f64().unwrap();
			mat_builder.insert(row, col, entry);
			num_inserted += 1;
			assert_eq!(mat_builder.len(), num_inserted);
		}

		for (row, col) in (0..=150).zip(0..150) {
			let entry = (row + 1).to_f64().unwrap() / col.to_f64().unwrap();
			assert_eq!(mat_builder.get_value(row, col).unwrap(), entry);
		}
	}

	#[test]
	fn overwrite_test() {
		let mut mat_builder = OptimalSprsMatBuilder::new();
		mat_builder.insert(1, 6, 0.1);
		assert_eq!(mat_builder.len(), 1);
		assert_eq!(mat_builder.get_value(1, 6), Some(0.1));
		eprintln!("Testing the overwrite functionality. Length should remain the same");
		mat_builder.insert(1, 6, 0.2);
		assert_eq!(mat_builder.len(), 1);
		assert_eq!(mat_builder.get_value(1, 6), Some(0.2));
		assert_eq!(mat_builder.get_value(2, 6), None);
	}

	#[test]
	fn randomized_roundtrip() {
		let mut mb = OptimalSprsMatBuilder::new();
		// Deterministic pseudo-random entries
		let mut seed: u64 = 12345;
		use std::collections::BTreeMap;
		let mut final_vals: BTreeMap<(usize, usize), f64> = BTreeMap::new();
		for _ in 0..200 {
			seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
			let row = (seed >> 32) as usize % 50;
			seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
			let col = (seed >> 32) as usize % 50;
			let val = ((seed as f64) / (u64::MAX as f64)).abs() + 0.001;
			mb.insert(row, col, val);
			final_vals.insert((row, col), val);
		}
		for ((row, col), val) in final_vals {
			let got = mb.get_value(row, col).unwrap();
			assert!((got - val).abs() < 1e-12);
		}
	}

	#[test]
	fn row_count_matches_state_count() {
		let mut mb = OptimalSprsMatBuilder::new();
		mb.insert(3, 5, 1.0);
		mb.insert(1, 0, 2.0);
		assert_eq!(mb.row_count(), 6);
		assert_eq!(mb.state_count, 6);
	}

	#[test]
	fn out_of_order_test() {
		let mut mat_builder = OptimalSprsMatBuilder::new();
		let mut num_inserted: usize = 0;
		// Iterate in the reverse direction
		for (row, col) in (1..=150).rev().zip((1..=150).rev()) {
			let entry = (row + 1).to_f64().unwrap() / col.to_f64().unwrap();
			// eprintln!("{row},{col}:{entry}");
			mat_builder.insert(row, col, entry);
			num_inserted += 1;
			assert_eq!(mat_builder.len(), num_inserted);
		}
		// Check in the forward direction
		for (row, col) in (1..=150).zip(1..=150) {
			let entry = (row + 1).to_f64().unwrap() / col.to_f64().unwrap();
			// eprintln!("{row},{col}:{entry}");
			assert_eq!(mat_builder.get_value(row, col).unwrap(), entry);
		}
	}
}

#[cfg(test)]
mod unif_matrix_tests {
	use super::*;

	#[test]
	fn rows_sum_to_one() {
		let mut mb = OptimalSprsMatBuilder::new();
		mb.insert(1, 0, 1.0);
		let (epoch, m) = mb.to_unif_matrix();
		assert!((epoch - 1.0).abs() < 1e-12);
		for row in 0..m.rows() {
			let mut sum = 0.0;
			for col in 0..m.cols() {
				sum += m.get(row, col).copied().unwrap_or(0.0);
			}
			assert!((sum - 1.0).abs() < 1e-9, "row {} sum {}", row, sum);
		}
	}

	#[test]
	fn random_ctmc_rows_sum_to_one() {
		// Deterministic pseudo-random CTMC over 6 states.
		let mut seed: u64 = 987654321;
		let n = 6usize;
		let mut mb = OptimalSprsMatBuilder::new();
		for src in 1..n {
			// ensure each non-absorbing state has at least one transition
			let mut exit_rate = 0.0;
			for dst in 0..n {
				seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
				let r = (seed >> 32) as usize % 4;
				if r == 0 {
					continue;
				}
				let rate = (r as f64) * 0.5 + 0.1;
				mb.insert(src, dst, rate);
				exit_rate += rate;
			}
			if exit_rate == 0.0 {
				mb.insert(src, 0, 1.0);
			}
		}
		// absorbing state 0 gets a self loop via uniformization (no inserts).
		let (_epoch, m) = mb.to_unif_matrix();
		for row in 0..m.rows() {
			let mut sum = 0.0;
			for col in 0..m.cols() {
				sum += m.get(row, col).copied().unwrap_or(0.0);
			}
			assert!((sum - 1.0).abs() < 1e-9, "row {} sum {}", row, sum);
		}
	}
}
