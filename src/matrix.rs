use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num::{Rational32, Rational64, Zero, pow::Pow};
use sprs::CsMat;
use vector_map::VecMap;

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
{
}

impl CheckableNumber for f64 {}
impl CheckableNumber for f32 {}
impl CheckableNumber for Rational64 {}
impl CheckableNumber for Rational32 {}

/// A trait that represents any type of sparse matrix construction.
pub trait SprsMatBuilder<EntryType>
where
	EntryType: CheckableNumber,
{
	/// Get the value (if it exists) at row `row` and column `col`. If it does not exist, this
	/// function will return None.
	fn get_value(&self, row: usize, col: usize) -> Option<EntryType>;
	/// Inserts an element. Depending on the struct implementing this trait, this function may
	/// either replace existing values, or just panic if a value already exists.
	fn insert(&mut self, row: usize, col: usize, entry: EntryType);
	/// Inserts an entire row. Again, depending on the struct implementing the trait, this function
	/// may replace the entire row, or panic if the row is not none.
	fn insert_row<I>(&mut self, row: usize, elements: I)
	where
		I: ExactSizeIterator<Item = (usize, EntryType)>,
	{
		for (col, entry) in elements {
			self.insert(row, col, entry);
		}
	}
	/// The sum of a row.
	fn row_sum(&self, row: usize) -> Option<EntryType>;
	/// Creates the sparse matrix from the data.
	fn to_sparse_matrix(&self) -> sprs::CsMat<EntryType>;
	/// Creates an infantesimile generator matrix.
	fn to_inf_matrix(&self) -> (EntryType, sprs::CsMat<EntryType>);
	/// Creates a uniformized matrix, suitable for model checking.
	fn to_unif_matrix(&self) -> (EntryType, sprs::CsMat<EntryType>);
}

/// A sparse matrix builder that allows for random access and updating and is optimized for VAS and
/// CRNs. I.e., models that have only a few abstract transitions, but potentially large numbers of
/// states.
pub struct OptimalSprsMatBuilder<EntryType>
where
	EntryType: CheckableNumber, // All we require for the entry is a numeric type
{
	/// The actual data storage. We use a `VecMap` here since most models we've encountered have
	/// only 5-30 reactions. Therefore it will be more efficient than dealing with the overhead of
	/// a `HashMap` or a `BTreeMap` for every single element.
	data: Vec<Option<VecMap<usize, EntryType>>>,
	/// The number of elements (not the number of reserved elements) in the builder
	length: usize,
	/// During state insertion, this will be set as the maximal y (destination) values from attempted
	/// insertions (x a.k.a., source must be resized immediately).
	queued_new_size: usize,
	/// The number of abstract transitions in the model serves as a hint for the amount of
	/// elements needed to pre-allocate in each row. In fact, it is an upper bound for it since
	/// the only way the number could be different is if a transition is not enabled at a state.
	abstract_transition_count: Option<usize>,
}

impl<EntryType> OptimalSprsMatBuilder<EntryType>
where
	EntryType: CheckableNumber,
{
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
			queued_new_size: capacity,
			length: 0,
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
		self.resize(self.queued_new_size);
	}

	/// Gets the number of elements in the matrix
	pub fn len(&self) -> usize {
		self.length
	}

	/// Gets the epoch time, i.e., the maximal row sum.
	pub fn epoch(&self) -> EntryType {
		let rows = self.data.len();
		(0..rows)
			.filter_map(|row| self.row_sum(row))
			.reduce(|epoch, mid| if epoch < mid { mid } else { epoch })
			.unwrap_or(EntryType::zero())
	}
}

impl<EntryType> SprsMatBuilder<EntryType> for OptimalSprsMatBuilder<EntryType>
where
	EntryType: CheckableNumber,
{
	/// Gets the value at a particular row and column
	fn get_value(&self, row: usize, col: usize) -> Option<EntryType> {
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
	fn insert(&mut self, row: usize, col: usize, entry: EntryType) {
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
			eprintln!("Warning: overwrite");
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
		I: ExactSizeIterator<Item = (usize, EntryType)>,
	{
		// As before, resize.
		self.resize(row);
		// Create a new VecMap with the capacity of the length of `elements`
		self.data[row] = Some(VecMap::with_capacity(elements.len()));
		let row_values = &mut self.data[row].as_mut().unwrap();
		for (col, entry) in elements {
			if let Some(_old_value) = row_values.insert(col, entry) {
				eprintln!("Warning: overwrite");
			} else {
				self.length += 1;
			}
		}
	}

	/// Gets the sum of a row. Useful for uniformization in CTMCs.
	fn row_sum(&self, row: usize) -> Option<EntryType> {
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
	fn to_sparse_matrix(&self) -> sprs::CsMat<EntryType> {
		let state_count = self.len();
		let mut rows = Vec::<usize>::with_capacity(state_count);
		let mut cols = Vec::<usize>::with_capacity(state_count);
		let mut values = Vec::<EntryType>::with_capacity(state_count);
		for (row, col_option) in self.data.iter().enumerate() {
			if let Some(col_data) = col_option {
				for (col, value) in col_data.iter() {
					rows.push(row);
					cols.push(*col);
					values.push(value.neg());
				}
			}
		}
		CsMat::new_csc((state_count, state_count), rows, cols, values)
	}

	fn to_inf_matrix(&self) -> (EntryType, sprs::CsMat<EntryType>) {
		let state_count = self.len();
		let row_cnt = self.data.len();
		let mut rows = Vec::<usize>::with_capacity(state_count + row_cnt);
		let mut cols = Vec::<usize>::with_capacity(state_count + row_cnt);
		let mut values = Vec::<EntryType>::with_capacity(state_count + row_cnt);
		let mut epoch = EntryType::zero();
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

	fn to_unif_matrix(&self) -> (EntryType, sprs::CsMat<EntryType>) {
		// We have to do it this way since the `Sub` trait isn't implemented for sparse matrices.
		let one = EntryType::one();
		let epoch = self.epoch();
		let state_count = self.len();
		let row_cnt = self.data.len();
		let mut rows = Vec::<usize>::with_capacity(state_count + row_cnt);
		let mut cols = Vec::<usize>::with_capacity(state_count + row_cnt);
		let mut values = Vec::<EntryType>::with_capacity(state_count + row_cnt);
		for (row, col_option) in self.data.iter().enumerate() {
			let row_sum = self.row_sum(row);
			if let Some(col_data) = col_option {
				for (col, value) in col_data.iter() {
					rows.push(row);
					cols.push(*col);
					values.push(*value / epoch);
				}
			}
			// Add the negative diagonal entry
			if let Some(row_sum) = row_sum {
				rows.push(row);
				cols.push(row);
				values.push(one - row_sum / epoch);
			}
		}
		(
			epoch,
			CsMat::new_csc((state_count, state_count), rows, cols, values),
		)
	}
}

type ExplicitSprsMatBuilder<EntryType> = sprs::TriMat<EntryType>;

impl<EntryType> SprsMatBuilder<EntryType> for ExplicitSprsMatBuilder<EntryType>
where
	EntryType: CheckableNumber,
{
	/// See documentation for `TriMat::add_triplet`
	fn insert(&mut self, row: usize, col: usize, entry: EntryType) {
		self.add_triplet(row, col, entry);
	}

	/// Only returns the first time the triplet was added
	fn get_value(&self, row: usize, col: usize) -> Option<EntryType> {
		let idxes = self.find_locations(row, col);
		if idxes.len() == 0 {
			None
		} else {
			let idx = idxes[0].0;
			Some(self.data()[idx])
		}
	}

	/// Highly unoptimized
	fn row_sum(&self, row: usize) -> Option<EntryType> {
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
	fn to_sparse_matrix(&self) -> sprs::CsMat<EntryType> {
		self.to_csc()
	}

	fn to_unif_matrix(&self) -> (EntryType, sprs::CsMat<EntryType>) {
		unimplemented!();
	}

	fn to_inf_matrix(&self) -> (EntryType, sprs::CsMat<EntryType>) {
		unimplemented!();
	}
}

#[cfg(test)]
mod matrix_tests {
	use num::ToPrimitive;

	use super::*;

	#[test]
	fn length_test() {
		let mut mat_builder = OptimalSprsMatBuilder::<f64>::new();
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
		let mut mat_builder = OptimalSprsMatBuilder::<f64>::new();
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
	fn out_of_order_test() {
		let mut mat_builder = OptimalSprsMatBuilder::<f64>::new();
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
