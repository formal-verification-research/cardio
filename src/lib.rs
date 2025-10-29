use vector_map::VecMap;


/// A trait that represents any type of sparse matrix construction.
pub trait SparseMatrixBuilder<EntryType>
where
	EntryType: num::Num + Clone,
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
        I: ExactSizeIterator<Item = (usize, EntryType)>;
	/// Creates the sparse matrix from the data.
	fn to_sparse_matrix(&self) -> sprs::CsMat<EntryType>;
}

/// A sparse matrix builder that allows for random access and update
pub struct RandomAccessSparseMatrixBuilder<EntryType>
where
    EntryType: num::Num + Clone, // All we require for the entry is a numeric type
{
    /// The actual data storage. We use a `VecMap` here since most models we've encountered have
    /// only 5-30 reactions. Therefore it will be more efficient than dealing with the overhead of
    /// a `HashMap` or a `BTreeMap` for every single element.
    data: Vec<Option<VecMap<usize, EntryType>>>,
    /// During state insertion, this will be set as the maximal y (destination) values from attempted
    /// insertions (x a.k.a., source must be resized immediately).
    queued_new_size: usize,
    /// The number of abstract transitions in the model serves as a hint for the amount of
    /// elements needed to pre-allocate in each row. In fact, it is an upper bound for it since
    /// the only way the number could be different is if a transition is not enabled at a state.
    abstract_transition_count: Option<usize>,
}

impl<EntryType> RandomAccessSparseMatrixBuilder<EntryType>
where
    EntryType: num::Num + Clone,
{	/// The number of nonzero entries in the sparse matrix
	fn num_entries(&self) -> usize {
		// Sum the counts of elements for any non-empty row
		self.data.iter().map(|row_opt| if let Some(row_val) = row_opt { row_val.len() } else { 0 }).sum()
	}
    /// Creates a new RandomAccessSparseMatrix for CTMC model checking
    pub fn new() -> Self {
        // by default just allocate with a capacity of 5000
        Self::with_row_capacity(5000, None)
    }

    /// Creates a new RandomAccessSparseMatrix for CTMC model checking with a known capacity
    pub fn with_row_capacity(capacity: usize, abstract_transition_count: Option<usize>) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            queued_new_size: capacity,
            abstract_transition_count,
        }
    }

    /// Resizes the underlying datastructure by using Rust's efficient `resize_with`. This function
    /// does NOT shrink the datastructure--only increasing the size if necessary.
    pub fn resize(&mut self, size: usize) {
        if size > self.data.len() {
            self.data.resize_with(size, Default::default)
        }
    }

    /// Applies the queued new size to the datastructure size
    pub fn apply_queued_size(&mut self) {
        self.resize(self.queued_new_size);
    }
}

impl<EntryType> SparseMatrixBuilder<EntryType> for RandomAccessSparseMatrixBuilder<EntryType>
where
    EntryType: num::Num + Clone,
{
    /// Gets the value at a particular row and column
    fn get_value(&self, row: usize, col: usize) -> Option<EntryType> {
        if row >= self.data.len() {
            None
        } else if let Some(row_val) = &self.data[row] {
            if let Some(element_val) = row_val.get(&col) {
                Some(element_val.clone())
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
        row_values.insert(col, entry);
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
            row_values.insert(col, entry);
        }
    }

	fn to_sparse_matrix(&self) -> sprs::CsMat<EntryType> {
		// TODO: the direct CSR constructor is more efficient than this. Maybe there's a way to
		// use that rather than the triplet matrix.
		let mut triplet_matrix = sprs::TriMatI::new((self.data.len(), self.data.len()));
		for (row_idx, column_opt) in self.data.iter().enumerate() {
			if let Some(colum_val) = column_opt {
				for (col_idx, value) in colum_val.iter() {
					triplet_matrix.add_triplet(row_idx, *col_idx, value.clone());
				}
			}
		}
		triplet_matrix.to_csr()
	}
}

#[cfg(test)]
mod tests {
    use super::*;
}
