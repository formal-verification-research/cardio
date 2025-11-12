pub struct Labels {
	/// Labels are stored as strings in a vector, with the index in the array representing the
	/// label's index in the bitvector for the labelling of each state. The number stored with the
	/// labels is the number of states which the label has.
	label_names: Vec<(String, usize)>,
	/// The state labelling is an option since after we've constructed it, no more labels can be
	/// added, otherwise we'd have to re-initialize every bitvector in the state labeling.
	state_labelling: Option<Vec<bitvector::BitVector>>,
}

impl Default for Labels {
	/// Default label structure is not constructed, thus the `state_labelling` field is empty. You
	/// must first create the state labelling.
	fn default() -> Self {
		Self {
			label_names: Vec::new(),
			state_labelling: None,
		}
	}
}

impl Labels {
	/// Creates a labelling structure with an absorbing label already created and assigned to the
	/// state at index zero, assuming that the artificial absorbing state as used by e.g., STAMINA
	/// is stored at index zero.
	pub fn with_absorbing() -> Self {
		let mut labelling = Self {
			label_names: vec![("absorbing".to_string(), 0)],
			state_labelling: Some(Vec::new()),
		};
		labelling.add_label_to_state(0, 0);
		labelling
	}

	/// Returns the number of labels in the labelling
	pub fn label_count(&self) -> usize {
		self.label_names.len()
	}

	/// Gets the index for a label given its name. If the name does not exist, returns `None`.
	pub fn label_to_index(&self, label_name: &str) -> Option<usize> {
		for (lindex, label) in self.label_names.iter().enumerate() {
			if label.0 == label_name {
				return Some(lindex);
			}
		}
		None
	}

	/// Gets the name of a label given its index. If the index is out of range, returns `None`.
	pub fn index_to_label(&self, label_index: usize) -> Option<String> {
		if self.label_names.len() >= label_index {
			None
		} else {
			Some(self.label_names[label_index].0.clone())
		}
	}

	/// Adds a new label to the labelling structure and returns its index
	pub fn add_label(&mut self, label_name: &str) -> usize {
		if let Some(_) = &self.state_labelling {
			panic!("Cannot add new label after state labelling has already been constructed.");
		} else {
			self.label_names.push((label_name.to_string(), 0));
			self.label_names.len() - 1
		}
	}

	/// Adds a label to a state given the state's index.
	pub fn add_label_to_state(&mut self, label_index: usize, state_index: usize) {
		let label_count = self.label_count();
		assert!(label_index < label_count);
		if let Some(labelling) = &mut self.state_labelling {
			// Resize the state labelling to the state index we've seen.
			labelling.resize(
				labelling.len().max(state_index + 1),
				bitvector::BitVector::new(label_count),
			);
			// Update the count of labels
			if labelling[state_index].insert(label_index) {
				self.label_names[label_index].1 += 1;
			}
		} else {
			unimplemented!(); // Should we panic or create the state labelling?
		}
	}
}

impl ToString for Labels {
	fn to_string(&self) -> String {
		let label_count = self.label_count();
		if let Some(_state_labelling) = &self.state_labelling {
			let label_list = itertools::join(
				self.label_names
					.iter()
					.map(|(label_name, count)| format!("\"{label_name}\": {count} items")),
				"\n - ",
			);
			format!("State labelling with {label_count} labels:\n{label_list}")
		} else {
			format!("Uninitialized state labeling with {label_count} labels.")
		}
	}
}
