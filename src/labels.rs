use bitvec::prelude::*;

#[derive(Clone, Debug)]
pub struct Labels {
	/// Labels are stored as strings in a vector, with the index in the array representing the
	/// label's index in the bitvec for the labelling of each state. The number stored with the
	/// labels is the number of states which the label has.
	label_names: Vec<(String, usize)>,
	/// The state labelling is an option since after we've constructed it, no more labels can be
	/// added, otherwise we'd have to re-initialize every bitvec in the state labeling.
	state_labelling: Option<Vec<BitVec>>,
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

	/// Creates a labelling structure with both an absorbing label, and a satisfying label. This is
	/// useful for manual matrix construction where cardio is not aware of state values.
	pub fn with_abs_and_sat() -> Self {
		let mut labelling = Self {
			label_names: vec![("absorbing".to_string(), 0), ("satisfying".to_string(), 1)],
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
				BitVec::with_capacity(label_count),
			);
			// Update the count of labels
			if labelling[state_index].len() >= label_index || labelling[state_index][label_index] {
				self.label_names[label_index].1 += 1;
			} else {
				labelling[state_index].insert(label_index, true)
			}
		} else {
			unimplemented!(); // Should we panic or create the state labelling?
		}
	}

	/// Gets whether or not a state has a particular label. If the label index exceeds the
	/// number of labels in the structure, then this function panics. However, the labelling is not
	/// conscious of the number of states, so if the state index is higher than the number of
	/// existing states, then it just returns false.
	pub fn state_has_label(&self, state_index: usize, label_index: usize) -> bool {
		let label_count = self.label_count();
		assert!(label_index < label_count);
		if let Some(labelling) = &self.state_labelling {
			if state_index >= labelling.len() {
				false
			} else {
				*labelling[state_index].get(label_index).as_deref().unwrap()
			}
		} else {
			false
		}
	}

	/// Works like `state_has_label` except all of the label indecies flagged in the bitmask have
	/// to be true for the state at `state_index` in order for the function to return true.
	pub fn state_has_labels(&self, state_index: usize, label_bitmask: &BitVec) -> bool {
		let label_count = self.label_count();
		assert!(label_bitmask.len() < label_count);
		if let Some(labelling) = &self.state_labelling {
			if state_index >= labelling.len() {
				false
			} else {
				labelling[state_index].clone() & label_bitmask == label_bitmask.as_bitslice()
			}
		} else {
			false
		}
	}

	/// Creates a bitvector of relevant states given a label bitmask
	pub fn create_relevant(&self, label_bitmask: &BitVec, num_states: usize) -> BitVec {
		let sat_iterator = (0..=num_states).map(|idx| self.state_has_labels(idx, label_bitmask));
		BitVec::from_iter(sat_iterator)
	}

	/// Creates a bitmask from a set of label strings.
	pub fn create_label_bitmask(&self, label_strings: Vec<String>) -> BitVec {
		let label_iterator = self
			.label_names
			.iter()
			.map(|(label_name, _states_per_label)| label_strings.contains(label_name));
		BitVec::from_iter(label_iterator)
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
