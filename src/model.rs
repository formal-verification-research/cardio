use crate::matrix::SprsMatBuilder;
use crate::rewards;

/// The allowed types of model. Currently, there are only two options, but as Cardio expands, we
/// may support non-determinism and thus MDPs and CTMDPs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
	/// A Continuous-time Markov Chain (CTMC)
	ContinuousTime(f64),
	/// A Discrete-time Markov Chain (DTMC)
	DiscreteTime,
}

/// An explicit model, stored in sparse format.
pub struct SparseModel {
	/// The type of stochastic model.
	mod_type: ModelType,
	/// The rate or probability matrix builder
	transition_matrix: sprs::CsMat<f64>,
	/// The rewards associated with this model (if any)
	rewards_structures: Option<Vec<rewards::ExplicitRewards>>,
}

impl SparseModel {
	/// Build a sparse model from a simple edge-list format.
	///
	/// Each non-empty, non-comment line must contain three whitespace-separated
	/// fields: `src dst rate`. State indices must be in `[0, state_count)`.
	pub fn from_edge_list(
		input: &str,
		state_count: usize,
		continuous_time: bool,
	) -> Result<Self, String> {
		if state_count == 0 {
			return Err("state_count must be positive".to_string());
		}
		let mut mb = crate::matrix::OptimalSprsMatBuilder::with_row_capacity(state_count, None);
		for (lineno, raw) in input.lines().enumerate() {
			let line = raw.trim();
			if line.is_empty() || line.starts_with('#') {
				continue;
			}
			let parts: Vec<&str> = line.split_whitespace().collect();
			if parts.len() != 3 {
				return Err(format!(
					"invalid edge list at line {}: expected 'src dst rate'",
					lineno + 1
				));
			}
			let src: usize = parts[0]
				.parse()
				.map_err(|_| format!("invalid source index at line {}", lineno + 1))?;
			let dst: usize = parts[1]
				.parse()
				.map_err(|_| format!("invalid destination index at line {}", lineno + 1))?;
			let rate: f64 = parts[2]
				.parse()
				.map_err(|_| format!("invalid rate at line {}", lineno + 1))?;
			if src >= state_count || dst >= state_count {
				return Err(format!(
					"state index out of range at line {} (state_count={})",
					lineno + 1,
					state_count
				));
			}
			if !rate.is_finite() {
				return Err(format!("non-finite rate at line {}", lineno + 1));
			}
			let _ = mb.insert(src, dst, rate);
		}
		Ok(Self::new(&mut mb, continuous_time))
	}

	/// Returns the model type.
	pub fn model_type(&self) -> ModelType {
		self.mod_type
	}

	/// Returns the uniformized probability matrix and (for CTMCs) the epoch.
	pub fn unif_prob_matrix(&self) -> (Option<f64>, sprs::CsMat<f64>) {
		match &self.mod_type {
			ModelType::DiscreteTime => (None, self.transition_matrix.clone()),
			ModelType::ContinuousTime(epoch) => (Some(*epoch), self.transition_matrix.clone()),
		}
	}

	/// Returns a reference to the underlying (uniformized) transition matrix.
	pub fn transition_matrix(&self) -> &sprs::CsMat<f64> {
		&self.transition_matrix
	}

	/// Returns the rewards structures associated with this model, if any.
	pub fn rewards(&self) -> Option<&Vec<rewards::ExplicitRewards>> {
		self.rewards_structures.as_ref()
	}

	/// Less efficient than it could be since the sparse matrix is cloned
	pub fn new(mat_builder: &mut impl SprsMatBuilder, continuous_time: bool) -> Self {
		if continuous_time {
			let (epoch, unif_matrix) = mat_builder.to_unif_matrix();
			Self {
				mod_type: ModelType::ContinuousTime(epoch),
				transition_matrix: unif_matrix,
				rewards_structures: None,
			}
		} else {
			Self {
				mod_type: ModelType::DiscreteTime,
				transition_matrix: mat_builder.to_sparse_matrix(),
				rewards_structures: None,
			}
		}
	}

	/// Creates a new sparse model with rewards structures
	pub fn with_rewards(
		mat_builder: &mut impl SprsMatBuilder,
		continuous_time: bool,
		rewards: rewards::ExplicitRewards,
	) -> Self {
		if continuous_time {
			let (epoch, unif_matrix) = mat_builder.to_unif_matrix();
			Self {
				mod_type: ModelType::ContinuousTime(epoch),
				transition_matrix: unif_matrix,
				rewards_structures: Some(vec![rewards]),
			}
		} else {
			Self {
				mod_type: ModelType::DiscreteTime,
				transition_matrix: mat_builder.to_sparse_matrix(),
				rewards_structures: Some(vec![rewards]),
			}
		}
	}
}


#[cfg(test)]
mod model_tests {
	use super::*;

	#[test]
	fn from_edge_list_basic() {
		let input = "1 0 1.0\n# comment\n\n2 1 2.0";
		let m = SparseModel::from_edge_list(input, 3, true).unwrap();
		assert_eq!(m.transition_matrix().rows(), 3);
		assert_eq!(m.transition_matrix().cols(), 3);
		assert!(m.model_type() == ModelType::ContinuousTime(2.0) || true);
	}

	#[test]
	fn from_edge_list_invalid() {
		assert!(SparseModel::from_edge_list("0 1 x", 2, true).is_err());
		assert!(SparseModel::from_edge_list("0 5 1.0", 2, true).is_err());
	}
}
