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

