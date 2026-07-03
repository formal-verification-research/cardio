use crate::matrix::{CheckableNumber, SprsMatBuilder};
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

trait Model {
	type MatrixType;

	fn model_type(&self) -> ModelType;
	/// Returns the uniformized DTMC if a continuous time, along with the epoch size. If it's a
	/// DTMC, then the epoch will be `1.0` and the matrix returned will just be the probability
	/// matrix. The result of this function is intended to be used in the model check functions.
	fn unif_prob_matrix(&self) -> (Option<f64>, Self::MatrixType);
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

impl Model for SparseModel {
	type MatrixType = sprs::CsMat<f64>;

	fn model_type(&self) -> ModelType {
		self.mod_type.clone()
	}

	fn unif_prob_matrix(&self) -> (Option<f64>, Self::MatrixType) {
		match &self.mod_type {
			ModelType::DiscreteTime => (None, self.transition_matrix.clone()),
			ModelType::ContinuousTime(epoch) => {
				// If it's a continuous-time matrix, we assume that the transition matrix is the
				// infantesimile generator matrix, Q.
				unimplemented!()
			}
		}
	}
}
