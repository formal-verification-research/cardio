use crate::matrix::{MatEntry, SprsMatBuilder};
use crate::rewards;

/// The allowed types of model. Currently, there are only two options, but as Cardio expands, we
/// may support non-determinism and thus MDPs and CTMDPs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType<ValueType>
where
	ValueType: MatEntry,
{
	/// A Continuous-time Markov Chain (CTMC)
	ContinuousTime(ValueType),
	/// A Discrete-time Markov Chain (DTMC)
	DiscreteTime,
}

trait Model<ValueType>
where
	ValueType: MatEntry,
{
	type MatrixType;

	fn model_type(&self) -> ModelType<ValueType>;
	/// Returns the uniformized DTMC if a continuous time, along with the epoch size. If it's a
	/// DTMC, then the epoch will be `1.0` and the matrix returned will just be the probability
	/// matrix. The result of this function is intended to be used in the model check functions.
	fn unif_prob_matrix(&self) -> (Option<ValueType>, Self::MatrixType);
}

/// An explicit model, stored in sparse format.
pub struct SparseModel<ValueType>
where
	ValueType: MatEntry,
{
	/// The type of stochastic model.
	mod_type: ModelType<ValueType>,
	/// The rate or probability matrix builder
	transition_matrix: sprs::CsMat<ValueType>,
	/// The rewards associated with this model (if any)
	rewards_structures: Option<Vec<rewards::ExplicitRewards<ValueType>>>,
}

impl<ValueType> SparseModel<ValueType>
where
	ValueType: MatEntry,
{
	/// Less efficient than it could be since the sparse matrix is cloned
	pub fn new(mat_builder: &impl SprsMatBuilder<ValueType>, continuous_time: bool) -> Self {
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
		mat_builder: &impl SprsMatBuilder<ValueType>,
		continuous_time: bool,
		rewards: rewards::ExplicitRewards<ValueType>,
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

impl<ValueType> Model<ValueType> for SparseModel<ValueType>
where
	ValueType: MatEntry,
{
	type MatrixType = sprs::CsMat<ValueType>;

	fn model_type(&self) -> ModelType<ValueType> {
		self.mod_type.clone()
	}

	fn unif_prob_matrix(&self) -> (Option<ValueType>, Self::MatrixType) {
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
