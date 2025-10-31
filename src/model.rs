/// The allowed types of model. Currently, there are only two options, but as Cardio expands, we
/// may support non-determinism and thus MDPs and CTMDPs.
pub enum ModelType {
	/// A Continuous-time Markov Chain (CTMC)
	ContinuousTimeMC,
	/// A Discrete-time Markov Chain (DTMC)
	DiscreteTimeMC,
}

/// An explicit model, stored in sparse format.
pub struct SparseModel<ValueType>
where
	ValueType: num::Num + Copy,
{
	/// The type of stochastic model.
	model_type: ModelType,
	/// The rate or probability matrix
	transition_matrix: sprs::CsMat<ValueType>,
}
