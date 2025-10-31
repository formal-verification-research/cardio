use crate::matrix::*;

/// An abstract representation of a rewards structure for a stochastic/Markovian model. By
/// "abstract", we mean that the rewards structures are stored as abstract functions which have not
/// yet been resolved and therefore cannot be used in CSL or PCTL model checking.
pub struct AbstractRewards<StateType, ValType>
where
	StateType: PartialEq,
	ValType: num::Num + Clone,
{
	name: String,
	state_rewards: Box<dyn Fn(&StateType) -> Option<ValType>>,
	transition_rewards: Box<dyn Fn(&StateType, &StateType) -> Option<ValType>>,
	state_rewards_cnt_hint: Option<usize>,
}

impl<StateType, ValType> AbstractRewards<StateType, ValType>
where
	StateType: PartialEq,
	ValType: num::Num + Clone,
{
	/// Creates an explicit rewards structure, given an iterator over states, and a function that
	/// converts a state to a state index (of type `usize`).
	pub fn to_explicit<StateIterator>(
		&self,
		states: StateIterator,
		state_to_id: &dyn Fn(StateType) -> usize,
	) -> ExplicitRewards<ValType>
	where
		StateIterator: ExactSizeIterator<Item = StateType> + Clone,
	{
		let state_rewards = &self.state_rewards;
		let tran_rewards = &self.transition_rewards;

		// Total number of states (may or may not have rewards)
		let state_count = states.len();

		// First, create the state rewards. This one we will do explicitly.
		
		// Reserve vectors with capacity based on the size hint. If the size hint doesn't exist, we
		// will use the state count since at *most* these vectors are that size. However, depending
		// on the rewards structure, the user may be able to know a better hint.
		let size_hint = self.state_rewards_cnt_hint.unwrap_or(state_count);
		let mut state_rewards_idxes = Vec::<usize>::with_capacity(size_hint);
		let mut state_rewards_values = Vec::<ValType>::with_capacity(size_hint);

		// The number of states with rewards
		let mut rew_st_count: usize = 0;
		// Clone the iterator so we can re-use it.
		for state in &mut states.clone() {
			if let Some(reward_val) = state_rewards(&state) {
				let idx = state_to_id(state);
				state_rewards_idxes.push(idx);
				state_rewards_values.push(reward_val);
				rew_st_count += 1;
			}
		}

		// Next, create transition rewards. We will use the optimal sparse matrix builder class
		// defined in this module.

		// Transition rewards matrix builder
		let mut trew_matb = OptimalSprsMatBuilder::with_row_capacity(state_count, Some(15));
		for (source, dest) in states.clone().zip(states) {
			if let Some(tran_reward) = tran_rewards(&source, &dest) {
				let row = state_to_id(source);
				let col = state_to_id(dest);

				trew_matb.insert(row, col, tran_reward);
			}
		}

		// Now put it all together
		ExplicitRewards::<ValType>::from_raw(
			// Obviously, just keep the same name
			self.name.clone(),
			// Explicitly construct the vector with the state rewards
			sprs::CsVec::new(rew_st_count, state_rewards_idxes, state_rewards_values),
			// Now construct the matrix with the transition rewards
			trew_matb.to_sparse_matrix(),
		)
	}
}

/// An explicit representation of a rewards structure for a stochastic/Markov model. By "explicit",
/// we mean that the rewards are stored in sparse matrix/vector form rather than unresolved
/// functions, as in `AbstractRewards`.
pub struct ExplicitRewards<ValType>
where
	ValType: num::Num + Clone,
{
	name: String,
	state_rewards: sprs::CsVec<ValType>,
	transition_rewards: sprs::CsMat<ValType>,
}

impl<ValType> ExplicitRewards<ValType>
where
	ValType: num::Num + Clone,
{
	/// Create explicit rewards from pre-built state reward vectors and transition rewards matrix.
	fn from_raw(
		name: String,
		state_rewards: sprs::CsVec<ValType>,
		transition_rewards: sprs::CsMat<ValType>,
	) -> Self {
		Self {
			name,
			state_rewards,
			transition_rewards,
		}
	}
}
