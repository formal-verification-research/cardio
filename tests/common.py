import cardio
import stormpy

import numpy as np
from collections import deque

class Transition:
	def __init__(self, update: np.matrix, needed: np.matrix | None, rate_fn):
		self.update = update
		self.needed = needed
		if needed is None:
			self.needed = np.zeros(self.update.size)
		self.rate_fn = rate_fn

	def get_update(self, cur_state: np.matrix):
		if (cur_state < self.needed).any() and (cur_state + self.update < 0).any():
			return None
		else:
			(cur_state + self.update, self.rate_fn(cur_state))

class Model:
	def __init__(self, init_state, transitions, var_bound, sat_predicate):
		self.init_state = init_state
		self.transitions = transitions
		self.var_bound = var_bound
		self.sat_predicate = sat_predicate

	def next_states(self, state):
		updates = [transition.get_update(state) for transition in self.transitions]
		return [update for update in updates if update is not None and (update[0] > self.var_bound).all()]

	def check_cardio_and_storm(self, time_bound):
		next_available_index = 2 # absorbing is 0 init is 1
		queue = deque([self.init_state])
		init_tuple = tuple(self.init_state.T.tolist()[0])
		state_to_id = {init_tuple:1}
		# Create matrices for cardio and storm
		cardio_rf = cardio.QuantitativeReachabilityFinder()
		stormpy_mat = stormpy.SparseMatrixBuilder()

		sat_indecies = []

		while len(queue) > 0:
			cur_state = queue.popleft()
			cur_state_tuple = tuple(cur_state.T.tolist()[0])
			cur_idx = state_to_id[tuple(cur_state_tuple)]
			updates = self.next_states(cur_state)
			for next_state, rate in updates:
				next_idx = -1
				if tuple(next_state) in state_to_id:
					next_idx = state_to_id[tuple(next_state)]
				else:
					# State is new
					state_to_id[tuple(next_state)] = next_available_index
					next_idx = next_available_index
					# Check if satisfying
					if self.sat_predicate(next_state):
						cardio_rf.set_state_satisfying(next_idx)
						sat_indecies.append(next_idx)
				# Add to both matrices
				rf.insert(cur_idx, next_idx, rate)
				mat.add_next_value(cur_idx, next_idx, rate)

		print("Finished building model.")
		# Model check for cardio
		print("Checking model with cardio")
		lower_bound, upper_bound = cardio_rf.build_matrix_and_get_bounds(time_bound)
		print(f"Cardio returned bound {lower_bound}, {upper_bound}")
		print("Checking model with storm")
		# We have to build labeling for storm
		stormpy_labels = stormpy.StateLabeling()
		stormpy_labels.add_label_to_state("absorbing", 0)
		for idx in sat_indecies:
			stormpy_labels.add_label_to_state("satisfying", next_idx)
		storm_ctmc = stormpy.SparseModelCtmc(transition_matrix=stormpy_mat.build(), state_labeling=stormpy_labels, rate_transitions=True)
		props_strs = [ f"P=? [ true U{time_bound} \"satisfying\" ]", f"P=? [ true U{time_bound} \"satisfying\" | \"absorbing\" ]" ]
		lprop, rprop = stormpy.parse_properties(props_strs)
		lresult = stormpy.check_model_sparse(model, lprop, only_initial_states=True)
		pmin = lresult.at(1)
		rresult = stormpy.check_model_sparse(model, rprop, only_initial_states=True)
		pmax = rresult.at(1)
		print(f"Storm returned {lower_bound},{upper_bound}")
