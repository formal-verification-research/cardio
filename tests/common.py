import cardio
import stormpy

import numpy as np
from collections import deque


class Transition:
	def __init__(self, update: np.matrix, needed: np.matrix | None, rate_fn):
		self.update = update
		self.needed = needed
		if needed is None:
			self.needed = np.zeros(self.update.shape)
		self.rate_fn = rate_fn

	def get_update(self, cur_state: np.matrix):
		doesnt_have_needed = (cur_state < self.needed).T.any()
		leads_outside_first_orthant = ((cur_state + self.update).T < 0).any()
		if doesnt_have_needed or leads_outside_first_orthant:
			return None
		else:
			return (cur_state + self.update, self.rate_fn(cur_state))


class Model:
	def __init__(self, init_state, transitions, var_bound, sat_predicate):
		self.init_state = init_state
		self.transitions = transitions
		self.var_bound = var_bound
		self.sat_predicate = sat_predicate

	def next_states(self, state):
		updates = [transition.get_update(state) for transition in self.transitions]
		# print(updates)
		valid_updates = [update for update in updates if update is not None]
		return valid_updates

	def check_cardio_and_storm(self, time_bound, bypass_cardio=False, bypass_storm=False):
		next_available_index = 2  # absorbing is 0 init is 1
		queue = deque([self.init_state])
		init_tuple = tuple(self.init_state.T.tolist()[0])
		state_to_id = {init_tuple: 1}
		# Create matrices for cardio and storm
		cardio_rf = cardio.QuantitativeReachabilityFinder()
		stormpy_mat = stormpy.SparseMatrixBuilder()

		sat_indecies = []

		state_count = 2  # Initial and absorbing state

		print("Building model to check with both Cardio and Stormpy")

		while len(queue) > 0:
			cur_state = queue.popleft()
			cur_state_tuple = tuple(cur_state.T.tolist()[0])
			# print(f"Dequeued state {cur_state_tuple}")
			cur_idx = state_to_id[tuple(cur_state_tuple)]
			print(f"\rCurrently exploring state with id {cur_idx}", end="", flush=True)
			updates = self.next_states(cur_state)
			# print("updates:", ' '.join([f"{update[0].T}, {rate}" for update, rate in updates]))
			absorbing_rate = 0
			for next_state, rate in updates:
				next_tuple = tuple(next_state.T.tolist()[0])
				next_idx = -1
				# If the state is outside the variable bound, just connect it to the absorbing state
				if (next_state >= self.var_bound).any():
					absorbing_rate += rate
					continue
				elif next_tuple in state_to_id:
					next_idx = state_to_id[next_tuple]
				else:
					# State is new
					state_to_id[next_tuple] = next_available_index
					next_idx = next_available_index
					state_count += 1
					# Check if satisfying
					if self.sat_predicate(next_state):
						cardio_rf.set_state_satisfying(next_idx)
						sat_indecies.append(next_idx)
					else:
						# only explore successors if not satisfying
						queue.append(next_state)
					# Update next available index
					next_available_index += 1
				# Add to both matrices
				# print(cur_idx, next_idx, rate)
				cardio_rf.insert(cur_idx, next_idx, rate)
				stormpy_mat.add_next_value(cur_idx, next_idx, rate)
			# Insert transition to absorbing state
			cardio_rf.insert(cur_idx, 0, absorbing_rate)
			stormpy_mat.add_next_value(cur_idx, 0, absorbing_rate)

		print(f"\rFinished building model with state count {state_count}")
		# Model check for cardio
		if not bypass_cardio:
			print("Checking model with cardio")
			lower_bound, upper_bound = cardio_rf.build_matrix_and_get_bounds(time_bound)
			print(f"Cardio returned bound {lower_bound}, {upper_bound}")
		print("Checking model with storm")
		# We have to build labeling for storm
		if not bypass_storm:
			stormpy_labels = stormpy.StateLabeling(state_count)
			stormpy_labels.add_label("absorbing")
			stormpy_labels.add_label("satisfying")
			stormpy_labels.add_label("init")
			stormpy_labels.add_label_to_state("absorbing", 0)
			stormpy_labels.add_label_to_state("init", 1)
			for idx in sat_indecies:
				stormpy_labels.add_label_to_state("satisfying", next_idx)
			m = stormpy_mat.build()
			print(m.nr_rows)
			components = stormpy.SparseModelComponents(
				m, stormpy_labels, {}, rate_transitions=True)
			storm_ctmc = stormpy.SparseCtmc(components)
			props_strs = [f"P=? [ true U[0, {time_bound}] \"satisfying\" ]",
                            f"P=? [ true U[0, {time_bound}] \"satisfying\" | \"absorbing\" ]"]
			lprop = stormpy.parse_properties(props_strs[0])[0]
			rprop = stormpy.parse_properties(props_strs[1])[0]
			lresult = stormpy.check_model_sparse(storm_ctmc, lprop, only_initial_states=True)
			pmin = lresult.at(1)
			rresult = stormpy.check_model_sparse(storm_ctmc, rprop, only_initial_states=True)
			pmax = rresult.at(1)
			print(f"Storm returned {pmin},{pmax}")
