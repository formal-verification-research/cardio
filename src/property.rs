use std::ops;

use crate::matrix::CheckableNumber;

/// A trait for anything we can throw in a .props, .csl, or .pctl file.
pub trait Property {
	fn parse(input: &str) -> Self;
	fn is_pctl(&self) -> bool;
}

/// An enum representing the possible time bounds that a CSL property can handle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval<ValueType>
where
	ValueType: CheckableNumber,
{
	/// A time bound of the form [0, T]
	TimeBoundedUpper(ValueType),
	/// A time bound of the form [T, T']
	TimeBoundWindow(ValueType, ValueType),
	/// A time bound of the form [T, oo] (T to infinity)
	TimeBoundedLower(ValueType),
	/// A bound in the number of steps. Because PCTL only supports an upper bound on the number of
	/// steps, the count in here is an upper bound, i.e., from [0, k] steps.
	StepBoundUpper(usize),
	/// The absence of a time bound (steady state)
	TimeUnbounded,
}

impl<ValueType> ToString for Interval<ValueType>
where
	ValueType: CheckableNumber,
{
	fn to_string(&self) -> String {
		match self {
			Self::TimeUnbounded => "".to_string(),
			Self::TimeBoundedUpper(ubound) => format!("[0, {ubound}]"),
			Self::StepBoundUpper(ubound) => format!("<= {ubound}]"),
			Self::TimeBoundedLower(lbound) => format!(">= {lbound}"),
			Self::TimeBoundWindow(lbound, ubound) => format!("[{lbound}, {ubound}]"),
		}
	}
}

/// The type of transient or steady state probability query.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProbabilityQueryType<ValueType>
where
	ValueType: CheckableNumber,
{
	/// A simple query that asks the probability
	SimpleQuery,
	/// A query that asks "is the probability less than p?"
	LessThan(ValueType),
	/// A query that asks "is the probability less than or equal to p?"
	LessThanEqual(ValueType),
	/// A query that asks "is the probability greater than p?"
	GreaterThan(ValueType),
	/// A query that asks "is the probability greater than or equal to p?"
	GreaterThanEqual(ValueType),
}

impl<ValueType> ToString for ProbabilityQueryType<ValueType>
where
	ValueType: CheckableNumber,
{
	fn to_string(&self) -> String {
		match self {
			Self::SimpleQuery => "=?".to_string(),
			Self::LessThan(p) => format!("< {p}"),
			Self::LessThanEqual(p) => format!("<= {p}"),
			Self::GreaterThan(p) => format!("> {p}"),
			Self::GreaterThanEqual(p) => format!(">= {p}"),
		}
	}
}

/// A CSL or PCTL state formula. PCTL excludes the `SteadyStateQuery` however, and tools should
/// panic or error if receiving a steady-state query.
#[derive(Clone, Debug, PartialEq)]
pub enum StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	/// Evaluates to `true` on all states
	True,
	/// An atomic proposition which evaluates on a particular state
	AtomicProposition(evalexpr::Node),
	/// A string label for a particular state, i.e., "Absorbing"
	StringLabel(String),
	/// The negation operator on another state formula
	Not(Box<StateFormula<ValueType>>),
	/// The conjunction of two state formulae
	Conjunction(Box<StateFormula<ValueType>>, Box<StateFormula<ValueType>>),
	/// A probability query on a path formula. I.e., "from this state, what is the probability that
	/// the next path formula holds?"
	TransientQuery(ProbabilityQueryType<ValueType>, Box<PathFormula<ValueType>>),
	/// A steady state formula. I.e., "from this state, what is the probability that in the steady
	/// state Phi holds?"
	SteadyStateQuery(
		ProbabilityQueryType<ValueType>,
		Box<StateFormula<ValueType>>,
	),
}

impl<ValueType> ops::Not for StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	type Output = Self;

	fn not(self) -> Self::Output {
		// Prevent the tree from getting too complex
		match self {
			Self::Not(state_formula) => (*state_formula).clone(),
			_ => Self::Not(Box::new(self.clone())),
		}
	}
}

impl<ValueType> ops::BitAnd for StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	type Output = Self;
	fn bitand(self, rhs: Self) -> Self::Output {
		Self::Conjunction(Box::new(self.clone()), Box::new(rhs.clone()))
	}
}

impl<ValueType> ops::BitOr for StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	type Output = Self;
	fn bitor(self, rhs: Self) -> Self::Output {
		// According to DeMorgan's law, A | B is equal to !(!A & !B)
		// Also, now we can use the other overloaded operators.
		!(!self & !rhs)
	}
}

impl<ValueType> ToString for StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	fn to_string(&self) -> String {
		match self {
			Self::True => "true".to_string(),
			Self::Not(subformula) => subformula.to_string(),
			Self::AtomicProposition(ap) => ap.to_string(),
			Self::StringLabel(label) => format!("\"{label}\""),
			Self::Conjunction(lhs, rhs) => {
				let lhs_str = lhs.to_string();
				let rhs_str = rhs.to_string();
				format!("({lhs_str}) & ({rhs_str})")
			}
			Self::TransientQuery(query_type, path_formula) => {
				let qt_str = query_type.to_string();
				let pf_str = path_formula.to_string();
				format!("P{qt_str} [ {pf_str} ]")
			}
			Self::SteadyStateQuery(query_type, state_formula) => {
				let qt_str = query_type.to_string();
				let sf_str = state_formula.to_string();
				format!("S{qt_str} [ {sf_str} ]")
			}
		}
	}
}

impl<ValueType> Property for StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	fn is_pctl(&self) -> bool {
		match self {
			Self::True | Self::StringLabel(_) | Self::AtomicProposition(_) => true,
			Self::Not(sf) => sf.is_pctl(),
			Self::Conjunction(lhs, rhs) => lhs.is_pctl() && rhs.is_pctl(),
			Self::TransientQuery(_, pf) => pf.is_pctl(),
			Self::SteadyStateQuery(_, _) => false,
		}
	}

	fn parse(input: &str) -> Self {
		unimplemented!();
	}
}

impl<ValueType> StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	/// Creates lower and upper bound properties, useful for STAMINA.
	pub fn create_bounds(&self) -> Option<(Self, Self)> {
		let abs = Self::StringLabel("Absorbing".to_string());
		match self {
			// If the state formula is just "true" we just return true.
			Self::True => Some((Self::True, Self::True)),
			// If the state formula only evaluates on the current state, we just have to make sure
			// we're not currently in the absorbing state for the lower bound, or allow it for the
			// upper bound.
			Self::AtomicProposition(_)
			| Self::StringLabel(_)
			| Self::Not(_)
			| Self::Conjunction(_, _) => {
				let lower_bound = self.clone() & !abs.clone();
				let upper_bound = self.clone() | abs.clone();
				Some((lower_bound, upper_bound))
			}
			// For a probability query, we have to modify the path formula.
			Self::TransientQuery(query_type, path_formula) => {
				let (lower_bound, upper_bound) = match &**path_formula {
					PathFormula::Next(phi) => {
						let phi_lower = *phi.clone() & !abs.clone();
						let phi_upper = *phi.clone() | abs.clone();
						(
							PathFormula::<ValueType>::Next(Box::new(phi_lower)),
							PathFormula::<ValueType>::Next(Box::new(phi_upper)),
						)
					}
					PathFormula::Until(phi, interval, psi) => {
						// For phi, the upper bound can just be the same, but the lower bound we
						// must require that we don't reach the absorbing state.
						let phi_lower = *phi.clone() & !abs.clone();
						let phi_upper = *phi.clone();
						// For psi, the lower bound requires us to end in psi but not the absorbing
						// state. The upper bound allows psi or the absorbing state.
						let psi_lower = *psi.clone() & !abs.clone();
						let psi_upper = *psi.clone() | abs.clone();
						(
							PathFormula::<ValueType>::Until(
								Box::new(phi_lower),
								*interval,
								Box::new(psi_lower),
							),
							PathFormula::<ValueType>::Until(
								Box::new(phi_upper),
								*interval,
								Box::new(psi_upper),
							),
						)
					}
				};
				Some((
					Self::TransientQuery(*query_type, Box::new(lower_bound)),
					Self::TransientQuery(*query_type, Box::new(upper_bound)),
				))
			}
			// Steady state is just & !absorbing for the lower, and | absorbing for the upper bound
			Self::SteadyStateQuery(query_type, state_formula) => {
				let lower_bound = *state_formula.clone() & !abs.clone();
				let upper_bound = *state_formula.clone() | abs.clone();
				Some((
					Self::SteadyStateQuery(*query_type, Box::new(lower_bound)),
					Self::SteadyStateQuery(*query_type, Box::new(upper_bound)),
				))
			}
			// Currently we cover all options, but this allows us to easily add some.
			_ => None,
		}
	}
}

/// A CSL or PCTL path formula. In PCTL the interval type must be restricted to upper bounded
/// intervals with integer bounds.
#[derive(Clone, Debug, PartialEq)]
pub enum PathFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	/// If the state formula holds in the next state.
	Next(Box<StateFormula<ValueType>>),
	/// An "until" path formula over an interval
	Until(
		Box<StateFormula<ValueType>>,
		Interval<ValueType>,
		Box<StateFormula<ValueType>>,
	),
}

impl<ValueType> Property for PathFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	fn is_pctl(&self) -> bool {
		match self {
			Self::Next(sf) => sf.is_pctl(),
			Self::Until(phi, interval, psi) => match interval {
				Interval::StepBoundUpper(_bound) => phi.is_pctl() && psi.is_pctl(),
				_ => false,
			},
		}
	}

	fn parse(input: &str) -> Self {
		unimplemented!();
	}
}

impl<ValueType> ToString for PathFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	fn to_string(&self) -> String {
		match self {
			Self::Next(sf) => {
				let sf_str = sf.to_string();
				format!("X {sf_str}")
			}
			Self::Until(phi, interval, psi) => {
				let phi_str = phi.to_string();
				let i_str = interval.to_string();
				let psi_str = psi.to_string();
				format!("{phi_str} U{i_str} {psi_str}")
			}
		}
	}
}
