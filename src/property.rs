use std::ops;

use crate::matrix::CheckableNumber;
use crate::parser::{lex, parse_path_formula, parse_state_formula};

/// A trait for anything we can throw in a .props, .csl, or .pctl file.
pub trait Property {
	fn parse(input: &str) -> Result<Self, String>
	where
		Self: Sized;
	fn is_pctl(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryApOperator {
	Plus,
	Minus,
	Multipy,
	// We don't support divide since this is over integers
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
	Eq,
	GreaterEq,
	LessEq,
	Greater,
	Less,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VariableType {
	BitIndecies(usize, usize),
	Name(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomicExpression {
	Value(i64),
	Variable(VariableType),
	BinaryOperator(
		BinaryApOperator,
		Box<AtomicExpression>,
		Box<AtomicExpression>,
	),
}

impl AtomicExpression {
	pub fn evaluate_value(&self) -> Option<i64> {
		match self {
			Self::Value(val) => Some(*val),
			Self::Variable(_) => None,
			Self::BinaryOperator(op, lhs, rhs) => {
				let lhs_val = lhs.evaluate_value()?;
				let rhs_val = rhs.evaluate_value()?;
				let res = match op {
					BinaryApOperator::Plus => lhs_val + rhs_val,
					BinaryApOperator::Minus => lhs_val + rhs_val,
					BinaryApOperator::Multipy => lhs_val * rhs_val,
				};
				Some(res)
			}
		}
	}

	pub fn is_convex(&self) -> bool {
		match self {
			Self::Value(_) | Self::Variable(_) => true,
			Self::BinaryOperator(op, lhs, rhs) => match op {
				BinaryApOperator::Plus | BinaryApOperator::Minus => {
					lhs.is_convex() & rhs.is_convex()
				}
				_ => false,
			},
		}
	}

	pub fn simplify(&self) -> Self {
		match self {
			// If a variable or a value, we can't simplify
			Self::Value(_) | Self::Variable(_) => self.clone(),
			// Here we can try to simplify it
			Self::BinaryOperator(op, lhs, rhs) => {
				unimplemented!()
			}
		}
	}
}

impl ToString for AtomicExpression {
	fn to_string(&self) -> String {
		match self {
			Self::Value(val) => format!("{val}"),
			Self::Variable(var_type) => match var_type {
				VariableType::Name(name) => name.clone(),
				VariableType::BitIndecies(lower, upper) => format!("bv[{lower}, {upper}]"),
			},
			Self::BinaryOperator(op, lhs, rhs) => {
				let lhs_str = lhs.to_string();
				let rhs_str = rhs.to_string();
				match op {
					BinaryApOperator::Plus => format!("({lhs_str} + {rhs_str})"),
					BinaryApOperator::Minus => format!("({lhs_str} - {rhs_str})"),
					BinaryApOperator::Multipy => format!("{lhs_str} * {rhs_str}"),
				}
			}
		}
	}
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomicProposition {
	lhs: Box<AtomicExpression>,
	operator: ComparisonOperator,
	rhs: Box<AtomicExpression>,
}

impl Property for AtomicProposition {
	fn is_pctl(&self) -> bool {
		// All atomic propositions on states can be checked in CSL
		true
	}

	fn parse(input: &str) -> Result<Self, String>
	where
		Self: Sized,
	{
		let tokens = lex(input);
		unimplemented!();
	}
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
	/// Evaluates to `false` on all states
	False,
	/// An atomic proposition which evaluates on a particular state
	AtomicProposition(evalexpr::Node),
	/// A string label for a particular state, i.e., "Absorbing"
	StringLabel(String),
	/// The negation operator on another state formula
	Not(Box<StateFormula<ValueType>>),
	/// The conjunction of two state formulae
	Conjunction(Box<StateFormula<ValueType>>, Box<StateFormula<ValueType>>),
	/// The Disjunction of two state formulae
	Disjunction(Box<StateFormula<ValueType>>, Box<StateFormula<ValueType>>),
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
			// True and false can just be inverted
			Self::True => Self::False,
			Self::False => Self::True,
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
		Self::Disjunction(Box::new(self.clone()), Box::new(rhs.clone()))
	}
}

impl<ValueType> ToString for StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	fn to_string(&self) -> String {
		match self {
			Self::True => "true".to_string(),
			Self::False => "false".to_string(),
			Self::Not(subformula) => subformula.to_string(),
			Self::AtomicProposition(ap) => ap.to_string(),
			Self::StringLabel(label) => format!("\"{label}\""),
			Self::Conjunction(lhs, rhs) => {
				let lhs_str = lhs.to_string();
				let rhs_str = rhs.to_string();
				format!("({lhs_str}) & ({rhs_str})")
			}
			Self::Disjunction(lhs, rhs) => {
				let lhs_str = lhs.to_string();
				let rhs_str = rhs.to_string();
				format!("({lhs_str}) | ({rhs_str})")
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
			Self::True | Self::False | Self::StringLabel(_) | Self::AtomicProposition(_) => true,
			Self::Not(sf) => sf.is_pctl(),
			Self::Conjunction(lhs, rhs) | Self::Disjunction(lhs, rhs) => {
				lhs.is_pctl() && rhs.is_pctl()
			}
			Self::TransientQuery(_, pf) => pf.is_pctl(),
			Self::SteadyStateQuery(_, _) => false,
		}
	}

	fn parse(input: &str) -> Result<Self, String> {
		let tokens = lex(input);
		parse_state_formula(&mut tokens.iter().peekable())
	}
}

impl<ValueType> StateFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	/// Creates lower and upper bound properties, useful for STAMINA.
	pub fn create_bounds(&self) -> Option<(Self, Self)> {
		let abs = Self::absorbing();
		match self {
			// If the state formula is just "true" we just return true.
			Self::True => Some((Self::True, Self::True)),
			// If the state formula is just "false" we just return false.
			Self::False => Some((Self::False, Self::False)),
			// If the state formula only evaluates on the current state, we just have to make sure
			// we're not currently in the absorbing state for the lower bound, or allow it for the
			// upper bound.
			Self::AtomicProposition(_)
			| Self::StringLabel(_)
			| Self::Not(_)
			| Self::Conjunction(_, _)
			| Self::Disjunction(_, _) => {
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
							PathFormula::<ValueType>::next(&phi_lower),
							PathFormula::<ValueType>::next(&phi_upper),
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
							PathFormula::<ValueType>::until(&phi_lower, *interval, &psi_lower),
							PathFormula::<ValueType>::until(&phi_upper, *interval, &psi_upper),
						)
					}
					PathFormula::Globally(phi) => {
						// This works the same as next
						let phi_lower = *phi.clone() & !abs.clone();
						let phi_upper = *phi.clone() | abs.clone();
						(
							PathFormula::<ValueType>::globally(&phi_lower),
							PathFormula::<ValueType>::globally(&phi_upper),
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
			} // Currently we cover all options, but this allows us to easily add some.
			  // _ => None,
		}
	}

	pub fn absorbing() -> Self {
		Self::StringLabel("absorbing".to_string())
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
	/// A state formula holds on an entire path. This is equivalent to Phi U False
	Globally(Box<StateFormula<ValueType>>),
}

impl<ValueType> Property for PathFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	fn is_pctl(&self) -> bool {
		match self {
			Self::Next(sf) | Self::Globally(sf) => sf.is_pctl(),
			Self::Until(phi, interval, psi) => match interval {
				Interval::StepBoundUpper(_bound) => phi.is_pctl() && psi.is_pctl(),
				_ => false,
			},
		}
	}

	fn parse(input: &str) -> Result<Self, String> {
		let tokens = lex(input);
		parse_path_formula(&mut tokens.iter().peekable())
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
			Self::Globally(sf) => {
				let sf_str = sf.to_string();
				format!("G {sf_str}")
			}
		}
	}
}

impl<ValueType> PathFormula<ValueType>
where
	ValueType: CheckableNumber,
{
	/// Creates a state formula of type `next`
	pub fn next(state_formula: &StateFormula<ValueType>) -> Self {
		Self::Next(Box::new(state_formula.clone()))
	}

	/// Creates a state formula of type `until`
	pub fn until(
		phi: &StateFormula<ValueType>,
		interval: Interval<ValueType>,
		psi: &StateFormula<ValueType>,
	) -> Self {
		Self::Until(Box::new(phi.clone()), interval, Box::new(psi.clone()))
	}

	pub fn globally(state_formula: &StateFormula<ValueType>) -> Self {
		Self::Globally(Box::new(state_formula.clone()))
	}

	/// Creates an "eventually" state formula, which is equivalently "true U psi"
	pub fn eventually(
		interval: Interval<ValueType>,
		state_formula: &StateFormula<ValueType>,
	) -> Self {
		Self::Until(
			Box::new(StateFormula::<ValueType>::True),
			interval,
			Box::new(state_formula.clone()),
		)
	}

	// TODO: Weak until and release.
}

#[cfg(test)]
mod property_tests {
	use super::{Interval, PathFormula, Property, StateFormula};

	#[test]
	fn construction_test() {
		let phi: StateFormula<f64> = StateFormula::absorbing();
		let neg_phi = !phi.clone();
		let eventually_abs: PathFormula<f64> =
			PathFormula::eventually(Interval::<_>::TimeUnbounded, &phi);
		let globally_not_abs: PathFormula<f64> = PathFormula::globally(&neg_phi);
		println!("Property 1: {}", phi.to_string());
		println!("Property 2: {}", neg_phi.to_string());
		println!("Property 3: {}", eventually_abs.to_string());
		println!("Property 4: {}", globally_not_abs.to_string());
	}

	#[test]
	fn negation_test() {
		// let phi: StateFormula<f64> = StateFormula::AtomicProposition(evalexpr::)
		unimplemented!();
	}

	#[test]
	fn parse_test() {
		let prop_result = StateFormula::<f64>::parse(&"P=? [true U \"absorbing\"]");
		assert!(prop_result.is_ok());
		println!("{}", prop_result.unwrap().to_string());
	}
}
