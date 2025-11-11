use std::ops;

use logos::Logos;
use regex::Regex;

use crate::matrix::CheckableNumber;

/// A trait for anything we can throw in a .props, .csl, or .pctl file.
pub trait Property {
	fn parse(input: &str) -> Result<Self, String>
	where
		Self: Sized;
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

	fn parse(input: &str) -> Result<Self, String> {
		if input == "true" {
			return Ok(StateFormula::True);
		} else if input == "false" {
			let tr = Self::True;
			return Ok(!tr);
		}
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
			}
			// Currently we cover all options, but this allows us to easily add some.
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
	/// A state formula holds on an entire path.
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
		let re = Regex::new(r#"(?P<true>true)|(?P<label>"[^"]*")|(?P<not>!)|(?P<and>\&)|(?P<or>\|)|(?P<lparen>$)|(?P<rparen>$)"#)
			.unwrap();

		let mut tokens: Vec<&str> = Vec::new();

		for cap in re.captures_iter(input) {
			if let Some(_) = cap.name("true") {
				tokens.push("true");
			} else if let Some(label) = cap.name("label") {
				tokens.push(label.as_str());
			} else if let Some(_) = cap.name("not") {
				tokens.push("!");
			} else if let Some(_) = cap.name("and") {
				tokens.push("&");
			} else if let Some(_) = cap.name("or") {
				tokens.push("|");
			} else if let Some(_) = cap.name("lparen") {
				tokens.push("(");
			} else if let Some(_) = cap.name("rparen") {
				tokens.push(")");
			}
		}

		// Token processing logic based on the vector 'tokens' would go here.
		// This involves building your StateFormula recursively based on the parsed tokens.

		// Err("Parsing failed.".to_string())

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

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
	#[regex(r"true")]
	True,

	#[regex(r#""([^"]*)""#, |lex| lex.slice().to_string())]
	StringLabel(String),

	#[token("!")]
	Not,

	#[token("&")]
	And,

	#[token("|")]
	Or,

	#[token("(")]
	LParen,

	#[token(")")]
	RParen,

	#[token("[")]
	LBracket,

	#[token("]")]
	RBracket,
}

pub fn lex(input: &str) -> Vec<Token> {
	let mut lexer = Token::lexer(input);
	let mut tokens = Vec::new();

	while let Some(Ok(token)) = lexer.next() {
		tokens.push(token);
	}

	tokens
}

pub fn parse_state_formula<ValueType>(tokens: &[Token]) -> Result<StateFormula<ValueType>, String>
	where
	ValueType: CheckableNumber,
{
	let mut iter = tokens.iter().peekable();
	parse_expression(&mut iter)
}

fn parse_expression<'a, ValueType, I>(iter: &mut std::iter::Peekable<I>) -> Result<StateFormula<ValueType>, String>
where
	ValueType: CheckableNumber,
	I: Iterator<Item = &'a Token>,
{
	let mut left = parse_primary(iter)?;

	while let Some(&token) = iter.peek() {
		match token {
			Token::And => {
				iter.next(); // Consume the `&`
				let right = parse_primary(iter)?;
				left = left & right;
			}
			Token::Or => {
				iter.next(); // Consume the `&`
				// Can use it via De Morgan's Laws via negations
				let right = parse_primary(iter)?;
				left = left | right;
			}
			_ => break, // Any other token means the end of this expression
		}
	}

	Ok(left)
}

fn parse_primary<'a, ValueType, I>(iter: &mut std::iter::Peekable<I>) -> Result<StateFormula<ValueType>, String>
where
	ValueType: CheckableNumber,
	I: Iterator<Item = &'a Token>,
{
	match iter.next() {
		Some(Token::True) => Ok(StateFormula::True),
		Some(Token::StringLabel(label)) => Ok(StateFormula::StringLabel(label.to_string())),
		Some(Token::Not) => {
			let inner = parse_primary(iter)?;
			Ok(StateFormula::Not(Box::new(inner)))
		}
		Some(Token::LBracket) => {
			let expr = parse_expression(iter)?;
			if let Some(Token::RBracket) = iter.next() {
				Ok(expr)
			} else {
				Err("Expected closing brackets".to_string())
			}
		}
		Some(Token::LParen) => {
			let expr = parse_expression(iter)?;
			if let Some(Token::RParen) = iter.next() {
				Ok(expr)
			} else {
				Err("Expected closing parenthesis".to_string())
			}
		}
		None | Some(Token::RBracket) | Some(Token::RParen) => {
			Err("Unexpected end of input".to_string())
		}
		_ => parse_primary(iter),
	}
}

#[cfg(test)]
mod property_tests {
	use super::{Interval, PathFormula, StateFormula};

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
}
