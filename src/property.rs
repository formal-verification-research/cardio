use std::ops;

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
	NotEqual,
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
					BinaryApOperator::Minus => lhs_val - rhs_val,
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
			// Constant-fold when both operands are known values.
			Self::BinaryOperator(op, lhs, rhs) => {
				let l = lhs.simplify();
				let r = rhs.simplify();
				if let (Some(lv), Some(rv)) = (l.evaluate_value(), r.evaluate_value()) {
					let res = match op {
						BinaryApOperator::Plus => lv + rv,
						BinaryApOperator::Minus => lv - rv,
						BinaryApOperator::Multipy => lv * rv,
					};
					Self::Value(res)
				} else {
					Self::BinaryOperator(*op, Box::new(l), Box::new(r))
				}
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


/// Parse a simple arithmetic atomic expression.
fn parse_atomic_expression(input: &str) -> Result<AtomicExpression, String> {
	let mut p = AtomicExprParser::new(input);
	let expr = p.parse_expr()?;
	p.skip_ws();
	if p.pos != p.chars.len() {
		return Err(format!("Unexpected trailing input: {}", &input[p.pos..]));
	}
	Ok(expr)
}

struct AtomicExprParser {
	chars: Vec<char>,
	pos: usize,
}

impl AtomicExprParser {
	fn new(input: &str) -> Self {
		Self {
			chars: input.chars().collect(),
			pos: 0,
		}
	}

	fn skip_ws(&mut self) {
		while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
			self.pos += 1;
		}
	}

	fn parse_expr(&mut self) -> Result<AtomicExpression, String> {
		self.parse_add_sub()
	}

	fn parse_add_sub(&mut self) -> Result<AtomicExpression, String> {
		let mut left = self.parse_mul()?;
		loop {
			self.skip_ws();
			if self.pos < self.chars.len() && (self.chars[self.pos] == '+' || self.chars[self.pos] == '-') {
				let op = if self.chars[self.pos] == '+' {
					BinaryApOperator::Plus
				} else {
					BinaryApOperator::Minus
				};
				self.pos += 1;
				let right = self.parse_mul()?;
				left = AtomicExpression::BinaryOperator(op, Box::new(left), Box::new(right));
			} else {
				break;
			}
		}
		Ok(left)
	}

	fn parse_mul(&mut self) -> Result<AtomicExpression, String> {
		let mut left = self.parse_primary()?;
		loop {
			self.skip_ws();
			if self.pos < self.chars.len() && self.chars[self.pos] == '*' {
				self.pos += 1;
				let right = self.parse_primary()?;
				left = AtomicExpression::BinaryOperator(
					BinaryApOperator::Multipy,
					Box::new(left),
					Box::new(right),
				);
			} else {
				break;
			}
		}
		Ok(left)
	}

	fn parse_primary(&mut self) -> Result<AtomicExpression, String> {
		self.skip_ws();
		if self.pos >= self.chars.len() {
			return Err("Unexpected end of expression".to_string());
		}
		if self.chars[self.pos] == '(' {
			self.pos += 1;
			let expr = self.parse_expr()?;
			self.skip_ws();
			if self.pos >= self.chars.len() || self.chars[self.pos] != ')' {
				return Err("Expected ')'".to_string());
			}
			self.pos += 1;
			return Ok(expr);
		}
		if self.chars[self.pos] == '-' {
			self.pos += 1;
			let inner = self.parse_primary()?;
			// Represent unary minus as 0 - expr.
			return Ok(AtomicExpression::BinaryOperator(
				BinaryApOperator::Minus,
				Box::new(AtomicExpression::Value(0)),
				Box::new(inner),
			));
		}
		if self.chars[self.pos].is_ascii_digit() {
			let start = self.pos;
			while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
				self.pos += 1;
			}
			let num_str: String = self.chars[start..self.pos].iter().collect();
			let val: i64 = num_str
				.parse()
				.map_err(|_| "Invalid integer".to_string())?;
			return Ok(AtomicExpression::Value(val));
		}
		if self.chars[self.pos].is_alphabetic() || self.chars[self.pos] == '_' {
			let start = self.pos;
			while self.pos < self.chars.len()
				&& (self.chars[self.pos].is_alphanumeric() || self.chars[self.pos] == '_')
			{
				self.pos += 1;
			}
			let name: String = self.chars[start..self.pos].iter().collect();
			self.skip_ws();
			if self.pos < self.chars.len() && self.chars[self.pos] == '[' {
				// Bit index range bv[lower, upper]
				self.pos += 1;
				self.skip_ws();
				let lower = self.parse_integer()?;
				self.skip_ws();
				if self.pos >= self.chars.len() || self.chars[self.pos] != ',' {
					return Err("Expected ',' in bit index range".to_string());
				}
				self.pos += 1;
				self.skip_ws();
				let upper = self.parse_integer()?;
				self.skip_ws();
				if self.pos >= self.chars.len() || self.chars[self.pos] != ']' {
					return Err("Expected ']' in bit index range".to_string());
				}
				self.pos += 1;
				return Ok(AtomicExpression::Variable(VariableType::BitIndecies(
					lower, upper,
				)));
			}
			return Ok(AtomicExpression::Variable(VariableType::Name(name)));
		}
		Err(format!("Unexpected character: {}", self.chars[self.pos]))
	}

	fn parse_integer(&mut self) -> Result<usize, String> {
		self.skip_ws();
		let start = self.pos;
		while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
			self.pos += 1;
		}
		if start == self.pos {
			return Err("Expected integer".to_string());
		}
		let s: String = self.chars[start..self.pos].iter().collect();
		let v: usize = s.parse().map_err(|_| "Invalid integer".to_string())?;
		Ok(v)
	}
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
		// Trim and locate the top-level comparison operator.
		let s = input.trim();
		if s.is_empty() {
			return Err("Empty atomic proposition".to_string());
		}
		let mut depth = 0usize;
		let mut op_pos: Option<usize> = None;
		let mut op_len = 0usize;
		let mut operator = ComparisonOperator::Eq;
		let bytes = s.as_bytes();
		let mut i = 0usize;
		while i < s.len() {
			let c = bytes[i] as char;
			match c {
				'(' => depth += 1,
				')' => {
					if depth == 0 {
						return Err("Unbalanced parentheses".to_string());
					}
					depth -= 1;
				}
				'<' | '>' | '=' | '!' if depth == 0 => {
					// Try two-character operators first.
					let two = if i + 1 < s.len() {
						&s[i..i + 2]
					} else {
						""
					};
					let (len, op) = if two == "<=" {
						(2, ComparisonOperator::LessEq)
					} else if two == ">=" {
						(2, ComparisonOperator::GreaterEq)
					} else if two == "==" {
						(2, ComparisonOperator::Eq)
					} else if two == "!=" {
						(2, ComparisonOperator::NotEqual)
					} else {
						match c {
							'<' => (1, ComparisonOperator::Less),
							'>' => (1, ComparisonOperator::Greater),
							'=' => (1, ComparisonOperator::Eq),
							'!' => return Err("Unexpected '!' in atomic proposition".to_string()),
							_ => (0, ComparisonOperator::Eq),
						}
					};
					if len > 0 && op_pos.is_none() {
						op_pos = Some(i);
						op_len = len;
						operator = op;
					}
				}
				_ => {}
			}
			i += c.len_utf8();
		}
		if depth != 0 {
			return Err("Unbalanced parentheses".to_string());
		}
		let pos = match op_pos {
			Some(p) => p,
			None => return Err("Missing comparison operator in atomic proposition".to_string()),
		};
		let lhs_str = s[..pos].trim();
		let rhs_str = s[pos + op_len..].trim();
		if lhs_str.is_empty() || rhs_str.is_empty() {
			return Err("Missing operand in atomic proposition".to_string());
		}
		let lhs = parse_atomic_expression(lhs_str)?;
		let rhs = parse_atomic_expression(rhs_str)?;
		Ok(Self {
			lhs: Box::new(lhs),
			operator,
			rhs: Box::new(rhs),
		})
	}
}

/// An enum representing the possible time bounds that a CSL property can handle
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interval {
	/// A time bound of the form [0, T]
	TimeBoundedUpper(f64),
	/// A time bound of the form [T, T']
	TimeBoundWindow(f64, f64),
	/// A time bound of the form [T, oo] (T to infinity)
	TimeBoundedLower(f64),
	/// A bound in the number of steps. Because PCTL only supports an upper bound on the number of
	/// steps, the count in here is an upper bound, i.e., from [0, k] steps.
	StepBoundUpper(usize),
	/// The absence of a time bound (steady state)
	TimeUnbounded,
}

impl ToString for Interval {
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
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ProbabilityQueryType {
	/// A simple query that asks the probability
	SimpleQuery,
	/// A query that asks "is the probability less than p?"
	LessThan(f64),
	/// A query that asks "is the probability less than or equal to p?"
	LessThanEqual(f64),
	/// A query that asks "is the probability greater than p?"
	GreaterThan(f64),
	/// A query that asks "is the probability greater than or equal to p?"
	GreaterThanEqual(f64),
}

impl ToString for ProbabilityQueryType {
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
pub enum StateFormula {
	/// Evaluates to `true` on all states
	True,
	/// Evaluates to `false` on all states
	False,
	/// An atomic proposition which evaluates on a particular state
	AtomicProposition(evalexpr::Node),
	/// A string label for a particular state, i.e., "Absorbing"
	StringLabel(String),
	/// The negation operator on another state formula
	Not(Box<StateFormula>),
	/// The conjunction of two state formulae
	Conjunction(Box<StateFormula>, Box<StateFormula>),
	/// The Disjunction of two state formulae
	Disjunction(Box<StateFormula>, Box<StateFormula>),
	/// A probability query on a path formula. I.e., "from this state, what is the probability that
	/// the next path formula holds?"
	TransientQuery(ProbabilityQueryType, Box<PathFormula>),
	/// A steady state formula. I.e., "from this state, what is the probability that in the steady
	/// state Phi holds?"
	SteadyStateQuery(ProbabilityQueryType, Box<StateFormula>),
}

impl ops::Not for StateFormula {
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

impl ops::BitAnd for StateFormula {
	type Output = Self;
	fn bitand(self, rhs: Self) -> Self::Output {
		Self::Conjunction(Box::new(self.clone()), Box::new(rhs.clone()))
	}
}

impl ops::BitOr for StateFormula {
	type Output = Self;
	fn bitor(self, rhs: Self) -> Self::Output {
		Self::Disjunction(Box::new(self.clone()), Box::new(rhs.clone()))
	}
}

impl ToString for StateFormula {
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

impl Property for StateFormula {
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

impl StateFormula {
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
						(PathFormula::next(&phi_lower), PathFormula::next(&phi_upper))
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
							PathFormula::until(&phi_lower, *interval, &psi_lower),
							PathFormula::until(&phi_upper, *interval, &psi_upper),
						)
					}
					PathFormula::Globally(phi) => {
						// This works the same as next
						let phi_lower = *phi.clone() & !abs.clone();
						let phi_upper = *phi.clone() | abs.clone();
						(
							PathFormula::globally(&phi_lower),
							PathFormula::globally(&phi_upper),
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
pub enum PathFormula {
	/// If the state formula holds in the next state.
	Next(Box<StateFormula>),
	/// An "until" path formula over an interval
	Until(Box<StateFormula>, Interval, Box<StateFormula>),
	/// A state formula holds on an entire path. This is equivalent to Phi U False
	Globally(Box<StateFormula>),
}

impl Property for PathFormula {
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

impl ToString for PathFormula {
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

impl PathFormula {
	/// Creates a state formula of type `next`
	pub fn next(state_formula: &StateFormula) -> Self {
		Self::Next(Box::new(state_formula.clone()))
	}

	/// Creates a state formula of type `until`
	pub fn until(phi: &StateFormula, interval: Interval, psi: &StateFormula) -> Self {
		Self::Until(Box::new(phi.clone()), interval, Box::new(psi.clone()))
	}

	pub fn globally(state_formula: &StateFormula) -> Self {
		Self::Globally(Box::new(state_formula.clone()))
	}

	/// Creates an "eventually" state formula, which is equivalently "true U psi"
	pub fn eventually(interval: Interval, state_formula: &StateFormula) -> Self {
		Self::Until(
			Box::new(StateFormula::True),
			interval,
			Box::new(state_formula.clone()),
		)
	}

	// TODO: Weak until and release.
}

#[cfg(kani)]
mod kani_proofs {
	use super::*;

	#[kani::proof]
	#[kani::unwind(4)]
	fn evaluate_value_minus_no_panic() {
		let a: i64 = kani::any();
		let b: i64 = kani::any();
		kani::assume(a > -1000 && a < 1000);
		kani::assume(b > -1000 && b < 1000);
		let expr = AtomicExpression::BinaryOperator(
			BinaryApOperator::Minus,
			Box::new(AtomicExpression::Value(a)),
			Box::new(AtomicExpression::Value(b)),
		);
		let r = expr.evaluate_value();
		assert!(r == Some(a - b));
	}

	#[kani::proof]
	#[kani::unwind(4)]
	fn evaluate_value_plus_no_panic() {
		let a: i64 = kani::any();
		let b: i64 = kani::any();
		kani::assume(a > -1000 && a < 1000);
		kani::assume(b > -1000 && b < 1000);
		let expr = AtomicExpression::BinaryOperator(
			BinaryApOperator::Plus,
			Box::new(AtomicExpression::Value(a)),
			Box::new(AtomicExpression::Value(b)),
		);
		let r = expr.evaluate_value();
		assert!(r == Some(a + b));
	}

	#[kani::proof]
	#[kani::unwind(4)]
	fn evaluate_value_multiply_no_panic() {
		let a: i64 = kani::any();
		let b: i64 = kani::any();
		kani::assume(a > -32 && a < 32);
		kani::assume(b > -32 && b < 32);
		let expr = AtomicExpression::BinaryOperator(
			BinaryApOperator::Multipy,
			Box::new(AtomicExpression::Value(a)),
			Box::new(AtomicExpression::Value(b)),
		);
		let r = expr.evaluate_value();
		assert!(r == Some(a * b));
	}
}

#[cfg(test)]
mod property_tests {
	use super::{
		AtomicExpression, AtomicProposition, BinaryApOperator, ComparisonOperator, Interval, PathFormula,
		ProbabilityQueryType, Property, StateFormula, VariableType,
	};

	#[test]
	fn construction_test() {
		let phi: StateFormula = StateFormula::absorbing();
		let neg_phi = !phi.clone();
		let eventually_abs: PathFormula = PathFormula::eventually(Interval::TimeUnbounded, &phi);
		let globally_not_abs: PathFormula = PathFormula::globally(&neg_phi);
		let p: StateFormula = StateFormula::TransientQuery(
			ProbabilityQueryType::SimpleQuery,
			Box::new(eventually_abs.clone()),
		);
		println!("Property 1: {}", phi.to_string());
		println!("Property 2: {}", neg_phi.to_string());
		println!("Property 3: {}", eventually_abs.to_string());
		println!("Property 4: {}", globally_not_abs.to_string());
		println!("Property 5: {}", p.to_string());
	}

	// #[test]
	// fn negation_test() {
	// let phi: StateFormula = StateFormula::AtomicProposition(evalexpr::)
	//unimplemented!();
	//}

	#[test]
	fn evaluate_value_minus() {
		let expr = AtomicExpression::BinaryOperator(
			BinaryApOperator::Minus,
			Box::new(AtomicExpression::Value(10)),
			Box::new(AtomicExpression::Value(3)),
		);
		assert_eq!(expr.evaluate_value(), Some(7));
	}

	#[test]
	fn parse_test() {
		let prop_result = StateFormula::parse(&"P=? [true U \"absorbing\"]");
		match &prop_result {
			Ok(prop) => println!("{}", prop.to_string()),
			Err(err) => println!("Got error: {}", err),
		}
		assert!(prop_result.is_ok());
	}

	#[test]
	fn atomic_proposition_parse_test() {
		let ap = AtomicProposition::parse("x + 1 > 2").unwrap();
		assert_eq!(ap.lhs, Box::new(AtomicExpression::BinaryOperator(
			BinaryApOperator::Plus,
			Box::new(AtomicExpression::Variable(VariableType::Name("x".to_string()))),
			Box::new(AtomicExpression::Value(1)),
		)));
		assert_eq!(ap.operator, ComparisonOperator::Greater);
		assert_eq!(ap.rhs, Box::new(AtomicExpression::Value(2)));
		let ap2 = AtomicProposition::parse("bv[0, 3] <= 5").unwrap();
		assert_eq!(ap2.operator, ComparisonOperator::LessEq);
		assert!(AtomicProposition::parse("x >").is_err());
	}
}
