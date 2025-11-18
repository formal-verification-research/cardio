// Parsing stuff

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
	#[regex(r"true")]
	True,
	#[regex(r"false")]
	False,
	#[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().unwrap())]
	Integer(i64),
	#[regex(r"[0-9]+\.[0-9]*", |lex| lex.slice().parse::<f64>().unwrap())]
	Float(f64),
	#[token("+")]
	Plus,
	#[token("-")]
	Minus,
	#[token("*")]
	Multiply,

	// Comparison operators
	#[token("==")]
	/// Strict equality
	Equal,
	/// Not equal (inverse of strict equality)
	#[token("!=")]
	NotEqual,
	/// Strict less than
	#[token("<")]
	LessThan,
	/// Strict greater than
	#[token(">")]
	GreaterThan,
	/// Less than or equal
	#[token("<=")]
	LessThanOrEqual,
	/// Greater than or equal
	#[token(">=")]
	GreaterThanOrEqual,
	/// Query operator. Not strictly a comparison operator but very similar.
	#[token("=?")]
	ValueQuery,

	// String label
	#[regex(r#""([^"]*)""#, |lex| lex.slice().to_string())]
	StringLabel(String),

	// Boolean (state-formula) tokens
	#[token("!")]
	Not,
	#[token("&")]
	And,
	#[token("|")]
	Or,

	// Parenthesis and bracket tokens
	#[token("(")]
	LParen,
	#[token(")")]
	RParen,
	#[token("[")]
	LBracket,
	#[token("]")]
	RBracket,

	/// The comma token only appears in intervals
	#[token(",")]
	Comma,

	// probability query operators
	#[token("P")]
	///A transient probability query
	ProbabilityQuery,
	// steady state query operators
	#[token("S")]
	/// A steady state probability query
	SteadyStateQuery,

	// path formula operators
	#[token("G")]
	/// The state formula holds everywhere
	Globally,
	#[token("X")]
	/// The state formula holds in the next state
	Next,
	#[token("U")]
	/// A state formula holds until another state formula takes over
	Until,
}

pub fn lex(input: &str) -> Vec<Token> {
	let mut lexer = Token::lexer(input);
	let mut tokens = Vec::new();

	while let Some(Ok(token)) = lexer.next() {
		tokens.push(token);
	}

	tokens
}

fn parse_interval<'a, ValueType, I>(
	iter: &mut std::iter::Peekable<I>,
) -> Result<Interval<ValueType>, String>
where
	ValueType: CheckableNumber + std::convert::From<f64> + std::convert::From<i64>,
	I: Iterator<Item = &'a Token>,
{
	let first_token = iter.peek();
	match first_token {
		// We have a classic interval
		Some(Token::LBracket) => {
			// Consume the iterator
			iter.next();
			// Next there must be a number, comma, then number
			let lower_bound = iter.next().ok_or("Missing lower bound")?;
			let comma = iter
				.next()
				.ok_or("Must have a comma in between interval bounds.")?;
			let upper_bound = iter.next().ok_or("Missing upper bound")?;

			let low = if let Token::Float(l) = lower_bound {
				Ok(ValueType::from(*l))
			} else if let Token::Integer(l) = lower_bound {
				Ok(ValueType::from(*l))
			} else {
				Err("Lower bound must be numeric!")
			}?;
			let up = if let Token::Float(l) = upper_bound {
				Ok(ValueType::from(*l))
			} else if let Token::Integer(l) = upper_bound {
				Ok(ValueType::from(*l))
			} else {
				Err("Upper bound must be numeric!".to_string())
			}?;
			match comma {
				Token::Comma => Ok(Interval::TimeBoundWindow(low, up)),
				_ => Err("Unexpected token in between bounds!".to_string()),
			}
		}
		// If we have a greater than or greater than equal, then we just need to consume one token
		// for the lower bound. Further, we can treat it the same since it's in continuous time and
		// PCTL does not support this kind of interval. Therefore, we can combine match arms here.
		Some(Token::GreaterThan) | Some(Token::GreaterThanOrEqual) => {
			let lower_bound = iter.next().ok_or("Missing lower bound")?;
			let low = match lower_bound {
				Token::Float(l) => Ok(ValueType::from(*l)),
				Token::Integer(i) => Ok(ValueType::from(*i)),
				_ => Err("Bound must be numeric!".to_string()),
			}?;
			Ok(Interval::TimeBoundedLower(low))
		}
		// If we have a less than or leq it may be PCTL, and so we have to treat them differently
		Some(Token::LessThan) => {
			let upper_bound = iter.next().ok_or("Missing upper bound")?;
			let up = match upper_bound {
				Token::Float(u) => Ok(ValueType::from(*u)),
				Token::Integer(i) => Ok(ValueType::from(*i)),
				_ => Err("Bound must be numeric!".to_string()),
			}?;
			Ok(Interval::TimeBoundedUpper(up))
		}
		// For less than equal to if it's an integer we have to adjust by 1 since a <= b is the
		// same as a < b + 1 for integers. Floats/reals can remain unchanged as since in CSL, we
		// are in continuous time so <T and <=T are largely equivalent for continuous R.V.
		Some(Token::LessThanOrEqual) => {
			let upper_bound = iter.next().ok_or("Missing upper bound")?;
			let up = match upper_bound {
				Token::Float(u) => Ok(ValueType::from(*u)),
				Token::Integer(i) => Ok(ValueType::from(*i + 1)),
				_ => Err("Bound must be numeric!".to_string()),
			}?;
			Ok(Interval::TimeBoundedUpper(up))
		}
		// We just assume there is no interval or it is time unbound
		_ => Ok(Interval::TimeUnbounded),
	}
}

/// Parses binary operators
fn parse_expression<'a, ValueType, I>(
	iter: &mut std::iter::Peekable<I>,
) -> Result<StateFormula<ValueType>, String>
where
	ValueType: CheckableNumber + std::convert::From<f64>,
	I: Iterator<Item = &'a Token>,
{
	let mut left = parse_state_formula(iter)?;

	while let Some(&token) = iter.peek() {
		match token {
			Token::And => {
				iter.next(); // Consume the `&`
				let right = parse_state_formula(iter)?;
				left = left & right;
			}
			Token::Or => {
				iter.next(); // Consume the `&`
				// Can use it via De Morgan's Laws via negations
				let right = parse_state_formula(iter)?;
				left = left | right;
			}
			_ => break, // Any other token means the end of this expression
		}
	}

	Ok(left)
}

fn parse_state_formula<'a, ValueType, I>(
	iter: &mut std::iter::Peekable<I>,
) -> Result<StateFormula<ValueType>, String>
where
	ValueType: CheckableNumber + std::convert::From<f64>,
	I: Iterator<Item = &'a Token>,
{
	match iter.next() {
		Some(Token::True) => Ok(StateFormula::True),
		Some(Token::StringLabel(label)) => Ok(StateFormula::StringLabel(label.to_string())),
		Some(Token::Not) => {
			let inner = parse_state_formula(iter)?;
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
		Some(Token::ProbabilityQuery) => {
			// Next token must be a comparison operator
			let comparison_token = iter.next();
			match comparison_token {
				Some(Token::ValueQuery) => {
					let path_formula = parse_path_formula(iter)?;
					Ok(StateFormula::TransientQuery(
						ProbabilityQueryType::<ValueType>::SimpleQuery,
						Box::new(path_formula),
					))
				}
				Some(Token::LessThan)
				| Some(Token::GreaterThan)
				| Some(Token::GreaterThanOrEqual)
				| Some(Token::LessThanOrEqual)
				| Some(Token::Equal)
				| Some(Token::NotEqual) => {
					// Next token must be a float
					let bound_token = iter.next();
					if let Some(Token::Float(bound)) = bound_token {
						// A little validation goes a long way
						if *bound < 0.0 || *bound > 1.0 {
							return Err("Probability bound for transient query must be between zero and one!".to_string());
						}
						match comparison_token {
							Some(Token::LessThan) => {
								let path_formula = parse_path_formula(iter)?;
								Ok(StateFormula::TransientQuery(
									ProbabilityQueryType::<ValueType>::LessThan((*bound).into()),
									Box::new(path_formula),
								))
							}
							Some(Token::GreaterThan) => {
								let path_formula = parse_path_formula(iter)?;
								Ok(StateFormula::TransientQuery(
									ProbabilityQueryType::<ValueType>::GreaterThan((*bound).into()),
									Box::new(path_formula),
								))
							}
							Some(Token::LessThanOrEqual) => {
								let path_formula = parse_path_formula(iter)?;
								Ok(StateFormula::TransientQuery(
									ProbabilityQueryType::<ValueType>::LessThanEqual(
										(*bound).into(),
									),
									Box::new(path_formula),
								))
							}
							Some(Token::GreaterThanOrEqual) => {
								let path_formula = parse_path_formula(iter)?;
								Ok(StateFormula::TransientQuery(
									ProbabilityQueryType::<ValueType>::GreaterThanEqual(
										(*bound).into(),
									),
									Box::new(path_formula),
								))
							}
							Some(Token::Equal) => {
								Err("Strict equality not supported in transient queries!"
									.to_string())
							}
							Some(Token::NotEqual) => {
								Err("Inequality not supported in transient queries!".to_string())
							}
							_ => Err("Expected a comparison or value query token!".to_string()),
						}
					} else {
						Err("Must have probability bound unless query type is `=?`!".to_string())
					}
				}
				_ => Err("Expected comparison token".to_string()),
			}
		}
		// While this case is very similar to the previous, a steady state query takes a state
		// formula afterward, whereas the transient query takes a path formula. Thus, it appears to
		// me that unfortunately the near-duplication can't be avoided.
		Some(Token::SteadyStateQuery) => {
			// Next token must be a comparison operator
			let comparison_token = iter.next();
			match comparison_token {
				Some(Token::ValueQuery) => {
					let state_formula = parse_state_formula(iter)?;
					Ok(StateFormula::SteadyStateQuery(
						ProbabilityQueryType::<ValueType>::SimpleQuery,
						Box::new(state_formula),
					))
				}
				Some(Token::LessThan)
				| Some(Token::GreaterThan)
				| Some(Token::GreaterThanOrEqual)
				| Some(Token::LessThanOrEqual)
				| Some(Token::Equal)
				| Some(Token::NotEqual) => {
					// Next token must be a float
					let bound_token = iter.next();
					if let Some(Token::Float(bound)) = bound_token {
						// Again, a little validation goes a long way
						if *bound < 0.0 || *bound > 1.0 {
							return Err("Probability bound for steady state query must be between zero and one!".to_string());
						}
						match comparison_token {
							Some(Token::LessThan) => {
								let state_formula = parse_state_formula(iter)?;
								Ok(StateFormula::SteadyStateQuery(
									ProbabilityQueryType::<ValueType>::LessThan((*bound).into()),
									Box::new(state_formula),
								))
							}
							Some(Token::GreaterThan) => {
								let state_formula = parse_state_formula(iter)?;
								Ok(StateFormula::SteadyStateQuery(
									ProbabilityQueryType::<ValueType>::GreaterThan((*bound).into()),
									Box::new(state_formula),
								))
							}
							Some(Token::LessThanOrEqual) => {
								let state_formula = parse_state_formula(iter)?;
								Ok(StateFormula::SteadyStateQuery(
									ProbabilityQueryType::<ValueType>::LessThanEqual(
										(*bound).into(),
									),
									Box::new(state_formula),
								))
							}
							Some(Token::GreaterThanOrEqual) => {
								let state_formula = parse_state_formula(iter)?;
								Ok(StateFormula::SteadyStateQuery(
									ProbabilityQueryType::<ValueType>::GreaterThanEqual(
										(*bound).into(),
									),
									Box::new(state_formula),
								))
							}
							Some(Token::Equal) => {
								Err("Strict equality not supported in transient queries!"
									.to_string())
							}
							Some(Token::NotEqual) => {
								Err("Inequality not supported in transient queries!".to_string())
							}
							_ => Err("Expected a comparison or value query token!".to_string()),
						}
					} else {
						Err("Must have probability bound unless query type is `=?`!".to_string())
					}
				}
				_ => Err("Expected comparison token".to_string()),
			}
		}
		None | Some(Token::RBracket) | Some(Token::RParen) => {
			Err("Unexpected end of input".to_string())
		}
		_ => parse_state_formula(iter),
	}
}

fn parse_path_formula<'a, ValueType, I>(
	iter: &mut std::iter::Peekable<I>,
) -> Result<PathFormula<ValueType>, String>
where
	ValueType: CheckableNumber + std::convert::From<f64>,
	I: Iterator<Item = &'a Token>,
{
	unimplemented!();
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
