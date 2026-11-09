// Verus proofs for selected Cardio helper logic.
// Run with: /home/landon/Downloads/verus/verus verification/verus/proofs.rs

use vstd::prelude::*;

verus! {
// Mirror of the corrected subtraction case in AtomicExpression::evaluate_value.
fn eval_minus(a: i64, b: i64) -> (r: Option<i64>)
    ensures
        match r {
            Some(s) => s as int == a as int - b as int,
            None => true,
        },
{
    a.checked_sub(b)
}

// checked_add never panics and is correct when no overflow occurs.
fn checked_add_spec(a: i64, b: i64) -> (r: Option<i64>)
    ensures
        match r {
            Some(s) => s as int == a as int + b as int,
            None => true,
        },
{
    a.checked_add(b)
}

fn checked_mul_spec(a: i64, b: i64) -> (r: Option<i64>)
    ensures
        match r {
            Some(s) => s as int == a as int * b as int,
            None => true,
        },
{
    a.checked_mul(b)
}

// Mirror of AtomicExpression::simplify constant-folding for addition.
fn simplify_add(a: i64, b: i64) -> (r: i64)
    requires a as int + b as int <= i64::MAX as int && a as int + b as int >= i64::MIN as int,
    ensures r as int == a as int + b as int,
{
    a + b
}

fn abs_i64(x: i64) -> (r: i64)
    requires x > i64::MIN,
    ensures r >= 0,
{
    if x < 0 { -x } else { x }
}
}

fn main() {}
