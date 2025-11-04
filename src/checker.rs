use crate::matrix::*;
use crate::*;

/// A CTMC transition matrix
pub trait CtmcTransMat {
	fn uniformize(rate_mat: Self) -> Self;
}

impl<EntryType> CtmcTransMat for sprs::CsMat<EntryType>
where
	EntryType: num::Num + Clone,
{
	fn uniformize(rate_mat: Self) -> Self {
		unimplemented!();
	}
}

pub struct CslChecker<EntryType>
where
	EntryType: num::Num + Clone,
{
	qualitative: bool,
	placeholder: EntryType,
}

pub struct CheckContext<EntryType>
where
	EntryType: MatEntry,
{
	last_dist: CsVec<EntryType>,
}
