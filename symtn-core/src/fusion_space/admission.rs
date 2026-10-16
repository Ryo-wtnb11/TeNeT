use super::*;

#[doc(hidden)]
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FusionSpaceAdmission {
    Unbound,
    Subset(RuleIdentity),
    Complete(RuleIdentity),
}

impl FusionSpaceAdmission {
    #[doc(hidden)]
    pub fn rule_identity(&self) -> Option<&RuleIdentity> {
        match self {
            Self::Unbound => None,
            Self::Subset(identity) | Self::Complete(identity) => Some(identity),
        }
    }
}
