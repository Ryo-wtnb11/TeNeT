use tenet_core::{
    FermionParityFusionRule, FibonacciFusionRule, LoweredMultiplicityFreeAlgebra,
    MultiplicityFreePivotalSymbols, SU2FusionRule, U1FusionRule, Z2FusionRule,
};

fn assert_lowered<T: LoweredMultiplicityFreeAlgebra>() {}

fn assert_pivotal<T: MultiplicityFreePivotalSymbols>() {}

#[test]
fn built_in_providers_implement_core_rule_bridge_traits() {
    assert_lowered::<Z2FusionRule>();
    assert_lowered::<FermionParityFusionRule>();
    assert_lowered::<U1FusionRule>();
    assert_lowered::<SU2FusionRule>();
    assert_pivotal::<Z2FusionRule>();
    assert_pivotal::<FermionParityFusionRule>();
    assert_pivotal::<FibonacciFusionRule>();
}
