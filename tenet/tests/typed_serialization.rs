use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SU2FusionRule, SU2Irrep, SectorId, SectorVec, TypedSectorAdmission, U1FusionRule, U1Irrep,
};
use tenet::prelude::{Complex64, Runtime};
use tenet::typed::{
    DecodeError, DecodeLimits, GradedSpace, NetworkReuseClass, SectorSpectrum, TensorMap,
    TypedPersistenceCodec,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodecError {
    MissingProvider,
    InvalidSector,
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for CodecError {}

struct Su2Codec {
    provider: Arc<SU2FusionRule>,
    resolve_calls: AtomicUsize,
    missing_provider: bool,
    accept_foreign_key: bool,
}

impl Su2Codec {
    fn new(provider: Arc<SU2FusionRule>) -> Self {
        Self {
            provider,
            resolve_calls: AtomicUsize::new(0),
            missing_provider: false,
            accept_foreign_key: false,
        }
    }

    fn missing(provider: Arc<SU2FusionRule>) -> Self {
        Self {
            missing_provider: true,
            ..Self::new(provider)
        }
    }

    fn mismatched(provider: Arc<SU2FusionRule>) -> Self {
        Self {
            accept_foreign_key: true,
            ..Self::new(provider)
        }
    }
}

impl TypedPersistenceCodec<SU2FusionRule> for Su2Codec {
    type Error = CodecError;

    fn provider_key(&self, _provider: &SU2FusionRule) -> Result<Vec<u8>, Self::Error> {
        Ok(b"su2".to_vec())
    }

    fn resolve_provider(&self, key: &[u8]) -> Result<Arc<SU2FusionRule>, Self::Error> {
        self.resolve_calls.fetch_add(1, Ordering::Relaxed);
        if self.missing_provider || (key != b"su2" && !self.accept_foreign_key) {
            Err(CodecError::MissingProvider)
        } else {
            Ok(Arc::clone(&self.provider))
        }
    }

    fn encode_sector(
        &self,
        _provider: &SU2FusionRule,
        sector: &SU2Irrep,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![sector.twice_spin() as u8])
    }

    fn decode_sector(
        &self,
        _provider: &SU2FusionRule,
        bytes: &[u8],
    ) -> Result<SU2Irrep, Self::Error> {
        let &[twice_spin] = bytes else {
            return Err(CodecError::InvalidSector);
        };
        SU2Irrep::try_from_twice_spin(twice_spin as usize).ok_or(CodecError::InvalidSector)
    }
}

struct U1Codec {
    provider: Arc<U1FusionRule>,
}

impl TypedPersistenceCodec<U1FusionRule> for U1Codec {
    type Error = CodecError;

    fn provider_key(&self, _provider: &U1FusionRule) -> Result<Vec<u8>, Self::Error> {
        Ok(b"u1".to_vec())
    }

    fn resolve_provider(&self, key: &[u8]) -> Result<Arc<U1FusionRule>, Self::Error> {
        if key == b"u1" {
            Ok(Arc::clone(&self.provider))
        } else {
            Err(CodecError::MissingProvider)
        }
    }

    fn encode_sector(
        &self,
        _provider: &U1FusionRule,
        sector: &U1Irrep,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(sector.charge().to_le_bytes().to_vec())
    }

    fn decode_sector(
        &self,
        _provider: &U1FusionRule,
        bytes: &[u8],
    ) -> Result<U1Irrep, Self::Error> {
        let bytes: [u8; 4] = bytes.try_into().map_err(|_| CodecError::InvalidSector)?;
        U1Irrep::try_new(i32::from_le_bytes(bytes)).ok_or(CodecError::InvalidSector)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum GenericLabel {
    Vacuum,
    X,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GenericError;

impl fmt::Display for GenericError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid synthetic Generic sector")
    }
}

impl std::error::Error for GenericError {}

struct GenericToy;

impl GenericToy {
    const VACUUM: SectorId = SectorId::new(0);
    const X: SectorId = SectorId::new(1);

    fn channels(left: SectorId, right: SectorId) -> Result<SectorVec, GenericError> {
        match (left, right) {
            (Self::VACUUM, sector) | (sector, Self::VACUUM)
                if sector == Self::VACUUM || sector == Self::X =>
            {
                Ok([sector].into_iter().collect())
            }
            (Self::X, Self::X) => Ok([Self::VACUUM, Self::X].into_iter().collect()),
            _ => Err(GenericError),
        }
    }

    fn multiplicity(left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left, right, coupled) == (Self::X, Self::X, Self::X) {
            2
        } else {
            usize::from(
                Self::channels(left, right).is_ok_and(|channels| channels.contains(&coupled)),
            )
        }
    }
}

impl CheckedGenericFusion for GenericToy {
    type Error = GenericError;

    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x1003, Arc::<[u8]>::from(*b"generic-toy"))
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Self::VACUUM
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        match sector {
            Self::VACUUM | Self::X => Ok(sector),
            _ => Err(GenericError),
        }
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Self::channels(left, right)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Self::channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(Self::multiplicity(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for GenericToy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        match sector {
            Self::VACUUM => Ok(1.0),
            Self::X => Ok((1.0 + 2.0_f64.sqrt()).sqrt()),
            _ => Err(GenericError),
        }
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        Ok(self.try_sqrt_dim_scalar(sector)?.recip())
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.try_dual(sector).map(|_| 1.0)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        let shape = (
            Self::multiplicity(a, b, e),
            Self::multiplicity(e, c, d),
            Self::multiplicity(b, c, f),
            Self::multiplicity(a, f, d),
        );
        let rows = shape.0 * shape.1;
        let cols = shape.2 * shape.3;
        Ok(GenericFArray::new(
            (0..rows * cols)
                .map(|index| f64::from(index / cols == index % cols))
                .collect(),
            shape,
        ))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        let size = Self::multiplicity(a, b, c);
        Ok(GenericRMatrix::new(
            (0..size * size)
                .map(|index| f64::from(index / size == index % size))
                .collect(),
            size,
            size,
        ))
    }
}

impl TypedSectorAdmission for GenericToy {
    type Sector = GenericLabel;
    type Error = GenericError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        Ok(match sector {
            GenericLabel::Vacuum => Self::VACUUM,
            GenericLabel::X => Self::X,
        })
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        match sector {
            Self::VACUUM => Ok(GenericLabel::Vacuum),
            Self::X => Ok(GenericLabel::X),
            _ => Err(GenericError),
        }
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.try_dual(sector)
    }
}

struct GenericCodec {
    provider: Arc<GenericToy>,
}

impl TypedPersistenceCodec<GenericToy> for GenericCodec {
    type Error = GenericError;

    fn provider_key(&self, _provider: &GenericToy) -> Result<Vec<u8>, Self::Error> {
        Ok(b"generic-toy".to_vec())
    }

    fn resolve_provider(&self, key: &[u8]) -> Result<Arc<GenericToy>, Self::Error> {
        if key == b"generic-toy" {
            Ok(Arc::clone(&self.provider))
        } else {
            Err(GenericError)
        }
    }

    fn encode_sector(
        &self,
        _provider: &GenericToy,
        sector: &GenericLabel,
    ) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![match sector {
            GenericLabel::Vacuum => 0,
            GenericLabel::X => 1,
        }])
    }

    fn decode_sector(
        &self,
        _provider: &GenericToy,
        bytes: &[u8],
    ) -> Result<GenericLabel, Self::Error> {
        match bytes {
            [0] => Ok(GenericLabel::Vacuum),
            [1] => Ok(GenericLabel::X),
            _ => Err(GenericError),
        }
    }
}

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn su2_leg(provider: &Arc<SU2FusionRule>, dual: bool) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
        dual,
    )
    .unwrap()
}

fn u64_at(bytes: &[u8], offset: usize) -> usize {
    usize::try_from(u64::from_le_bytes(
        bytes[offset..offset + 8].try_into().unwrap(),
    ))
    .unwrap()
}

fn header_end(bytes: &[u8]) -> usize {
    20 + u64_at(bytes, 12)
}

fn dense_leg_degeneracy_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut position = header_end(bytes) + 1;
    let rank = u64_at(bytes, position) + u64_at(bytes, position + 8);
    position += 16;
    let mut offsets = Vec::new();
    for _ in 0..rank {
        position += 1;
        let sectors = u64_at(bytes, position);
        position += 8;
        for _ in 0..sectors {
            let label_len = u64_at(bytes, position);
            position += 8 + label_len;
            offsets.push(position);
            position += 8;
        }
    }
    offsets
}

#[test]
fn checked_generic_multiplicity_keys_payload_and_resolver_arc_roundtrip() {
    let runtime = runtime();
    let provider = Arc::new(GenericToy);
    let codec = GenericCodec {
        provider: Arc::clone(&provider),
    };
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(GenericLabel::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            let coupled = usize::from(*trees.coupled() == GenericLabel::X);
            let codomain_mu = trees.codomain_vertices()[0].get();
            let domain_mu = trees.domain_vertices()[0].get();
            (1000 * coupled + 100 * codomain_mu + 10 * domain_mu + indices.iter().sum::<usize>())
                as f64
        })
        .unwrap();

    let keys = (0..source.block_count())
        .map(|index| source.block_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    let mu_one = keys
        .iter()
        .find(|key| {
            key.coupled() == &GenericLabel::X
                && key.codomain_vertices()[0].get() == 1
                && key.domain_vertices()[0].get() == 1
        })
        .unwrap();
    let mu_two = keys
        .iter()
        .find(|key| {
            key.coupled() == &GenericLabel::X
                && key.codomain_vertices()[0].get() == 2
                && key.domain_vertices()[0].get() == 1
        })
        .unwrap();
    assert_eq!(mu_one.coupled(), mu_two.coupled());
    assert_eq!(mu_one.codomain_uncoupled(), mu_two.codomain_uncoupled());
    assert_eq!(mu_one.codomain_innerlines(), mu_two.codomain_innerlines());
    assert_eq!(mu_one.domain_uncoupled(), mu_two.domain_uncoupled());
    assert_eq!(mu_one.domain_innerlines(), mu_two.domain_innerlines());
    assert_eq!(mu_one.domain_vertices(), mu_two.domain_vertices());
    assert_eq!(mu_one.codomain_vertices()[0].get(), 1);
    assert_eq!(mu_two.codomain_vertices()[0].get(), 2);

    let bytes = source.to_bytes_with(&codec).unwrap();
    let restored = TensorMap::<GenericToy, f64>::from_bytes_with(
        &runtime,
        &bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    for space in restored.codomain().iter().chain(restored.domain().iter()) {
        assert!(std::ptr::eq(space.provider(), provider.as_ref()));
    }
    assert_eq!(restored.data().len(), source.data().len());
    assert!(restored
        .data()
        .iter()
        .zip(source.data())
        .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    assert_eq!(restored.block_count(), source.block_count());
    for index in 0..source.block_count() {
        assert_eq!(
            restored.block_fusion_trees(index).unwrap(),
            source.block_fusion_trees(index).unwrap()
        );
        assert_eq!(restored.block(index).unwrap(), source.block(index).unwrap());
    }

    let hermitian = source.add(&source.adjoint().unwrap(), 1.0, 1.0).unwrap();
    let factor = hermitian.eigh_full().unwrap().0;
    assert!(matches!(
        factor.network_reuse_class(false),
        NetworkReuseClass::Compact
    ));
    let factor_adjoint = factor.adjoint().unwrap();
    assert!(matches!(
        factor_adjoint.network_reuse_class(false),
        NetworkReuseClass::LazyAdjoint
    ));
    let restored_adjoint = TensorMap::<GenericToy, f64>::from_bytes_with(
        &runtime,
        &factor_adjoint.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(std::ptr::eq(restored_adjoint.provider(), provider.as_ref()));
    assert!(
        restored_adjoint.network_reuse_class(false) == factor_adjoint.network_reuse_class(false)
    );
    assert_eq!(restored_adjoint.data(), factor_adjoint.data());
    assert!(matches!(
        restored_adjoint
            .adjoint()
            .unwrap()
            .network_reuse_class(false),
        NetworkReuseClass::Compact
    ));
}

#[test]
fn admitted_shape_limit_precedes_dense_payload_allocation() {
    let runtime = runtime();
    let provider = Arc::new(GenericToy);
    let codec = GenericCodec {
        provider: Arc::clone(&provider),
    };
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(GenericLabel::X, 1)], false).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg, &leg], [&leg, &leg], |_, _| 1.0).unwrap();
    let mut forged = source.to_bytes_with(&codec).unwrap();
    let offsets = dense_leg_degeneracy_offsets(&forged);
    assert_eq!(offsets.len(), 4);
    for offset in offsets {
        forged[offset..offset + 8].copy_from_slice(&1024u64.to_le_bytes());
    }

    let limit = source.data().len();
    assert!(matches!(
        TensorMap::<GenericToy, f64>::from_bytes_with(
            &runtime,
            &forged,
            DecodeLimits {
                max_elements: limit,
                ..DecodeLimits::default()
            },
            &codec,
        ),
        Err(DecodeError::LimitExceeded {
            resource: "admitted dense elements",
            actual,
            limit: observed_limit,
        }) if actual > observed_limit && observed_limit == limit
    ));
}

#[test]
fn graded_space_roundtrip_keeps_duality_exact_arc_and_deterministic_bytes() {
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let source = su2_leg(&provider, true);

    let bytes = source.to_bytes_with(&codec).unwrap();
    assert_eq!(source.to_bytes_with(&codec).unwrap(), bytes);
    let restored =
        GradedSpace::<SU2FusionRule>::from_bytes_with(&bytes, DecodeLimits::default(), &codec)
            .unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert!(restored.is_dual());
    assert_eq!(restored.sectors().unwrap(), source.sectors().unwrap());
    assert_eq!(restored.degeneracies(), source.degeneracies());
}

#[test]
fn non_self_dual_u1_space_roundtrip_does_not_dualize_twice() {
    let provider = Arc::new(U1FusionRule);
    let codec = U1Codec {
        provider: Arc::clone(&provider),
    };
    let source = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(3), 7)], true).unwrap();
    assert_eq!(source.sectors().unwrap(), [U1Irrep::new(-3)]);

    let restored = GradedSpace::<U1FusionRule>::from_bytes_with(
        &source.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert!(restored.is_dual());
    assert_eq!(restored.sectors().unwrap(), [U1Irrep::new(-3)]);
    assert_eq!(restored.degeneracies(), [7]);
}

#[test]
fn dense_su2_f64_and_c64_roundtrip_exact_bits_and_semantic_blocks() {
    let runtime = runtime();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let codomain = su2_leg(&provider, false);
    let domain = su2_leg(&provider, true);
    let real_bits = [
        0x8000_0000_0000_0000,
        0x3ff0_0000_0000_0001,
        0x7ff8_0000_0000_0042,
    ];
    let real: TensorMap<_, f64> = TensorMap::from_block_fn(
        &runtime,
        [&codomain, &codomain, &codomain],
        [&domain],
        |trees, indices| {
            f64::from_bits(
                real_bits[(trees.coupled().twice_spin() + indices.iter().sum::<usize>()) % 3],
            )
        },
    )
    .unwrap();
    let complex: TensorMap<_, Complex64> = TensorMap::from_block_fn(
        &runtime,
        [&codomain, &codomain, &codomain],
        [&domain],
        |trees, indices| {
            let index = (trees.coupled().twice_spin() + indices.iter().sum::<usize>()) % 3;
            Complex64::new(
                f64::from_bits(real_bits[index]),
                f64::from_bits(real_bits[(index + 1) % 3]),
            )
        },
    )
    .unwrap();
    assert!((0..real.block_count()).any(|index| {
        !real
            .block_fusion_trees(index)
            .unwrap()
            .codomain_innerlines()
            .is_empty()
    }));

    let real_bytes = real.to_bytes_with(&codec).unwrap();
    let restored_real = TensorMap::<SU2FusionRule, f64>::from_bytes_with(
        &runtime,
        &real_bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    let complex_bytes = complex.to_bytes_with(&codec).unwrap();
    let restored_complex = TensorMap::<SU2FusionRule, Complex64>::from_bytes_with(
        &runtime,
        &complex_bytes,
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();

    assert_eq!(real.to_bytes_with(&codec).unwrap(), real_bytes);
    assert_eq!(complex.to_bytes_with(&codec).unwrap(), complex_bytes);
    assert_eq!(
        restored_real
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        real.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        restored_complex
            .data()
            .iter()
            .map(|x| (x.re.to_bits(), x.im.to_bits()))
            .collect::<Vec<_>>(),
        complex
            .data()
            .iter()
            .map(|x| (x.re.to_bits(), x.im.to_bits()))
            .collect::<Vec<_>>()
    );
    assert!(std::ptr::eq(restored_real.provider(), provider.as_ref()));
    assert_eq!(restored_real.block_count(), real.block_count());
    for index in 0..real.block_count() {
        assert_eq!(
            restored_real.block_fusion_trees(index).unwrap(),
            real.block_fusion_trees(index).unwrap()
        );
        assert_eq!(
            restored_real.block(index).unwrap().shape(),
            real.block(index).unwrap().shape()
        );
    }
}

#[test]
fn compact_and_lazy_representations_survive_roundtrip() {
    let runtime = runtime();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let leg = su2_leg(&provider, false);
    let spectrum = vec![
        SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(0),
            values: vec![f64::from_bits(0x8000_0000_0000_0000)],
        },
        SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(1),
            values: vec![2.0, f64::from_bits(0x7ff8_0000_0000_0007)],
        },
    ];
    let compact = TensorMap::<SU2FusionRule, f64>::diagonal(&runtime, &leg, spectrum).unwrap();
    let restored_compact = TensorMap::<SU2FusionRule, f64>::from_bytes_with(
        &runtime,
        &compact.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(matches!(
        restored_compact.network_reuse_class(false),
        NetworkReuseClass::Compact
    ));
    let actual = restored_compact.diagonal_spectrum().unwrap().unwrap();
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].values[0].to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(actual[1].values[1].to_bits(), 0x7ff8_0000_0000_0007);

    let dense: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [&leg], |_, indices| {
            (1 + indices.iter().sum::<usize>()) as f64
        })
        .unwrap();
    let lazy = dense.adjoint().unwrap();
    let restored_lazy = TensorMap::<SU2FusionRule, f64>::from_bytes_with(
        &runtime,
        &lazy.to_bytes_with(&codec).unwrap(),
        DecodeLimits::default(),
        &codec,
    )
    .unwrap();
    assert!(matches!(
        restored_lazy.network_reuse_class(false),
        NetworkReuseClass::LazyAdjoint
    ));
    assert_eq!(
        restored_lazy
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        lazy.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn malformed_framing_limits_and_provider_failures_are_typed_and_ordered() {
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let bytes = su2_leg(&provider, true).to_bytes_with(&codec).unwrap();
    let decode = |input: &[u8], limits, codec: &Su2Codec| {
        GradedSpace::<SU2FusionRule>::from_bytes_with(input, limits, codec)
    };

    for (offset, value) in [(0, b'X'), (10, 2), (11, 1)] {
        let mut malformed = bytes.clone();
        malformed[offset] = value;
        assert!(matches!(
            decode(&malformed, DecodeLimits::default(), &codec),
            Err(DecodeError::InvalidFormat(_))
        ));
    }
    let mut unsupported = bytes.clone();
    unsupported[8..10].copy_from_slice(&2u16.to_le_bytes());
    assert!(matches!(
        decode(&unsupported, DecodeLimits::default(), &codec),
        Err(DecodeError::UnsupportedVersion {
            actual: 2,
            supported: 1
        })
    ));
    let mut bad_dual = bytes.clone();
    let dual_offset = header_end(&bad_dual);
    bad_dual[dual_offset] = 2;
    assert!(matches!(
        decode(&bad_dual, DecodeLimits::default(), &codec),
        Err(DecodeError::InvalidFormat(_))
    ));
    assert!(matches!(
        decode(&bytes[..bytes.len() - 1], DecodeLimits::default(), &codec),
        Err(DecodeError::InvalidFormat(_))
    ));
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(matches!(
        decode(&trailing, DecodeLimits::default(), &codec),
        Err(DecodeError::InvalidFormat(_))
    ));

    let limits = DecodeLimits {
        max_bytes: bytes.len() - 1,
        ..DecodeLimits::default()
    };
    codec.resolve_calls.store(0, Ordering::Relaxed);
    assert!(matches!(
        decode(&bytes, limits, &codec),
        Err(DecodeError::LimitExceeded { .. })
    ));
    assert_eq!(codec.resolve_calls.load(Ordering::Relaxed), 0);

    let missing = Su2Codec::missing(Arc::clone(&provider));
    let mut bad_magic = bytes.clone();
    bad_magic[0] = 0;
    assert!(matches!(
        decode(&bad_magic, DecodeLimits::default(), &missing),
        Err(DecodeError::InvalidFormat(_))
    ));
    assert_eq!(missing.resolve_calls.load(Ordering::Relaxed), 0);
    assert!(matches!(
        decode(&bytes, DecodeLimits::default(), &missing),
        Err(DecodeError::Codec(CodecError::MissingProvider))
    ));

    let mut mismatched_key = bytes.clone();
    mismatched_key[20..23].copy_from_slice(b"bad");
    let mismatched = Su2Codec::mismatched(Arc::clone(&provider));
    assert!(matches!(
        decode(&mismatched_key, DecodeLimits::default(), &mismatched),
        Err(DecodeError::ProviderMismatch)
    ));
}

#[test]
fn malformed_tensor_tags_and_duplicate_or_missing_blocks_are_rejected() {
    let runtime = runtime();
    let provider = Arc::new(SU2FusionRule);
    let codec = Su2Codec::new(Arc::clone(&provider));
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(SU2Irrep::from_twice_spin(0), 1)],
        false,
    )
    .unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&leg], [], |_, _| 3.0).unwrap();
    let bytes = tensor.to_bytes_with(&codec).unwrap();
    let decode = |input: &[u8]| {
        TensorMap::<SU2FusionRule, f64>::from_bytes_with(
            &runtime,
            input,
            DecodeLimits::default(),
            &codec,
        )
    };

    for (offset, value) in [(10, 1), (11, 2), (header_end(&bytes), 99)] {
        let mut malformed = bytes.clone();
        malformed[offset] = value;
        assert!(matches!(
            decode(&malformed),
            Err(DecodeError::InvalidFormat(_))
        ));
    }

    // Header + repr + ranks + one leg (dual, count, one-byte label, degeneracy).
    let block_count_offset = header_end(&bytes) + 1 + 16 + 1 + 8 + 8 + 1 + 8;
    assert_eq!(u64_at(&bytes, block_count_offset), 1);
    let first_block = block_count_offset + 8;

    let mut missing = bytes[..first_block].to_vec();
    missing[block_count_offset..first_block].copy_from_slice(&0u64.to_le_bytes());
    assert!(matches!(
        decode(&missing),
        Err(DecodeError::InvalidFormat(_))
    ));

    let mut duplicate = bytes.clone();
    duplicate[block_count_offset..first_block].copy_from_slice(&2u64.to_le_bytes());
    duplicate.extend_from_slice(&bytes[first_block..]);
    assert!(matches!(
        decode(&duplicate),
        Err(DecodeError::InvalidFormat(_))
    ));

    for limits in [
        DecodeLimits {
            max_rank: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_sectors_per_leg: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_blocks: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_elements: 0,
            ..DecodeLimits::default()
        },
        DecodeLimits {
            max_blob_bytes: 0,
            ..DecodeLimits::default()
        },
    ] {
        assert!(matches!(
            TensorMap::<SU2FusionRule, f64>::from_bytes_with(&runtime, &bytes, limits, &codec),
            Err(DecodeError::LimitExceeded { .. })
        ));
    }
}
