use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenet::core::{SU2FusionRule, SU2Irrep};
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
    let real: TensorMap<_, f64> =
        TensorMap::from_block_fn(&runtime, [&codomain], [&domain], |trees, indices| {
            f64::from_bits(
                real_bits[(trees.coupled().twice_spin() + indices.iter().sum::<usize>()) % 3],
            )
        })
        .unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_block_fn(&runtime, [&codomain], [&domain], |trees, indices| {
            let index = (trees.coupled().twice_spin() + indices.iter().sum::<usize>()) % 3;
            Complex64::new(
                f64::from_bits(real_bits[index]),
                f64::from_bits(real_bits[(index + 1) % 3]),
            )
        })
        .unwrap();

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

    for (offset, value) in [(0, b'X'), (8, 2), (10, 2), (11, 1)] {
        let mut malformed = bytes.clone();
        malformed[offset] = value;
        assert!(matches!(
            decode(&malformed, DecodeLimits::default(), &codec),
            Err(DecodeError::InvalidFormat(_))
        ));
    }
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
