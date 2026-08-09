//! Stable semantic persistence for the public typed Host representation.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use num_complex::Complex64;
use tenet_core::{FusionTreeKey, MultiplicityIndex, SectorLeg, TypedSectorAdmission};

use super::{
    decode_block_fusion_trees, is_diagonal_bond_space, owned_repr, BlockFusionTrees,
    FusionProductSpace, FusionTreeHomSpace, GradedSpace, Runtime, TensorMap, TensorScalar,
    TypedData, TypedFacadeError, TypedTensorAdjointDispatch, TypedTensorBody,
    TypedTensorModeDispatch, TypedTensorRepr, TypedTensorRootDispatch,
};

const MAGIC: &[u8; 8] = b"TENETTS\0";
const VERSION: u16 = 1;
const KIND_SPACE: u8 = 1;
const KIND_TENSOR: u8 = 2;
const SCALAR_NONE: u8 = 0;
const SCALAR_F64: u8 = 1;
const SCALAR_C64: u8 = 2;
const REPR_DENSE: u8 = 1;
const REPR_DIAGONAL: u8 = 2;
const REPR_ADJOINT: u8 = 3;

/// Caller-owned authority for stable provider keys and semantic sector labels.
///
/// TeNeT never serializes a provider or owns a global provider registry.
/// `resolve_provider` must return the provider allocation authorized by `key`;
/// the decoder re-derives and compares the key before issuing provider queries.
/// Sector encodings must be canonical: equal labels must produce equal bytes,
/// and decoding followed by encoding must reproduce the original bytes.
pub trait TypedPersistenceCodec<R>
where
    R: TypedSectorAdmission,
{
    /// Error produced by provider resolution or semantic-label conversion.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable key naming the provider's full category, codec, and gauge authority.
    fn provider_key(&self, provider: &R) -> Result<Vec<u8>, Self::Error>;

    /// Resolves a stable key to the exact provider allocation used by restoration.
    fn resolve_provider(&self, key: &[u8]) -> Result<Arc<R>, Self::Error>;

    /// Canonically encodes one public semantic sector label.
    fn encode_sector(&self, provider: &R, sector: &R::Sector) -> Result<Vec<u8>, Self::Error>;

    /// Decodes one public semantic sector label for `provider`.
    fn decode_sector(&self, provider: &R, bytes: &[u8]) -> Result<R::Sector, Self::Error>;
}

/// Resource ceilings applied before untrusted lengths drive allocation or admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeLimits {
    /// Maximum input byte length.
    pub max_bytes: usize,
    /// Maximum codomain-plus-domain rank.
    pub max_rank: usize,
    /// Maximum nonzero sectors on one leg.
    pub max_sectors_per_leg: usize,
    /// Maximum semantic blocks in one dense payload.
    pub max_blocks: usize,
    /// Maximum scalar values across one tensor payload.
    pub max_elements: usize,
    /// Maximum provider-key or sector-label byte length.
    pub max_blob_bytes: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 1 << 30,
            max_rank: 64,
            max_sectors_per_leg: 1 << 20,
            max_blocks: 1 << 20,
            max_elements: 1 << 32,
            max_blob_bytes: 1 << 20,
        }
    }
}

/// Failure while producing a typed semantic snapshot.
#[derive(Debug)]
pub enum EncodeError<C, P> {
    /// Provider-key or semantic-label codec failure.
    Codec(C),
    /// Provider label readback failure.
    Provider(P),
    /// The live tensor violates an admitted representation invariant.
    InvalidState(String),
    /// A host length cannot be represented by the v1 `u64` wire field.
    LengthOverflow,
}

impl<C: fmt::Display, P: fmt::Display> fmt::Display for EncodeError<C, P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(formatter, "persistence codec error: {error}"),
            Self::Provider(error) => write!(formatter, "provider label error: {error}"),
            Self::InvalidState(message) => write!(formatter, "invalid tensor state: {message}"),
            Self::LengthOverflow => formatter.write_str("length does not fit the v1 wire format"),
        }
    }
}

impl<C, P> std::error::Error for EncodeError<C, P>
where
    C: std::error::Error + 'static,
    P: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            Self::Provider(error) => Some(error),
            Self::InvalidState(_) | Self::LengthOverflow => None,
        }
    }
}

/// Failure while parsing, resolving, admitting, or reconstructing a snapshot.
#[derive(Debug)]
pub enum DecodeError<C, F> {
    /// Invalid or unsupported v1 framing or semantic payload.
    InvalidFormat(String),
    /// A declared resource exceeds the caller's decode ceiling.
    LimitExceeded {
        /// Resource being limited.
        resource: &'static str,
        /// Declared or accumulated amount.
        actual: usize,
        /// Accepted maximum.
        limit: usize,
    },
    /// Provider resolution or semantic-label codec failure.
    Codec(C),
    /// The resolver returned a provider whose stable key differs from the file.
    ProviderMismatch,
    /// Typed provider/layout admission failure.
    Facade(F),
}

impl<C: fmt::Display, F: fmt::Display> fmt::Display for DecodeError<C, F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(message) => write!(formatter, "invalid typed snapshot: {message}"),
            Self::LimitExceeded {
                resource,
                actual,
                limit,
            } => write!(
                formatter,
                "typed snapshot {resource} {actual} exceeds decode limit {limit}"
            ),
            Self::Codec(error) => write!(formatter, "persistence codec error: {error}"),
            Self::ProviderMismatch => formatter.write_str("resolved provider key does not match"),
            Self::Facade(error) => write!(formatter, "typed admission error: {error}"),
        }
    }
}

impl<C, F> std::error::Error for DecodeError<C, F>
where
    C: std::error::Error + 'static,
    F: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            Self::Facade(error) => Some(error),
            Self::InvalidFormat(_) | Self::LimitExceeded { .. } | Self::ProviderMismatch => None,
        }
    }
}

trait WireScalar: TensorScalar + Copy {
    const TAG: u8;
    const WIDTH: usize;

    fn write(self, output: &mut Vec<u8>);
    fn read(reader: &mut Reader<'_>) -> Result<Self, String>;
}

impl WireScalar for f64 {
    const TAG: u8 = SCALAR_F64;
    const WIDTH: usize = 8;

    fn write(self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.to_bits().to_le_bytes());
    }

    fn read(reader: &mut Reader<'_>) -> Result<Self, String> {
        Ok(Self::from_bits(reader.u64()?))
    }
}

impl WireScalar for Complex64 {
    const TAG: u8 = SCALAR_C64;
    const WIDTH: usize = 16;

    fn write(self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.re.to_bits().to_le_bytes());
        output.extend_from_slice(&self.im.to_bits().to_le_bytes());
    }

    fn read(reader: &mut Reader<'_>) -> Result<Self, String> {
        Ok(Self::new(
            f64::from_bits(reader.u64()?),
            f64::from_bits(reader.u64()?),
        ))
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| "byte offset overflow".to_string())?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| "truncated input".to_string())?;
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn usize(&mut self, resource: &'static str) -> Result<usize, String> {
        usize::try_from(self.u64()?)
            .map_err(|_| format!("{resource} does not fit this target's usize"))
    }

    fn finish(self) -> Result<(), String> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err("trailing bytes".to_string())
        }
    }
}

fn put_len(output: &mut Vec<u8>, len: usize) -> Result<(), EncodeError<(), ()>> {
    let len = u64::try_from(len).map_err(|_| EncodeError::LengthOverflow)?;
    output.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn put_blob(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), EncodeError<(), ()>> {
    put_len(output, bytes.len())?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn map_length_error<C, P>(error: EncodeError<(), ()>) -> EncodeError<C, P> {
    match error {
        EncodeError::LengthOverflow => EncodeError::LengthOverflow,
        _ => unreachable!(),
    }
}

fn check_limit<C, F>(
    resource: &'static str,
    actual: usize,
    limit: usize,
) -> Result<(), DecodeError<C, F>> {
    if actual > limit {
        Err(DecodeError::LimitExceeded {
            resource,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn read_blob<'a, C, F>(
    reader: &mut Reader<'a>,
    limits: DecodeLimits,
    resource: &'static str,
) -> Result<&'a [u8], DecodeError<C, F>> {
    let len = reader.usize(resource).map_err(DecodeError::InvalidFormat)?;
    check_limit(resource, len, limits.max_blob_bytes)?;
    reader.take(len).map_err(DecodeError::InvalidFormat)
}

fn put_header<C, P>(
    output: &mut Vec<u8>,
    kind: u8,
    scalar: u8,
    provider_key: &[u8],
) -> Result<(), EncodeError<C, P>> {
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&VERSION.to_le_bytes());
    output.push(kind);
    output.push(scalar);
    put_blob(output, provider_key).map_err(map_length_error)
}

fn read_header<'a, C, F>(
    bytes: &'a [u8],
    limits: DecodeLimits,
    expected_kind: u8,
    expected_scalar: u8,
) -> Result<(Reader<'a>, &'a [u8]), DecodeError<C, F>> {
    check_limit("input bytes", bytes.len(), limits.max_bytes)?;
    let mut reader = Reader::new(bytes);
    if reader
        .take(MAGIC.len())
        .map_err(DecodeError::InvalidFormat)?
        != MAGIC
    {
        return Err(DecodeError::InvalidFormat("bad magic".to_string()));
    }
    let version = reader.u16().map_err(DecodeError::InvalidFormat)?;
    if version != VERSION {
        return Err(DecodeError::InvalidFormat(format!(
            "unsupported format version {version}"
        )));
    }
    let kind = reader.u8().map_err(DecodeError::InvalidFormat)?;
    if kind != expected_kind {
        return Err(DecodeError::InvalidFormat("wrong object kind".to_string()));
    }
    let scalar = reader.u8().map_err(DecodeError::InvalidFormat)?;
    if scalar != expected_scalar {
        return Err(DecodeError::InvalidFormat("wrong scalar kind".to_string()));
    }
    let provider_key = read_blob(&mut reader, limits, "provider key bytes")?;
    Ok((reader, provider_key))
}

fn resolve_provider<R, C, F>(codec: &C, key: &[u8]) -> Result<Arc<R>, DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    let provider = codec.resolve_provider(key).map_err(DecodeError::Codec)?;
    let actual = codec
        .provider_key(provider.as_ref())
        .map_err(DecodeError::Codec)?;
    if actual.as_slice() != key {
        return Err(DecodeError::ProviderMismatch);
    }
    Ok(provider)
}

// This pass is intentionally provider-free. It ensures malformed framing and
// declared resource excess cannot trigger resolver or category-provider work.
fn preflight_leg<C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
) -> Result<(), DecodeError<C, F>> {
    match reader.u8().map_err(DecodeError::InvalidFormat)? {
        0 | 1 => {}
        _ => {
            return Err(DecodeError::InvalidFormat(
                "duality flag is not 0 or 1".to_string(),
            ));
        }
    }
    let count = reader
        .usize("sectors per leg")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("sectors per leg", count, limits.max_sectors_per_leg)?;
    for _ in 0..count {
        read_blob(reader, limits, "sector label bytes")?;
        if reader
            .usize("sector degeneracy")
            .map_err(DecodeError::InvalidFormat)?
            == 0
        {
            return Err(DecodeError::InvalidFormat(
                "zero-degeneracy sector is not canonical".to_string(),
            ));
        }
    }
    Ok(())
}

fn preflight_space<C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
) -> Result<usize, DecodeError<C, F>> {
    let nout = reader
        .usize("codomain rank")
        .map_err(DecodeError::InvalidFormat)?;
    let nin = reader
        .usize("domain rank")
        .map_err(DecodeError::InvalidFormat)?;
    let rank = nout
        .checked_add(nin)
        .ok_or_else(|| DecodeError::InvalidFormat("rank overflow".to_string()))?;
    check_limit("rank", rank, limits.max_rank)?;
    for _ in 0..rank {
        preflight_leg(reader, limits)?;
    }
    Ok(rank)
}

fn preflight_tree<'a, C, F>(
    reader: &mut Reader<'a>,
    limits: DecodeLimits,
) -> Result<&'a [u8], DecodeError<C, F>> {
    let coupled = read_blob(reader, limits, "sector label bytes")?;
    for _ in 0..2 {
        let count = reader
            .usize("fusion-tree labels")
            .map_err(DecodeError::InvalidFormat)?;
        check_limit("fusion-tree labels", count, limits.max_rank)?;
        for _ in 0..count {
            read_blob(reader, limits, "sector label bytes")?;
        }
    }
    let vertices = reader
        .usize("fusion vertices")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("fusion vertices", vertices, limits.max_rank)?;
    for _ in 0..vertices {
        if reader.u64().map_err(DecodeError::InvalidFormat)? == 0 {
            return Err(DecodeError::InvalidFormat(
                "fusion vertex labels are one-based".to_string(),
            ));
        }
    }
    Ok(coupled)
}

fn preflight_block_key<C, F>(bytes: &[u8], limits: DecodeLimits) -> Result<(), DecodeError<C, F>> {
    let mut reader = Reader::new(bytes);
    let codomain = preflight_tree(&mut reader, limits)?;
    let domain = preflight_tree(&mut reader, limits)?;
    if codomain != domain {
        return Err(DecodeError::InvalidFormat(
            "fusion-tree pair has unequal encoded coupled sectors".to_string(),
        ));
    }
    reader.finish().map_err(DecodeError::InvalidFormat)
}

fn preflight_values<D, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    total: &mut usize,
) -> Result<usize, DecodeError<C, F>>
where
    D: WireScalar,
{
    let count = reader
        .usize("payload elements")
        .map_err(DecodeError::InvalidFormat)?;
    *total = total
        .checked_add(count)
        .ok_or_else(|| DecodeError::InvalidFormat("payload element count overflow".to_string()))?;
    check_limit("payload elements", *total, limits.max_elements)?;
    let bytes = count
        .checked_mul(D::WIDTH)
        .ok_or_else(|| DecodeError::InvalidFormat("payload byte count overflow".to_string()))?;
    reader.take(bytes).map_err(DecodeError::InvalidFormat)?;
    Ok(count)
}

fn preflight_dense<D, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
) -> Result<(), DecodeError<C, F>>
where
    D: WireScalar,
{
    let rank = preflight_space(reader, limits)?;
    let blocks = reader
        .usize("dense blocks")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("dense blocks", blocks, limits.max_blocks)?;
    let mut total = 0usize;
    for _ in 0..blocks {
        let key = read_blob(reader, limits, "block key bytes")?;
        preflight_block_key(key, limits)?;
        let shape_rank = reader
            .usize("block rank")
            .map_err(DecodeError::InvalidFormat)?;
        if shape_rank != rank {
            return Err(DecodeError::InvalidFormat(
                "block shape rank does not match tensor rank".to_string(),
            ));
        }
        let mut expected = 1usize;
        for _ in 0..shape_rank {
            let dimension = reader
                .usize("block dimension")
                .map_err(DecodeError::InvalidFormat)?;
            expected = expected.checked_mul(dimension).ok_or_else(|| {
                DecodeError::InvalidFormat("block element count overflow".to_string())
            })?;
        }
        if preflight_values::<D, C, F>(reader, limits, &mut total)? != expected {
            return Err(DecodeError::InvalidFormat(
                "block value count does not match shape".to_string(),
            ));
        }
    }
    Ok(())
}

fn preflight_diagonal<D, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
) -> Result<(), DecodeError<C, F>>
where
    D: WireScalar,
{
    preflight_space(reader, limits)?;
    let count = reader
        .usize("diagonal sectors")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("diagonal sectors", count, limits.max_sectors_per_leg)?;
    let mut total = 0usize;
    for _ in 0..count {
        read_blob(reader, limits, "sector label bytes")?;
        preflight_values::<D, C, F>(reader, limits, &mut total)?;
    }
    Ok(())
}

fn preflight_tensor<D, C, F>(bytes: &[u8], limits: DecodeLimits) -> Result<(), DecodeError<C, F>>
where
    D: WireScalar,
{
    let (mut reader, _) = read_header(bytes, limits, KIND_TENSOR, D::TAG)?;
    match reader.u8().map_err(DecodeError::InvalidFormat)? {
        REPR_DENSE => preflight_dense::<D, C, F>(&mut reader, limits)?,
        REPR_DIAGONAL => preflight_diagonal::<D, C, F>(&mut reader, limits)?,
        REPR_ADJOINT => match reader.u8().map_err(DecodeError::InvalidFormat)? {
            REPR_DENSE => preflight_dense::<D, C, F>(&mut reader, limits)?,
            REPR_DIAGONAL => preflight_diagonal::<D, C, F>(&mut reader, limits)?,
            _ => {
                return Err(DecodeError::InvalidFormat(
                    "unknown lazy-adjoint parent representation".to_string(),
                ));
            }
        },
        _ => {
            return Err(DecodeError::InvalidFormat(
                "unknown tensor representation".to_string(),
            ));
        }
    }
    reader.finish().map_err(DecodeError::InvalidFormat)
}

fn preflight_graded_space<C, F>(
    bytes: &[u8],
    limits: DecodeLimits,
) -> Result<(), DecodeError<C, F>> {
    let (mut reader, _) = read_header(bytes, limits, KIND_SPACE, SCALAR_NONE)?;
    preflight_leg(&mut reader, limits)?;
    reader.finish().map_err(DecodeError::InvalidFormat)
}

#[derive(Clone)]
struct LegRecord<S> {
    dual: bool,
    sectors: Vec<(S, usize)>,
}

#[derive(Clone)]
struct SpaceRecord<S> {
    codomain: Vec<LegRecord<S>>,
    domain: Vec<LegRecord<S>>,
}

struct BlockRecord<S, D> {
    key: BlockFusionTrees<S>,
    shape: Vec<usize>,
    values: Vec<D>,
}

enum TensorRecord<S, D> {
    Dense {
        space: SpaceRecord<S>,
        blocks: Vec<BlockRecord<S, D>>,
    },
    Diagonal {
        space: SpaceRecord<S>,
        spectrum: Vec<(S, Vec<D>)>,
    },
    AdjointDense {
        parent_space: SpaceRecord<S>,
        parent_blocks: Vec<BlockRecord<S, D>>,
    },
    AdjointDiagonal {
        parent_space: SpaceRecord<S>,
        parent_spectrum: Vec<(S, Vec<D>)>,
    },
}

fn encode_label<R, C>(
    output: &mut Vec<u8>,
    codec: &C,
    provider: &R,
    sector: &R::Sector,
) -> Result<(), EncodeError<C::Error, R::Error>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    let bytes = codec
        .encode_sector(provider, sector)
        .map_err(EncodeError::Codec)?;
    put_blob(output, &bytes).map_err(map_length_error)
}

fn encode_leg<R, C>(
    output: &mut Vec<u8>,
    codec: &C,
    provider: &R,
    leg: &SectorLeg,
) -> Result<(), EncodeError<C::Error, R::Error>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    output.push(u8::from(leg.is_dual()));
    let mut entries = leg
        .sectors()
        .iter()
        .copied()
        .zip(leg.degeneracies().iter().copied())
        .map(|(id, degeneracy)| {
            let label = R::try_decode_label(provider, id).map_err(EncodeError::Provider)?;
            let bytes = codec
                .encode_sector(provider, &label)
                .map_err(EncodeError::Codec)?;
            Ok((bytes, degeneracy))
        })
        .collect::<Result<Vec<_>, EncodeError<C::Error, R::Error>>>()?;
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    put_len(output, entries.len()).map_err(map_length_error)?;
    for (label, degeneracy) in entries {
        put_blob(output, &label).map_err(map_length_error)?;
        output.extend_from_slice(
            &u64::try_from(degeneracy)
                .map_err(|_| EncodeError::LengthOverflow)?
                .to_le_bytes(),
        );
    }
    Ok(())
}

fn encode_space<R, C>(
    output: &mut Vec<u8>,
    codec: &C,
    provider: &R,
    homspace: &FusionTreeHomSpace,
) -> Result<(), EncodeError<C::Error, R::Error>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    put_len(output, homspace.codomain().len()).map_err(map_length_error)?;
    put_len(output, homspace.domain().len()).map_err(map_length_error)?;
    for leg in homspace
        .codomain()
        .legs()
        .iter()
        .chain(homspace.domain().legs())
    {
        encode_leg(output, codec, provider, leg)?;
    }
    Ok(())
}

fn encode_tree<R, C>(
    output: &mut Vec<u8>,
    codec: &C,
    provider: &R,
    tree: &FusionTreeKey,
) -> Result<(), EncodeError<C::Error, R::Error>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    let coupled = R::try_decode_label(provider, tree.coupled()).map_err(EncodeError::Provider)?;
    encode_label(output, codec, provider, &coupled)?;
    for sectors in [tree.uncoupled(), tree.innerlines()] {
        put_len(output, sectors.len()).map_err(map_length_error)?;
        for &sector in sectors {
            let label = R::try_decode_label(provider, sector).map_err(EncodeError::Provider)?;
            encode_label(output, codec, provider, &label)?;
        }
    }
    put_len(output, tree.vertices().len()).map_err(map_length_error)?;
    for vertex in tree.vertices() {
        output.extend_from_slice(
            &u64::try_from(vertex.get())
                .map_err(|_| EncodeError::LengthOverflow)?
                .to_le_bytes(),
        );
    }
    Ok(())
}

fn encode_block_key<R, C>(
    output: &mut Vec<u8>,
    codec: &C,
    provider: &R,
    key: &tenet_core::FusionTreePairKey,
) -> Result<(), EncodeError<C::Error, R::Error>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    encode_tree(output, codec, provider, key.codomain_tree())?;
    encode_tree(output, codec, provider, key.domain_tree())
}

fn checked_shape_len<C, P>(shape: &[usize]) -> Result<usize, EncodeError<C, P>> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| EncodeError::InvalidState("block element count overflow".to_string()))
    })
}

fn logical_values<D: Copy, C, P>(
    data: &[D],
    block: &tenet_core::BlockRef<'_>,
) -> Result<Vec<D>, EncodeError<C, P>> {
    let count = checked_shape_len(block.shape())?;
    let mut values = Vec::with_capacity(count);
    for linear in 0..count {
        let mut residual = linear;
        let mut position = block.offset();
        for (&dimension, &stride) in block.shape().iter().zip(block.strides()) {
            if dimension != 0 {
                position = position
                    .checked_add((residual % dimension).checked_mul(stride).ok_or_else(|| {
                        EncodeError::InvalidState("block offset overflow".to_string())
                    })?)
                    .ok_or_else(|| {
                        EncodeError::InvalidState("block offset overflow".to_string())
                    })?;
                residual /= dimension;
            }
        }
        values.push(*data.get(position).ok_or_else(|| {
            EncodeError::InvalidState("block addresses outside storage".to_string())
        })?);
    }
    Ok(values)
}

fn encode_dense_body<R, D, C>(
    output: &mut Vec<u8>,
    codec: &C,
    body: &TypedTensorBody<R, D>,
) -> Result<(), EncodeError<C::Error, R::Error>>
where
    R: TypedSectorAdmission,
    D: WireScalar,
    C: TypedPersistenceCodec<R>,
{
    let TypedData::Dense(data) = body.data.as_ref() else {
        return Err(EncodeError::InvalidState(
            "dense representation contains compact payload".to_string(),
        ));
    };
    let space = body.space.space();
    if space
        .required_len()
        .map_err(|error| EncodeError::InvalidState(format!("invalid admitted layout: {error}")))?
        != data.len()
    {
        return Err(EncodeError::InvalidState(
            "dense storage length does not match admitted layout".to_string(),
        ));
    }
    encode_space(output, codec, body.space.provider(), space.homspace())?;
    let mut blocks = Vec::with_capacity(space.structure().block_count());
    for index in 0..space.structure().block_count() {
        let block = space.structure().block(index).map_err(|error| {
            EncodeError::InvalidState(format!("invalid admitted block: {error}"))
        })?;
        let key = block.key().as_fusion_tree_pair().ok_or_else(|| {
            EncodeError::InvalidState("typed block is not a fusion-tree pair".to_string())
        })?;
        let mut encoded_key = Vec::new();
        encode_block_key(&mut encoded_key, codec, body.space.provider(), key)?;
        blocks.push((encoded_key, block));
    }
    blocks.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    put_len(output, blocks.len()).map_err(map_length_error)?;
    for (key, block) in blocks {
        put_blob(output, &key).map_err(map_length_error)?;
        put_len(output, block.shape().len()).map_err(map_length_error)?;
        for &dimension in block.shape() {
            output.extend_from_slice(
                &u64::try_from(dimension)
                    .map_err(|_| EncodeError::LengthOverflow)?
                    .to_le_bytes(),
            );
        }
        let values = logical_values(data, &block)?;
        put_len(output, values.len()).map_err(map_length_error)?;
        for value in values {
            value.write(output);
        }
    }
    Ok(())
}

fn encode_diagonal_body<R, D, C>(
    output: &mut Vec<u8>,
    codec: &C,
    body: &TypedTensorBody<R, D>,
    spectrum: &[tenet_matrixalgebra::SectorSpectrum<D>],
) -> Result<(), EncodeError<C::Error, R::Error>>
where
    R: TypedSectorAdmission,
    D: WireScalar,
    C: TypedPersistenceCodec<R>,
{
    encode_space(
        output,
        codec,
        body.space.provider(),
        body.space.space().homspace(),
    )?;
    let mut entries = spectrum
        .iter()
        .map(|entry| {
            let label = R::try_decode_label(body.space.provider(), entry.sector)
                .map_err(EncodeError::Provider)?;
            let key = codec
                .encode_sector(body.space.provider(), &label)
                .map_err(EncodeError::Codec)?;
            Ok((key, entry.values.as_slice()))
        })
        .collect::<Result<Vec<_>, EncodeError<C::Error, R::Error>>>()?;
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    put_len(output, entries.len()).map_err(map_length_error)?;
    for (label, values) in entries {
        put_blob(output, &label).map_err(map_length_error)?;
        put_len(output, values.len()).map_err(map_length_error)?;
        for &value in values {
            value.write(output);
        }
    }
    Ok(())
}

fn encode_tensor<R, D, C>(
    tensor: &TensorMap<R, D>,
    codec: &C,
) -> Result<Vec<u8>, EncodeError<C::Error, R::Error>>
where
    R: TypedSectorAdmission,
    D: WireScalar,
    C: TypedPersistenceCodec<R>,
{
    let provider_key = codec
        .provider_key(tensor.provider())
        .map_err(EncodeError::Codec)?;
    let mut output = Vec::new();
    put_header(&mut output, KIND_TENSOR, D::TAG, &provider_key)?;
    match &tensor.repr {
        TypedTensorRepr::Owned(body) => match body.data.as_ref() {
            TypedData::Dense(_) => {
                output.push(REPR_DENSE);
                encode_dense_body(&mut output, codec, body)?;
            }
            TypedData::Diagonal(spectrum) => {
                output.push(REPR_DIAGONAL);
                encode_diagonal_body(&mut output, codec, body, spectrum)?;
            }
        },
        TypedTensorRepr::Adjoint(view) => {
            output.push(REPR_ADJOINT);
            // Why not serialize the logical cache: it is derived state and may be warm.
            match view.parent.data.as_ref() {
                TypedData::Dense(_) => {
                    output.push(REPR_DENSE);
                    encode_dense_body(&mut output, codec, &view.parent)?;
                }
                TypedData::Diagonal(spectrum) => {
                    output.push(REPR_DIAGONAL);
                    encode_diagonal_body(&mut output, codec, &view.parent, spectrum)?;
                }
            }
        }
    }
    Ok(output)
}

fn read_label<R, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<R::Sector, DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    let bytes = read_blob(reader, limits, "sector label bytes")?;
    let label = codec
        .decode_sector(provider, bytes)
        .map_err(DecodeError::Codec)?;
    let canonical = codec
        .encode_sector(provider, &label)
        .map_err(DecodeError::Codec)?;
    if canonical.as_slice() != bytes {
        return Err(DecodeError::InvalidFormat(
            "sector label is not canonically encoded".to_string(),
        ));
    }
    Ok(label)
}

fn read_leg<R, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<LegRecord<R::Sector>, DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    let dual = match reader.u8().map_err(DecodeError::InvalidFormat)? {
        0 => false,
        1 => true,
        _ => {
            return Err(DecodeError::InvalidFormat(
                "duality flag is not 0 or 1".to_string(),
            ))
        }
    };
    let count = reader
        .usize("sectors per leg")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("sectors per leg", count, limits.max_sectors_per_leg)?;
    let mut sectors = Vec::with_capacity(count);
    for _ in 0..count {
        let label = read_label(reader, limits, codec, provider)?;
        let degeneracy = reader
            .usize("sector degeneracy")
            .map_err(DecodeError::InvalidFormat)?;
        if degeneracy == 0 {
            return Err(DecodeError::InvalidFormat(
                "zero-degeneracy sector is not canonical".to_string(),
            ));
        }
        sectors.push((label, degeneracy));
    }
    sectors.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    if sectors.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(DecodeError::InvalidFormat(
            "duplicate sector label on one leg".to_string(),
        ));
    }
    Ok(LegRecord { dual, sectors })
}

fn read_space<R, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<SpaceRecord<R::Sector>, DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    let nout = reader
        .usize("codomain rank")
        .map_err(DecodeError::InvalidFormat)?;
    let nin = reader
        .usize("domain rank")
        .map_err(DecodeError::InvalidFormat)?;
    let rank = nout
        .checked_add(nin)
        .ok_or_else(|| DecodeError::InvalidFormat("rank overflow".to_string()))?;
    check_limit("rank", rank, limits.max_rank)?;
    let mut legs = Vec::with_capacity(rank);
    for _ in 0..rank {
        legs.push(read_leg(reader, limits, codec, provider)?);
    }
    let domain = legs.split_off(nout);
    Ok(SpaceRecord {
        codomain: legs,
        domain,
    })
}

fn read_label_vec<R, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<Vec<R::Sector>, DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    let count = reader
        .usize("fusion-tree labels")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("fusion-tree labels", count, limits.max_rank)?;
    (0..count)
        .map(|_| read_label(reader, limits, codec, provider))
        .collect()
}

fn read_vertices<C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
) -> Result<Vec<MultiplicityIndex>, DecodeError<C, F>> {
    let count = reader
        .usize("fusion vertices")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("fusion vertices", count, limits.max_rank)?;
    (0..count)
        .map(|_| {
            let value = reader
                .usize("fusion vertex")
                .map_err(DecodeError::InvalidFormat)?;
            MultiplicityIndex::new(value).ok_or_else(|| {
                DecodeError::InvalidFormat("fusion vertex labels are one-based".to_string())
            })
        })
        .collect()
}

struct TreeRecord<S> {
    coupled: S,
    uncoupled: Vec<S>,
    innerlines: Vec<S>,
    vertices: Vec<MultiplicityIndex>,
}

fn read_tree<R, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<TreeRecord<R::Sector>, DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    Ok(TreeRecord {
        coupled: read_label(reader, limits, codec, provider)?,
        uncoupled: read_label_vec(reader, limits, codec, provider)?,
        innerlines: read_label_vec(reader, limits, codec, provider)?,
        vertices: read_vertices(reader, limits)?,
    })
}

fn read_block_key<R, C, F>(
    bytes: &[u8],
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<BlockFusionTrees<R::Sector>, DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    C: TypedPersistenceCodec<R>,
{
    let mut reader = Reader::new(bytes);
    let codomain = read_tree(&mut reader, limits, codec, provider)?;
    let domain = read_tree(&mut reader, limits, codec, provider)?;
    reader.finish().map_err(DecodeError::InvalidFormat)?;
    if codomain.coupled != domain.coupled {
        return Err(DecodeError::InvalidFormat(
            "fusion-tree pair has unequal coupled sectors".to_string(),
        ));
    }
    Ok(BlockFusionTrees {
        coupled: codomain.coupled,
        codomain_uncoupled: codomain.uncoupled,
        codomain_innerlines: codomain.innerlines,
        codomain_vertices: codomain.vertices,
        domain_uncoupled: domain.uncoupled,
        domain_innerlines: domain.innerlines,
        domain_vertices: domain.vertices,
    })
}

fn read_values<D, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    total: &mut usize,
) -> Result<Vec<D>, DecodeError<C, F>>
where
    D: WireScalar,
{
    let count = reader
        .usize("payload elements")
        .map_err(DecodeError::InvalidFormat)?;
    *total = total
        .checked_add(count)
        .ok_or_else(|| DecodeError::InvalidFormat("payload element count overflow".to_string()))?;
    check_limit("payload elements", *total, limits.max_elements)?;
    let byte_count = count
        .checked_mul(D::WIDTH)
        .ok_or_else(|| DecodeError::InvalidFormat("payload byte count overflow".to_string()))?;
    if byte_count > reader.bytes.len().saturating_sub(reader.position) {
        return Err(DecodeError::InvalidFormat("truncated input".to_string()));
    }
    (0..count)
        .map(|_| D::read(reader).map_err(DecodeError::InvalidFormat))
        .collect()
}

fn read_dense<R, D, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<(SpaceRecord<R::Sector>, Vec<BlockRecord<R::Sector, D>>), DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    D: WireScalar,
    C: TypedPersistenceCodec<R>,
{
    let space = read_space(reader, limits, codec, provider)?;
    let rank = space.codomain.len() + space.domain.len();
    let count = reader
        .usize("dense blocks")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("dense blocks", count, limits.max_blocks)?;
    let mut total = 0usize;
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        let key_bytes = read_blob(reader, limits, "block key bytes")?;
        let key = read_block_key(key_bytes, limits, codec, provider)?;
        let shape_rank = reader
            .usize("block rank")
            .map_err(DecodeError::InvalidFormat)?;
        if shape_rank != rank {
            return Err(DecodeError::InvalidFormat(
                "block shape rank does not match tensor rank".to_string(),
            ));
        }
        let shape = (0..shape_rank)
            .map(|_| {
                reader
                    .usize("block dimension")
                    .map_err(DecodeError::InvalidFormat)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let values = read_values(reader, limits, &mut total)?;
        let expected = shape.iter().try_fold(1usize, |count, dimension| {
            count.checked_mul(*dimension).ok_or_else(|| {
                DecodeError::InvalidFormat("block element count overflow".to_string())
            })
        })?;
        if values.len() != expected {
            return Err(DecodeError::InvalidFormat(
                "block value count does not match shape".to_string(),
            ));
        }
        blocks.push(BlockRecord { key, shape, values });
    }
    Ok((space, blocks))
}

fn read_diagonal<R, D, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<(SpaceRecord<R::Sector>, Vec<(R::Sector, Vec<D>)>), DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    D: WireScalar,
    C: TypedPersistenceCodec<R>,
{
    let space = read_space(reader, limits, codec, provider)?;
    let count = reader
        .usize("diagonal sectors")
        .map_err(DecodeError::InvalidFormat)?;
    check_limit("diagonal sectors", count, limits.max_sectors_per_leg)?;
    let mut total = 0usize;
    let mut spectrum = Vec::with_capacity(count);
    for _ in 0..count {
        let label = read_label(reader, limits, codec, provider)?;
        let values = read_values(reader, limits, &mut total)?;
        spectrum.push((label, values));
    }
    Ok((space, spectrum))
}

fn read_tensor_record<R, D, C, F>(
    reader: &mut Reader<'_>,
    limits: DecodeLimits,
    codec: &C,
    provider: &R,
) -> Result<TensorRecord<R::Sector, D>, DecodeError<C::Error, F>>
where
    R: TypedSectorAdmission,
    D: WireScalar,
    C: TypedPersistenceCodec<R>,
{
    match reader.u8().map_err(DecodeError::InvalidFormat)? {
        REPR_DENSE => {
            let (space, blocks) = read_dense(reader, limits, codec, provider)?;
            Ok(TensorRecord::Dense { space, blocks })
        }
        REPR_DIAGONAL => {
            let (space, spectrum) = read_diagonal(reader, limits, codec, provider)?;
            Ok(TensorRecord::Diagonal { space, spectrum })
        }
        REPR_ADJOINT => match reader.u8().map_err(DecodeError::InvalidFormat)? {
            REPR_DENSE => {
                let (parent_space, parent_blocks) = read_dense(reader, limits, codec, provider)?;
                Ok(TensorRecord::AdjointDense {
                    parent_space,
                    parent_blocks,
                })
            }
            REPR_DIAGONAL => {
                let (parent_space, parent_spectrum) =
                    read_diagonal(reader, limits, codec, provider)?;
                Ok(TensorRecord::AdjointDiagonal {
                    parent_space,
                    parent_spectrum,
                })
            }
            _ => Err(DecodeError::InvalidFormat(
                "unknown lazy-adjoint parent representation".to_string(),
            )),
        },
        _ => Err(DecodeError::InvalidFormat(
            "unknown tensor representation".to_string(),
        )),
    }
}

fn build_leg<R>(
    provider: Arc<R>,
    record: LegRecord<R::Sector>,
) -> Result<GradedSpace<R>, TypedFacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    let mut encoded = Vec::with_capacity(record.sectors.len());
    for (label, degeneracy) in record.sectors {
        let id = R::try_encode_label(provider.as_ref(), &label)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        encoded.push((id, degeneracy));
    }
    encoded.sort_unstable_by_key(|entry| entry.0);
    if encoded.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(super::Error::InvalidArgument(
            "persistence codec labels alias one provider sector".to_string(),
        )
        .into());
    }
    let leg = SectorLeg::try_new(encoded, record.dual).map_err(|error| {
        TypedFacadeError::<R>::from(super::Error::InvalidArgument(error.to_string()))
    })?;
    Ok(GradedSpace { provider, leg })
}

fn build_legs<R>(
    provider: &Arc<R>,
    records: Vec<LegRecord<R::Sector>>,
) -> Result<Vec<GradedSpace<R>>, TypedFacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    records
        .into_iter()
        .map(|record| build_leg(Arc::clone(provider), record))
        .collect()
}

fn build_space<R>(
    provider: &Arc<R>,
    record: SpaceRecord<R::Sector>,
) -> Result<super::BoundDynamicFusionMapSpace<R>, TypedFacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
{
    let codomain = build_legs(provider, record.codomain)?;
    let domain = build_legs(provider, record.domain)?;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(codomain.iter().map(|leg| leg.leg.clone())),
        FusionProductSpace::new(domain.iter().map(|leg| leg.leg.clone())),
    );
    <R::Mode as TypedTensorRootDispatch<R>>::build_root(Arc::clone(provider), homspace)
}

fn copy_logical_values<D: Copy, C, F>(
    data: &mut [D],
    block: &tenet_core::BlockRef<'_>,
    values: &[D],
) -> Result<(), DecodeError<C, F>> {
    for (linear, &value) in values.iter().enumerate() {
        let mut residual = linear;
        let mut position = block.offset();
        for (&dimension, &stride) in block.shape().iter().zip(block.strides()) {
            if dimension != 0 {
                position = position
                    .checked_add((residual % dimension).checked_mul(stride).ok_or_else(|| {
                        DecodeError::InvalidFormat("block offset overflow".to_string())
                    })?)
                    .ok_or_else(|| {
                        DecodeError::InvalidFormat("block offset overflow".to_string())
                    })?;
                residual /= dimension;
            }
        }
        *data.get_mut(position).ok_or_else(|| {
            DecodeError::InvalidFormat("block addresses outside storage".to_string())
        })? = value;
    }
    Ok(())
}

fn build_dense<R, D, C>(
    runtime: &Runtime,
    provider: &Arc<R>,
    record: SpaceRecord<R::Sector>,
    blocks: Vec<BlockRecord<R::Sector, D>>,
) -> Result<TensorMap<R, D>, DecodeError<C, TypedFacadeError<R>>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: WireScalar,
{
    let space = build_space(provider, record).map_err(DecodeError::Facade)?;
    let mut supplied = HashMap::with_capacity(blocks.len());
    for block in blocks {
        if supplied.insert(block.key.clone(), block).is_some() {
            return Err(DecodeError::InvalidFormat(
                "duplicate semantic block key".to_string(),
            ));
        }
    }
    let structure = space.space().structure();
    let required = space
        .space()
        .required_len()
        .map_err(|error| DecodeError::InvalidFormat(error.to_string()))?;
    let mut data = vec![D::from_real(0.0); required];
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(|error| DecodeError::InvalidFormat(error.to_string()))?;
        let key = decode_block_fusion_trees(provider.as_ref(), block.key())
            .map_err(DecodeError::Facade)?;
        let record = supplied.remove(&key).ok_or_else(|| {
            DecodeError::InvalidFormat("dense payload is missing an admitted block".to_string())
        })?;
        if record.shape.as_slice() != block.shape() {
            return Err(DecodeError::InvalidFormat(
                "serialized block shape differs from admitted shape".to_string(),
            ));
        }
        copy_logical_values(&mut data, &block, &record.values)?;
    }
    if !supplied.is_empty() {
        return Err(DecodeError::InvalidFormat(
            "dense payload contains an unadmitted block".to_string(),
        ));
    }
    Ok(TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    })
}

fn build_diagonal<R, D, C>(
    runtime: &Runtime,
    provider: &Arc<R>,
    record: SpaceRecord<R::Sector>,
    entries: Vec<(R::Sector, Vec<D>)>,
) -> Result<TensorMap<R, D>, DecodeError<C, TypedFacadeError<R>>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: WireScalar,
{
    let space = build_space(provider, record).map_err(DecodeError::Facade)?;
    if !is_diagonal_bond_space(space.space()) {
        return Err(DecodeError::InvalidFormat(
            "compact payload is not on a diagonal bond space".to_string(),
        ));
    }
    let mut spectrum = Vec::with_capacity(entries.len());
    for (label, values) in entries {
        let sector = R::try_encode_label(provider.as_ref(), &label)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
            .map_err(DecodeError::Facade)?;
        spectrum.push(tenet_matrixalgebra::SectorSpectrum { sector, values });
    }
    spectrum.sort_unstable_by_key(|entry| entry.sector);
    if spectrum
        .windows(2)
        .any(|pair| pair[0].sector == pair[1].sector)
    {
        return Err(DecodeError::InvalidFormat(
            "duplicate compact spectrum sector".to_string(),
        ));
    }
    let structure = space.space().structure();
    if structure.block_count() != spectrum.len() {
        return Err(DecodeError::InvalidFormat(
            "compact spectrum does not cover every admitted sector".to_string(),
        ));
    }
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(|error| DecodeError::InvalidFormat(error.to_string()))?;
        let pair = block.key().as_fusion_tree_pair().ok_or_else(|| {
            DecodeError::InvalidFormat("typed block is not a fusion-tree pair".to_string())
        })?;
        let entry = spectrum
            .iter()
            .find(|entry| entry.sector == pair.coupled())
            .ok_or_else(|| {
                DecodeError::InvalidFormat("compact spectrum is missing a sector".to_string())
            })?;
        if block.shape().len() != 2
            || block.shape()[0] != block.shape()[1]
            || entry.values.len() != block.shape()[0]
        {
            return Err(DecodeError::InvalidFormat(
                "compact spectrum length differs from admitted bond dimension".to_string(),
            ));
        }
    }
    Ok(TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::diagonal(space, spectrum)),
    })
}

fn build_adjoint<R, D, C>(
    parent: TensorMap<R, D>,
) -> Result<TensorMap<R, D>, DecodeError<C, TypedFacadeError<R>>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorAdjointDispatch<R, D>,
    D: TensorScalar,
{
    let tensor = <R::Mode as TypedTensorAdjointDispatch<R, D>>::adjoint(&parent)
        .map_err(DecodeError::Facade)?;
    if !matches!(tensor.repr, TypedTensorRepr::Adjoint(_)) {
        return Err(DecodeError::InvalidFormat(
            "lazy-adjoint record is not lazy for this provider mode".to_string(),
        ));
    }
    Ok(tensor)
}

fn decode_tensor<R, D, C>(
    runtime: &Runtime,
    bytes: &[u8],
    limits: DecodeLimits,
    codec: &C,
) -> Result<TensorMap<R, D>, DecodeError<C::Error, TypedFacadeError<R>>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R> + TypedTensorAdjointDispatch<R, D>,
    D: WireScalar,
    C: TypedPersistenceCodec<R>,
{
    preflight_tensor::<D, C::Error, TypedFacadeError<R>>(bytes, limits)?;
    let (mut reader, key) = read_header(bytes, limits, KIND_TENSOR, D::TAG)?;
    let provider = resolve_provider(codec, key)?;
    let record = read_tensor_record(&mut reader, limits, codec, provider.as_ref())?;
    reader.finish().map_err(DecodeError::InvalidFormat)?;
    match record {
        TensorRecord::Dense { space, blocks } => build_dense(runtime, &provider, space, blocks),
        TensorRecord::Diagonal { space, spectrum } => {
            build_diagonal(runtime, &provider, space, spectrum)
        }
        TensorRecord::AdjointDense {
            parent_space,
            parent_blocks,
        } => {
            let parent = build_dense(runtime, &provider, parent_space, parent_blocks)?;
            build_adjoint(parent)
        }
        TensorRecord::AdjointDiagonal {
            parent_space,
            parent_spectrum,
        } => {
            let parent = build_diagonal(runtime, &provider, parent_space, parent_spectrum)?;
            build_adjoint(parent)
        }
    }
}

impl<R> GradedSpace<R>
where
    R: TypedSectorAdmission,
{
    /// Encodes this Host-independent leg as a deterministic v1 semantic snapshot.
    pub fn to_bytes_with<C>(&self, codec: &C) -> Result<Vec<u8>, EncodeError<C::Error, R::Error>>
    where
        C: TypedPersistenceCodec<R>,
    {
        let provider_key = codec
            .provider_key(self.provider())
            .map_err(EncodeError::Codec)?;
        let mut output = Vec::new();
        put_header(&mut output, KIND_SPACE, SCALAR_NONE, &provider_key)?;
        encode_leg(&mut output, codec, self.provider(), &self.leg)?;
        Ok(output)
    }
}

impl<R> GradedSpace<R>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    /// Restores one leg with the exact provider `Arc` returned by `codec`.
    pub fn from_bytes_with<C>(
        bytes: &[u8],
        limits: DecodeLimits,
        codec: &C,
    ) -> Result<Self, DecodeError<C::Error, TypedFacadeError<R>>>
    where
        C: TypedPersistenceCodec<R>,
    {
        preflight_graded_space::<C::Error, TypedFacadeError<R>>(bytes, limits)?;
        let (mut reader, key) = read_header(bytes, limits, KIND_SPACE, SCALAR_NONE)?;
        let provider = resolve_provider(codec, key)?;
        let record = read_leg(&mut reader, limits, codec, provider.as_ref())?;
        reader.finish().map_err(DecodeError::InvalidFormat)?;
        build_leg(provider, record).map_err(DecodeError::Facade)
    }
}

macro_rules! tensor_persistence_impl {
    ($scalar:ty) => {
        impl<R> TensorMap<R, $scalar>
        where
            R: TypedSectorAdmission,
        {
            /// Encodes the exact dense, compact-diagonal, or lazy-adjoint Host representation.
            pub fn to_bytes_with<C>(
                &self,
                codec: &C,
            ) -> Result<Vec<u8>, EncodeError<C::Error, R::Error>>
            where
                C: TypedPersistenceCodec<R>,
            {
                encode_tensor(self, codec)
            }
        }

        impl<R> TensorMap<R, $scalar>
        where
            R: TypedSectorAdmission,
            R::Mode: TypedTensorRootDispatch<R> + TypedTensorAdjointDispatch<R, $scalar>,
        {
            /// Restores a Host `Vec` tensor transactionally under the resolved provider authority.
            pub fn from_bytes_with<C>(
                runtime: &Runtime,
                bytes: &[u8],
                limits: DecodeLimits,
                codec: &C,
            ) -> Result<Self, DecodeError<C::Error, TypedFacadeError<R>>>
            where
                C: TypedPersistenceCodec<R>,
            {
                decode_tensor(runtime, bytes, limits, codec)
            }
        }
    };
}

tensor_persistence_impl!(f64);
tensor_persistence_impl!(Complex64);
