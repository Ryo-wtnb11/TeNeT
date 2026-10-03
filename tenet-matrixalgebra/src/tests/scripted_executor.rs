//! The one `DenseExecutor` test double of tenet-matrixalgebra (#1826).
//!
//! Each entry point has a scripted [`Action`]: forward to the wrapped
//! `DefaultDenseExecutor`, keep the trait's default body (which funnels into
//! the required `svd`/`qr`/`eigh`/`dot_general_into` entries of this same
//! executor), or panic. Faults fail chosen calls with a backend error, and an
//! [`Observer`] can inspect arguments and outputs, or answer a call itself.
//! Every entry is counted where it enters, before its action runs.
//!
//! The crate's unit tests use it as `crate::tests::scripted_executor`;
//! integration tests include this file with `#[path]`.

#![allow(dead_code)]

use std::ops::{Deref, DerefMut};

use tenet_dense::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseFactorization, DenseOwned, DenseRead, DenseTensor, DenseWrite,
};

/// One scripted executor entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Svd,
    SvdInto,
    SvdVals,
    SvdFullOwned,
    Qr,
    QrInto,
    Eigh,
    EighInto,
    EighVals,
    Eig,
    EigVals,
    Solve,
    DotGeneral,
    FactorizeBatch,
}

const OPS: usize = 14;

/// What an entry point does once counted, faults and the observer allowing.
#[derive(Clone, Copy, Debug)]
pub enum Action {
    /// Call the wrapped executor's entry.
    Forward,
    /// Run the trait's default body on this executor. For the required
    /// entries (`svd`, `qr`, `eigh`, `dot_general_into`), which have none,
    /// this forwards.
    Default,
    /// Panic with the message: the test must not reach this entry.
    Panic(&'static str),
}

/// Calls per entry point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub svd: usize,
    pub svd_into: usize,
    pub svd_vals: usize,
    pub svd_full: usize,
    pub qr: usize,
    pub qr_into: usize,
    pub eigh: usize,
    pub eigh_into: usize,
    pub eigh_vals: usize,
    pub eig: usize,
    pub eig_vals: usize,
    pub solve: usize,
    pub dot_general: usize,
    pub factorize_batch: usize,
}

impl Counts {
    fn slot(&mut self, op: Op) -> &mut usize {
        match op {
            Op::Svd => &mut self.svd,
            Op::SvdInto => &mut self.svd_into,
            Op::SvdVals => &mut self.svd_vals,
            Op::SvdFullOwned => &mut self.svd_full,
            Op::Qr => &mut self.qr,
            Op::QrInto => &mut self.qr_into,
            Op::Eigh => &mut self.eigh,
            Op::EighInto => &mut self.eigh_into,
            Op::EighVals => &mut self.eigh_vals,
            Op::Eig => &mut self.eig,
            Op::EigVals => &mut self.eig_vals,
            Op::Solve => &mut self.solve,
            Op::DotGeneral => &mut self.dot_general,
            Op::FactorizeBatch => &mut self.factorize_batch,
        }
    }

    pub fn get(mut self, op: Op) -> usize {
        *self.slot(op)
    }

    /// Calls summed over `ops`.
    pub fn of(self, ops: &[Op]) -> usize {
        ops.iter().map(|&op| self.get(op)).sum()
    }
}

/// Fails the `nth` call (1-based, counted over `ops` together), or every call
/// when `nth` is `None`, with `DenseError::Backend { op: label, message }`.
#[derive(Clone, Copy, Debug)]
pub struct Fault {
    pub ops: &'static [Op],
    pub nth: Option<usize>,
    pub label: &'static str,
    pub message: &'static str,
}

/// The arguments of one call, lent to the [`Observer`] before it runs.
pub enum Call<'c, 'v> {
    /// `svd`, `qr`, `eigh`, `eig` and the values-only entries.
    Read(Op, &'c DenseRead<'v>),
    /// `svd_into` (`[u, s, vt]`), `qr_into` (`[q, r]`), `eigh_into`
    /// (`[values, vectors]`).
    Into(Op, &'c DenseRead<'v>, &'c [&'c DenseWrite<'v>]),
    Solve {
        a: &'c DenseRead<'v>,
        b: &'c DenseRead<'v>,
        x: &'c DenseWrite<'v>,
    },
    Gemm {
        lhs: &'c DenseRead<'v>,
        rhs: &'c DenseRead<'v>,
        config: &'c DenseDotConfig,
    },
}

/// A reply that replaces the scripted action: owned outputs (empty for a
/// destination entry, one tensor for a values-only entry) or an error.
pub type Reply = Result<Vec<DenseTensor>, DenseError>;

/// Per-test observation and scripting. Every method defaults to doing nothing.
pub trait Observer {
    /// Sets the actions and faults this double starts with.
    fn script(_executor: &mut Script) {}

    /// Sees a call after counting, panics and faults, before its action;
    /// `Some` answers the call instead of the action.
    fn call(&mut self, _call: Call<'_, '_>) -> Option<Reply> {
        None
    }

    /// Sees the owned outputs of a forwarded or defaulted call.
    fn outputs(&mut self, _op: Op, _outputs: &mut Vec<DenseTensor>) {}

    /// Sees the outputs of a forwarded `factorize_batch`.
    fn batch_outputs(&mut self, _outputs: &mut Vec<Vec<DenseTensor>>) {}
}

impl Observer for () {}

/// The scripted part of a [`ScriptedExecutor`]: actions, faults, counts and
/// the full-SVD capability.
pub struct Script {
    pub inner: DefaultDenseExecutor,
    actions: [Action; OPS],
    pub faults: Vec<Fault>,
    pub counts: Counts,
    pub supports_svd_full: bool,
}

impl Script {
    pub fn set(&mut self, op: Op, action: Action) -> &mut Self {
        self.actions[op as usize] = action;
        self
    }

    /// Sets `action` on every op in `ops`.
    pub fn set_all(&mut self, ops: &[Op], action: Action) -> &mut Self {
        for &op in ops {
            self.set(op, action);
        }
        self
    }

    pub fn fail(
        &mut self,
        ops: &'static [Op],
        nth: Option<usize>,
        label: &'static str,
        message: &'static str,
    ) -> &mut Self {
        self.faults.push(Fault {
            ops,
            nth,
            label,
            message,
        });
        self
    }

    /// Counts the call, then applies a panic action or a matching fault.
    fn enter(&mut self, op: Op) -> Result<Action, DenseError> {
        *self.counts.slot(op) += 1;
        let action = self.actions[op as usize];
        if let Action::Panic(message) = action {
            panic!("{message}");
        }
        for fault in &self.faults {
            if fault.ops.contains(&op)
                && fault.nth.is_none_or(|nth| self.counts.of(fault.ops) == nth)
            {
                return Err(DenseError::Backend {
                    backend: DenseBackend::Tenferro,
                    op: fault.label,
                    message: fault.message.to_string(),
                });
            }
        }
        Ok(action)
    }
}

/// See the module documentation. `Deref`s to its observer, so a test reads
/// what the observer recorded as fields of the executor.
pub struct ScriptedExecutor<O: Observer = ()> {
    pub script: Script,
    pub observer: O,
}

impl<O: Observer> ScriptedExecutor<O> {
    pub fn new(observer: O) -> Self {
        Self::with_inner(DefaultDenseExecutor::new(), observer)
    }

    pub fn with_inner(inner: DefaultDenseExecutor, observer: O) -> Self {
        let mut script = Script {
            inner,
            actions: [Action::Default; OPS],
            faults: Vec::new(),
            counts: Counts::default(),
            supports_svd_full: false,
        };
        O::script(&mut script);
        Self { script, observer }
    }

    pub fn counts(&self) -> Counts {
        self.script.counts
    }
}

impl<O: Observer + Default> Default for ScriptedExecutor<O> {
    fn default() -> Self {
        Self::new(O::default())
    }
}

impl<O: Observer> Deref for ScriptedExecutor<O> {
    type Target = O;
    fn deref(&self) -> &O {
        &self.observer
    }
}

impl<O: Observer> DerefMut for ScriptedExecutor<O> {
    fn deref_mut(&mut self) -> &mut O {
        &mut self.observer
    }
}

/// Runs a trait default body with this executor: implements only the
/// required entries, by calling back into the scripted executor, so the
/// default's inner calls are scripted and counted like direct ones.
struct Defaulted<'a, O: Observer>(&'a mut ScriptedExecutor<O>);

impl<O: Observer> DenseExecutor for Defaulted<'_, O> {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.svd(input)
    }
    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.qr(input)
    }
    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.eigh(input)
    }
    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.0.dot_general_into(output, lhs, rhs, config)
    }
}

fn single(mut outputs: Vec<DenseTensor>) -> DenseTensor {
    assert_eq!(outputs.len(), 1, "a values-only reply holds one tensor");
    outputs.pop().unwrap()
}

impl<O: Observer> ScriptedExecutor<O> {
    /// The owned-output entries: `svd`, `qr`, `eigh`, `eig`.
    fn owned(
        &mut self,
        op: Op,
        input: DenseRead<'_>,
        forward: fn(
            &mut DefaultDenseExecutor,
            DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError>,
        default: fn(&mut Defaulted<'_, O>, DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let action = self.script.enter(op)?;
        if let Some(reply) = self.observer.call(Call::Read(op, &input)) {
            return reply;
        }
        let mut outputs = match action {
            Action::Forward => forward(&mut self.script.inner, input),
            _ => default(&mut Defaulted(self), input),
        }?;
        self.observer.outputs(op, &mut outputs);
        Ok(outputs)
    }

    /// The values-only entries.
    fn values(
        &mut self,
        op: Op,
        input: DenseRead<'_>,
        forward: fn(&mut DefaultDenseExecutor, DenseRead<'_>) -> Result<DenseTensor, DenseError>,
        default: fn(&mut Defaulted<'_, O>, DenseRead<'_>) -> Result<DenseTensor, DenseError>,
    ) -> Result<DenseTensor, DenseError> {
        let action = self.script.enter(op)?;
        if let Some(reply) = self.observer.call(Call::Read(op, &input)) {
            return reply.map(single);
        }
        match action {
            Action::Forward => forward(&mut self.script.inner, input),
            _ => default(&mut Defaulted(self), input),
        }
    }
}

impl<O: Observer> DenseExecutor for ScriptedExecutor<O> {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.owned(
            Op::Svd,
            input,
            |e, i| e.svd(i),
            |e, i| e.0.script.inner.svd(i),
        )
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.owned(Op::Qr, input, |e, i| e.qr(i), |e, i| e.0.script.inner.qr(i))
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.owned(
            Op::Eigh,
            input,
            |e, i| e.eigh(i),
            |e, i| e.0.script.inner.eigh(i),
        )
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.owned(Op::Eig, input, |e, i| e.eig(i), |e, i| e.eig(i))
    }

    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.values(
            Op::SvdVals,
            input,
            |e, i| e.svd_vals(i),
            |e, i| e.svd_vals(i),
        )
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.values(
            Op::EighVals,
            input,
            |e, i| e.eigh_vals(i),
            |e, i| e.eigh_vals(i),
        )
    }

    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        self.values(
            Op::EigVals,
            input,
            |e, i| e.eig_vals(i),
            |e, i| e.eig_vals(i),
        )
    }

    fn svd_into(
        &mut self,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let action = self.script.enter(Op::SvdInto)?;
        if let Some(reply) = self
            .observer
            .call(Call::Into(Op::SvdInto, &input, &[&u, &s, &vt]))
        {
            return reply.map(drop);
        }
        match action {
            Action::Forward => self.script.inner.svd_into(input, u, s, vt),
            _ => Defaulted(self).svd_into(input, u, s, vt),
        }
    }

    fn qr_into(
        &mut self,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let action = self.script.enter(Op::QrInto)?;
        if let Some(reply) = self
            .observer
            .call(Call::Into(Op::QrInto, &input, &[&q, &r]))
        {
            return reply.map(drop);
        }
        match action {
            Action::Forward => self.script.inner.qr_into(input, q, r),
            _ => Defaulted(self).qr_into(input, q, r),
        }
    }

    fn eigh_into(
        &mut self,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let action = self.script.enter(Op::EighInto)?;
        if let Some(reply) =
            self.observer
                .call(Call::Into(Op::EighInto, &input, &[&values, &vectors]))
        {
            return reply.map(drop);
        }
        match action {
            Action::Forward => self.script.inner.eigh_into(input, values, vectors),
            _ => Defaulted(self).eigh_into(input, values, vectors),
        }
    }

    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let action = self.script.enter(Op::Solve)?;
        if let Some(reply) = self.observer.call(Call::Solve {
            a: &a,
            b: &b,
            x: &x,
        }) {
            return reply.map(drop);
        }
        match action {
            Action::Forward => self.script.inner.solve_into(a, b, x),
            _ => Defaulted(self).solve_into(a, b, x),
        }
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.script.enter(Op::DotGeneral)?;
        if let Some(reply) = self.observer.call(Call::Gemm {
            lhs: &lhs,
            rhs: &rhs,
            config,
        }) {
            return reply.map(drop);
        }
        self.script.inner.dot_general_into(output, lhs, rhs, config)
    }

    fn supports_svd_full(&self) -> bool {
        self.script.supports_svd_full
    }

    fn svd_full_owned(
        &mut self,
        input: DenseOwned,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        match self.script.enter(Op::SvdFullOwned)? {
            Action::Forward => self.script.inner.svd_full_owned(input, rows, cols),
            _ => Defaulted(self).svd_full_owned(input, rows, cols),
        }
    }

    fn factorize_batch(
        &mut self,
        op: DenseFactorization,
        inputs: &[DenseRead<'_>],
    ) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
        let mut outputs = match self.script.enter(Op::FactorizeBatch)? {
            Action::Forward => self.script.inner.factorize_batch(op, inputs),
            _ => Defaulted(self).factorize_batch(op, inputs),
        }?;
        self.observer.batch_outputs(&mut outputs);
        Ok(outputs)
    }
}
