use crate::{KrylovVector, LinearOperator};

/// Options for [`cg`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgOptions {
    /// Relative residual tolerance.
    pub rtol: f64,
    /// Absolute residual tolerance.
    pub atol: f64,
    /// Maximum CG iterations.
    pub max_iter: usize,
    /// Optional Tikhonov/identity damping. Solves `(A + damping I)x = b`.
    pub damping: f64,
    /// Detect non-positive curvature before taking a CG step.
    pub check_curvature: bool,
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            rtol: 1.0e-12,
            atol: 0.0,
            max_iter: 1_000,
            damping: 0.0,
            check_curvature: true,
        }
    }
}

/// CG return value.
#[derive(Clone, Debug, PartialEq)]
pub struct CgResult<V> {
    pub solution: V,
    pub stats: CgStats,
}

/// CG convergence and breakdown statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct CgStats {
    pub iterations: usize,
    pub matvecs: usize,
    pub converged: bool,
    pub initial_residual: f64,
    pub final_residual: f64,
    pub tolerance: f64,
    pub breakdown: Option<CgBreakdown>,
}

/// Reasons CG stopped without convergence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CgBreakdown {
    InvalidOptions,
    NonFiniteResidual,
    NonPositiveCurvature,
    NonFiniteCurvature,
    MaxIterations,
}

/// Solve `A x = b` or `(A + damping I) x = b` by Conjugate Gradient.
///
/// The initial guess is `zero_like(b)`. The method stops when
/// `||r|| <= atol + rtol * ||b||`; all exceptional exits are reported in
/// [`CgStats`] instead of panicking.
pub fn cg<A, V>(op: &A, b: &V, options: CgOptions) -> CgResult<V>
where
    A: LinearOperator<V>,
    V: KrylovVector,
{
    let mut x = b.zero_like();
    let mut r = b.clone();
    let mut p = r.clone();

    let b_norm = b.norm2();
    let initial_residual = r.norm2();
    let tolerance = options.atol + options.rtol * b_norm;

    let mut stats = CgStats {
        iterations: 0,
        matvecs: 0,
        converged: false,
        initial_residual,
        final_residual: initial_residual,
        tolerance,
        breakdown: None,
    };

    if !options.rtol.is_finite()
        || !options.atol.is_finite()
        || !options.damping.is_finite()
        || options.rtol < 0.0
        || options.atol < 0.0
        || options.damping < 0.0
    {
        stats.breakdown = Some(CgBreakdown::InvalidOptions);
        return CgResult { solution: x, stats };
    }

    if !initial_residual.is_finite() || !b_norm.is_finite() || !tolerance.is_finite() {
        stats.breakdown = Some(CgBreakdown::NonFiniteResidual);
        return CgResult { solution: x, stats };
    }

    if initial_residual <= tolerance {
        stats.converged = true;
        return CgResult { solution: x, stats };
    }

    let mut rho = r.dot_real(&r);
    if !rho.is_finite() || rho < 0.0 {
        stats.breakdown = Some(CgBreakdown::NonFiniteResidual);
        return CgResult { solution: x, stats };
    }

    while stats.iterations < options.max_iter {
        let mut ap = op.apply(&p);
        stats.matvecs += 1;

        if options.damping != 0.0 {
            ap.axpy(options.damping, &p);
        }

        let curvature = p.dot_real(&ap);
        if !curvature.is_finite() {
            stats.breakdown = Some(CgBreakdown::NonFiniteCurvature);
            return CgResult { solution: x, stats };
        }
        if options.check_curvature && curvature <= 0.0 {
            stats.breakdown = Some(CgBreakdown::NonPositiveCurvature);
            return CgResult { solution: x, stats };
        }

        let alpha = rho / curvature;
        if !alpha.is_finite() {
            stats.breakdown = Some(CgBreakdown::NonFiniteCurvature);
            return CgResult { solution: x, stats };
        }

        x.axpy(alpha, &p);
        r.axpy(-alpha, &ap);
        stats.iterations += 1;

        let residual = r.norm2();
        stats.final_residual = residual;
        if !residual.is_finite() {
            stats.breakdown = Some(CgBreakdown::NonFiniteResidual);
            return CgResult { solution: x, stats };
        }
        if residual <= tolerance {
            stats.converged = true;
            return CgResult { solution: x, stats };
        }

        let rho_next = r.dot_real(&r);
        if !rho_next.is_finite() || rho_next < 0.0 {
            stats.breakdown = Some(CgBreakdown::NonFiniteResidual);
            return CgResult { solution: x, stats };
        }

        let beta = rho_next / rho;
        if !beta.is_finite() {
            stats.breakdown = Some(CgBreakdown::NonFiniteResidual);
            return CgResult { solution: x, stats };
        }

        p.scale(beta);
        p.axpy(1.0, &r);
        rho = rho_next;
    }

    stats.breakdown = Some(CgBreakdown::MaxIterations);
    CgResult { solution: x, stats }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= 1.0e-10,
            "expected {lhs} to be close to {rhs}"
        );
    }

    #[test]
    fn solves_small_real_spd_system() {
        let a = |x: &Vec<f64>| vec![4.0 * x[0] + x[1], x[0] + 3.0 * x[1]];
        let b = vec![1.0, 2.0];

        let result = cg(&a, &b, CgOptions::default());

        assert!(result.stats.converged, "{:?}", result.stats);
        assert_eq!(result.stats.breakdown, None);
        assert_close(result.solution[0], 1.0 / 11.0);
        assert_close(result.solution[1], 7.0 / 11.0);
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct C64 {
        re: f64,
        im: f64,
    }

    impl C64 {
        fn new(re: f64, im: f64) -> Self {
            Self { re, im }
        }

        fn scale(self, alpha: f64) -> Self {
            Self {
                re: alpha * self.re,
                im: alpha * self.im,
            }
        }

        fn add(self, rhs: Self) -> Self {
            Self {
                re: self.re + rhs.re,
                im: self.im + rhs.im,
            }
        }

        fn mul(self, rhs: Self) -> Self {
            Self {
                re: self.re * rhs.re - self.im * rhs.im,
                im: self.re * rhs.im + self.im * rhs.re,
            }
        }

        fn conj_dot_real(self, rhs: Self) -> f64 {
            self.re * rhs.re + self.im * rhs.im
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct ComplexVec(Vec<C64>);

    impl KrylovVector for ComplexVec {
        fn zero_like(&self) -> Self {
            Self(vec![C64::new(0.0, 0.0); self.0.len()])
        }

        fn axpy(&mut self, alpha: f64, x: &Self) {
            for (lhs, rhs) in self.0.iter_mut().zip(&x.0) {
                *lhs = lhs.add(rhs.scale(alpha));
            }
        }

        fn scale(&mut self, alpha: f64) {
            for value in &mut self.0 {
                *value = value.scale(alpha);
            }
        }

        fn dot_real(&self, rhs: &Self) -> f64 {
            self.0
                .iter()
                .zip(&rhs.0)
                .map(|(lhs, rhs)| lhs.conj_dot_real(*rhs))
                .sum()
        }
    }

    #[test]
    fn solves_small_complex_hermitian_spd_system() {
        // A = [[2, i], [-i, 2]], x = [1+i, 2-i], b = A*x = [3+4i, 5-3i].
        let op = |x: &ComplexVec| {
            ComplexVec(vec![
                x.0[0].scale(2.0).add(C64::new(0.0, 1.0).mul(x.0[1])),
                C64::new(0.0, -1.0).mul(x.0[0]).add(x.0[1].scale(2.0)),
            ])
        };
        let b = ComplexVec(vec![C64::new(3.0, 4.0), C64::new(5.0, -3.0)]);

        let result = cg(&op, &b, CgOptions::default());

        assert!(result.stats.converged, "{:?}", result.stats);
        assert_close(result.solution.0[0].re, 1.0);
        assert_close(result.solution.0[0].im, 1.0);
        assert_close(result.solution.0[1].re, 2.0);
        assert_close(result.solution.0[1].im, -1.0);
    }

    #[test]
    fn damping_changes_diagonal_solution() {
        let op = |x: &Vec<f64>| vec![2.0 * x[0], 4.0 * x[1]];
        let b = vec![6.0, 10.0];
        let options = CgOptions {
            damping: 1.0,
            ..CgOptions::default()
        };

        let result = cg(&op, &b, options);

        assert!(result.stats.converged, "{:?}", result.stats);
        assert_close(result.solution[0], 2.0);
        assert_close(result.solution[1], 2.0);
    }

    #[test]
    fn detects_non_positive_curvature_for_negative_operator() {
        let op = |x: &Vec<f64>| vec![-x[0], -x[1]];
        let b = vec![1.0, 2.0];

        let result = cg(&op, &b, CgOptions::default());

        assert!(!result.stats.converged);
        assert_eq!(
            result.stats.breakdown,
            Some(CgBreakdown::NonPositiveCurvature)
        );
        assert_eq!(result.stats.iterations, 0);
        assert_eq!(result.stats.matvecs, 1);
    }

    #[test]
    fn reports_max_iterations_without_panic() {
        let op = |x: &Vec<f64>| vec![4.0 * x[0] + x[1], x[0] + 3.0 * x[1]];
        let b = vec![1.0, 2.0];
        let options = CgOptions {
            max_iter: 1,
            rtol: 0.0,
            atol: 0.0,
            ..CgOptions::default()
        };

        let result = cg(&op, &b, options);

        assert!(!result.stats.converged);
        assert_eq!(result.stats.breakdown, Some(CgBreakdown::MaxIterations));
        assert_eq!(result.stats.iterations, 1);
    }

    #[test]
    fn rejects_invalid_options_without_matvec() {
        let op = |_x: &Vec<f64>| panic!("invalid options should stop before matvec");
        let b = vec![1.0, 2.0];
        let options = CgOptions {
            rtol: -1.0,
            ..CgOptions::default()
        };

        let result = cg(&op, &b, options);

        assert!(!result.stats.converged);
        assert_eq!(result.stats.breakdown, Some(CgBreakdown::InvalidOptions));
        assert_eq!(result.stats.matvecs, 0);
    }
}
