//! Performs batch Groth16 proof verification.
//!
//! Batch verification asks whether *all* proofs in some set are valid,
//! rather than asking whether *each* of them is valid. This allows sharing
//! computations among all proof verifications, performing less work overall
//! at the cost of higher latency (the entire batch must complete), complexity of
//! caller code (which must assemble a batch of proofs across work-items),
//! and loss of the ability to easily pinpoint failing proofs.
//!
//! This batch verification implementation is non-adaptive, in the sense that it
//! assumes that all the proofs in the batch are verifiable by the same
//! `VerifyingKey`. The reason is that if you have different proof statements,
//! you need to specify which statement you are proving, which means that you
//! need to refer to or lookup a particular `VerifyingKey`. In practice, with
//! large enough batches, it's manageable and not much worse performance-wise to
//! keep batches of each statement type, vs one large adaptive batch.
//!
//! Each batch gives one proof the weight one, and every other proof an independent
//! weight drawn uniformly from `1..=2^128 - 1` (or uniformly from the nonzero
//! scalars, if the scalar field has at most 128 bits). A batch containing any
//! invalid proof is accepted with probability at most `1 / (2^128 - 1)` (or
//! `1 / (r - 1)` for a scalar field of order `r`), however many of its proofs are
//! invalid: the batch equation holds in a group of prime order `r`, so once every
//! other weight is fixed, at most one weight of an invalid, randomly weighted proof
//! satisfies it. A batch whose only invalid proof has the unit weight is always
//! rejected.
//!
//! Proofs, public inputs and verifying keys are public, and the weights only need to
//! be unpredictable until the proofs are fixed, so this module uses variable-time
//! arithmetic.

use std::ops::AddAssign;

use bellman::VerificationError;
use ff::{Field, PrimeField};
use group::{Curve, CurveAffine, Group};
use pairing::{MillerLoopResult, MultiMillerLoop};
use rand_core::{CryptoRng, Rng};

#[cfg(feature = "multicore")]
use rand::rngs::SysRng;
#[cfg(feature = "multicore")]
use rand_core::UnwrapErr;

#[cfg(feature = "multicore")]
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    prelude::ParallelSlice,
};

use crate::{PreparedVerifyingKey, Proof, VerifyingKey};

mod prepared;
pub use prepared::{PreparedBatchG2, PreparedBatchVerifyingKey};

/// Samples a random batch weight, as described in the module documentation.
fn random_weight<F: PrimeField>(rng: &mut impl Rng) -> F {
    loop {
        let weight = if F::NUM_BITS > 128 {
            // The field order exceeds 2^128, so these integers are distinct scalars.
            let mut bytes = [0; 16];
            rng.fill_bytes(&mut bytes);
            F::from_u128(u128::from_le_bytes(bytes))
        } else {
            F::random(&mut *rng)
        };
        // A zero weight would remove its proof from the batch equation.
        if !weight.is_zero_vartime() {
            return weight;
        }
    }
}

/// Returns the width-4 NAF digits of a little-endian integer, least significant first,
/// without trailing zeros.
pub(crate) fn wnaf(bytes: &[u8]) -> Vec<i8> {
    let bit = |i: usize| bytes.get(i / 8).map_or(0, |byte| (byte >> (i % 8)) & 1);
    let mut digits = Vec::with_capacity(8 * bytes.len() + 4);
    let mut carry = 0;
    let mut i = 0;
    // The position after the last byte absorbs a final carry.
    while i <= 8 * bytes.len() {
        let window = carry + (0..4).map(|k| bit(i + k) << k).sum::<u8>();
        if window % 2 == 0 {
            digits.push(0);
            i += 1;
        } else {
            // Every nonzero digit is odd, in -7..=7, and followed by three zeros.
            let digit = i8::try_from(window).expect("windows are at most 16");
            let digit = if window < 8 {
                carry = 0;
                digit
            } else {
                carry = 1;
                digit - 16
            };
            digits.extend([digit, 0, 0, 0]);
            i += 4;
        }
    }
    while digits.last() == Some(&0) {
        digits.pop();
    }
    digits
}

/// Returns whether `to_repr()` encodes scalars as little-endian integers.
///
/// Fields too small to represent the 64-bit probe value return `false`, so they use the
/// group's own multiplication.
pub(crate) fn is_little_endian<F: PrimeField>() -> bool {
    let probe = F::from(0x0807_0605_0403_0201).to_repr();
    let probe = probe.as_ref();
    probe.starts_with(&[1, 2, 3, 4, 5, 6, 7, 8]) && probe[8..].iter().all(|&byte| byte == 0)
}

/// Computes `sum(bases[i] * scalars[i])` in variable time.
///
/// Little-endian scalar encodings use interleaved wNAF, so all terms share one sequence
/// of doublings. Other encodings use the group's own multiplication.
pub(crate) fn multiexp_vartime<G: Group>(bases: &[G], scalars: &[G::Scalar]) -> G {
    if !is_little_endian::<G::Scalar>() {
        return bases
            .iter()
            .zip(scalars)
            .map(|(base, scalar)| *base * scalar)
            .sum();
    }

    // Odd multiples of each base, from one to seven times.
    let tables = bases
        .iter()
        .map(|&base| {
            let double = base.double();
            let mut table = [base; 4];
            for i in 1..4 {
                table[i] = table[i - 1] + double;
            }
            table
        })
        .collect::<Vec<_>>();
    let digits = scalars
        .iter()
        .map(|scalar| wnaf(scalar.to_repr().as_ref()))
        .collect::<Vec<_>>();

    let mut sum = G::identity();
    for i in (0..digits.iter().map(Vec::len).max().unwrap_or(0)).rev() {
        sum = sum.double();
        for (table, digits) in tables.iter().zip(&digits) {
            let digit = digits.get(i).copied().unwrap_or(0);
            let term = &table[usize::from(digit.unsigned_abs() / 2)];
            match digit.signum() {
                1 => sum += term,
                -1 => sum -= term,
                _ => {}
            }
        }
    }
    sum
}

/// Weighs batch items, returning their Miller loop terms, the weighted sums of their public
/// inputs, and the weighted sum of their `C` points.
///
/// The first public input is implicitly one, so its weighted sum is the sum of the weights.
#[allow(clippy::type_complexity)]
fn weigh<E: MultiMillerLoop>(
    items: &[Item<E>],
    weights: &[E::Fr],
    ic_len: usize,
) -> (Vec<E::G1>, Vec<E::G2Prepared>, Vec<E::Fr>, E::G1) {
    let mut a = Vec::with_capacity(items.len());
    let mut b = Vec::with_capacity(items.len());
    let mut c = Vec::with_capacity(items.len());
    let mut gammas = vec![E::Fr::ZERO; ic_len];
    for (Item { proof, inputs }, weight) in items.iter().zip(weights) {
        a.push(multiexp_vartime(
            &[proof.a.to_curve()],
            std::slice::from_ref(weight),
        ));
        b.push((-proof.b).into());
        c.push(proof.c.to_curve());
        gammas[0] += weight;
        for (input, gamma) in inputs.iter().zip(&mut gammas[1..]) {
            *gamma += *weight * input;
        }
    }
    (a, b, gammas, multiexp_vartime(&c, weights))
}

/// Returns the Miller loop terms that pair the batch sums with the verifying key.
fn shared_terms<E: MultiMillerLoop>(
    vk: &VerifyingKey<E>,
    gammas: &[E::Fr],
    delta: E::G1,
) -> [E::G1; 3] {
    let ic = vk.ic.iter().map(|ic| ic.to_curve()).collect::<Vec<_>>();
    let psi = multiexp_vartime(&ic, gammas);
    // Multiplying `alpha_g1` by the sum of the weights is much cheaper than raising
    // `e(alpha_g1, beta_g2)` to that power in the target group.
    let alpha = multiexp_vartime(&[vk.alpha_g1.to_curve()], &gammas[..1]);
    [delta, psi, alpha]
}

/// Runs a Miller loop, normalizing all G1 points with one shared inversion.
fn miller_loop<E: MultiMillerLoop>(g1: &[E::G1], g2: &[E::G2Prepared]) -> E::Result {
    let mut affine = vec![E::G1Affine::identity(); g1.len()];
    E::G1::batch_normalize(g1, &mut affine);
    let terms = affine.iter().zip(g2).collect::<Vec<_>>();
    E::multi_miller_loop(&terms)
}

/// A batch verification item.
///
/// This struct exists to allow batch processing to be decoupled from the
/// lifetime of the message. This is useful when using the batch verification
/// API in an async context.
#[derive(Clone, Debug)]
pub struct Item<E: MultiMillerLoop> {
    proof: Proof<E>,
    inputs: Vec<E::Fr>,
}

impl<E: MultiMillerLoop> From<(&Proof<E>, &[E::Fr])> for Item<E> {
    fn from((proof, inputs): (&Proof<E>, &[E::Fr])) -> Self {
        (proof.clone(), inputs.to_owned()).into()
    }
}

impl<E: MultiMillerLoop> From<(Proof<E>, Vec<E::Fr>)> for Item<E> {
    fn from((proof, inputs): (Proof<E>, Vec<E::Fr>)) -> Self {
        Self { proof, inputs }
    }
}

impl<E: MultiMillerLoop> Item<E> {
    /// Perform non-batched verification of this `Item`.
    ///
    /// This is useful (in combination with `Item::clone`) for implementing
    /// fallback logic when batch verification fails.
    pub fn verify_single(self, pvk: &PreparedVerifyingKey<E>) -> Result<(), VerificationError> {
        super::verify_proof(pvk, &self.proof, &self.inputs)
    }
}

/// A batch verification context.
///
/// In practice, you would create a batch verifier for each proof statement
/// requiring the same `VerifyingKey`.
#[derive(Debug)]
pub struct Verifier<E: MultiMillerLoop> {
    items: Vec<Item<E>>,
}

// Need to impl Default by hand to avoid a derived E: Default bound
impl<E: MultiMillerLoop> Default for Verifier<E> {
    fn default() -> Self {
        Self { items: Vec::new() }
    }
}

impl<E: MultiMillerLoop> Verifier<E>
where
    E::G1: AddAssign<E::G1>,
{
    /// Construct a new batch verifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a (proof, inputs) tuple for verification.
    pub fn queue<I: Into<Item<E>>>(&mut self, item: I) {
        self.items.push(item.into())
    }

    /// Perform batch verification with a particular `VerifyingKey`, returning
    /// `Ok(())` if all proofs were verified and `VerificationError` otherwise.
    pub fn verify<R: Rng + CryptoRng>(
        self,
        rng: R,
        vk: &VerifyingKey<E>,
    ) -> Result<(), VerificationError> {
        if self.items.is_empty() {
            return Ok(());
        }
        let g2 = PreparedBatchG2::for_one_batch(vk);
        self.verify_prepared(rng, &PreparedBatchVerifyingKey::from_cached(vk, &g2)?)
    }

    /// Perform batch verification with a particular `VerifyingKey`, returning
    /// `Ok(())` if all proofs were verified and `VerificationError` otherwise.
    ///
    /// This performs the bulk of internal arithmetic over the global rayon
    /// threadpool.
    #[cfg(feature = "multicore")]
    pub fn verify_multicore(self, vk: &VerifyingKey<E>) -> Result<(), VerificationError> {
        if self.items.is_empty() {
            return Ok(());
        }
        let g2 = PreparedBatchG2::for_one_batch(vk);
        self.verify_multicore_prepared(&PreparedBatchVerifyingKey::from_cached(vk, &g2)?)
    }

    /// Verifies a batch in parallel while borrowing reusable fixed G2 terms.
    #[cfg(feature = "multicore")]
    pub fn verify_multicore_prepared(
        self,
        key: &PreparedBatchVerifyingKey<'_, E>,
    ) -> Result<(), VerificationError> {
        let vk = key.vk;
        if self
            .items
            .iter()
            .any(|Item { inputs, .. }| inputs.len() + 1 != vk.ic.len())
        {
            return Err(VerificationError::InvalidVerifyingKey);
        }

        struct Accumulator<E: MultiMillerLoop> {
            gammas: Vec<E::Fr>,
            delta: E::G1,
            ml_result: Option<E::Result>,
        }

        impl<E: MultiMillerLoop> Accumulator<E> {
            fn new(ic_len: usize) -> Self {
                Accumulator {
                    gammas: vec![E::Fr::ZERO; ic_len],
                    delta: E::G1::identity(),
                    ml_result: None,
                }
            }
        }

        let ic_len = vk.ic.len();

        let chunk_size = self
            .items
            .len()
            .div_ceil(rayon::current_num_threads())
            .clamp(1, 8);
        let acc = self
            .items
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk, items)| {
                let mut rng = UnwrapErr(SysRng);
                // Exactly one item in the whole batch has the unit weight. Other weights are
                // independent, rather than powers of one random value, so that they stay 128
                // bits long and keep the serial bound.
                let weights = (0..items.len())
                    .map(|i| {
                        if chunk == 0 && i == 0 {
                            E::Fr::ONE
                        } else {
                            random_weight(&mut rng)
                        }
                    })
                    .collect::<Vec<_>>();
                let (g1, g2, gammas, delta) = weigh(items, &weights, ic_len);
                Accumulator {
                    gammas,
                    delta,
                    ml_result: Some(miller_loop::<E>(&g1, &g2)),
                }
            })
            .reduce(
                || Accumulator::<E>::new(ic_len),
                |mut a, b| {
                    for (a, b) in a.gammas.iter_mut().zip(b.gammas.into_iter()) {
                        *a += b;
                    }
                    a.delta += b.delta;
                    a.ml_result = match (a.ml_result, b.ml_result) {
                        (Some(a), Some(b)) => Some(a + b),
                        (Some(a), None) | (None, Some(a)) => Some(a),
                        (None, None) => None,
                    };
                    a
                },
            );

        match acc.ml_result {
            None => Ok(()),
            Some(mut ml_result) => {
                let g1 = shared_terms(vk, &acc.gammas, acc.delta);
                let mut affine = [E::G1Affine::identity(); 3];
                E::G1::batch_normalize(&g1, &mut affine);
                let terms = affine.iter().zip(&key.g2.terms).collect::<Vec<_>>();
                ml_result += E::multi_miller_loop(&terms);

                if ml_result.final_exponentiation_vartime() == E::Gt::identity() {
                    Ok(())
                } else {
                    Err(VerificationError::InvalidProof)
                }
            }
        }
    }
}
