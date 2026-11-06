use super::*;

/// Fixed G2 terms prepared once for repeated batches under one verifying key.
#[derive(Clone)]
pub struct PreparedBatchG2<E: MultiMillerLoop> {
    source: [E::G2Affine; 3],
    pub(super) terms: [E::G2Prepared; 3],
}

impl<E: MultiMillerLoop> From<&VerifyingKey<E>> for PreparedBatchG2<E> {
    fn from(vk: &VerifyingKey<E>) -> Self {
        let source = [vk.delta_g2, vk.gamma_g2, vk.beta_g2];
        Self {
            source,
            terms: source.map(E::prepare_reusable_g2),
        }
    }
}

impl<E: MultiMillerLoop> PreparedBatchG2<E> {
    pub(super) fn for_one_batch(vk: &VerifyingKey<E>) -> Self {
        let source = [vk.delta_g2, vk.gamma_g2, vk.beta_g2];
        Self {
            source,
            terms: source.map(Into::into),
        }
    }
}

/// A verifying key borrowing its reusable fixed G2 pairing terms.
pub struct PreparedBatchVerifyingKey<'a, E: MultiMillerLoop> {
    pub(super) vk: &'a VerifyingKey<E>,
    pub(super) g2: &'a PreparedBatchG2<E>,
}

impl<'a, E: MultiMillerLoop> PreparedBatchVerifyingKey<'a, E> {
    /// Borrows cached terms, rejecting terms prepared for a different verifying key.
    pub fn from_cached(
        vk: &'a VerifyingKey<E>,
        g2: &'a PreparedBatchG2<E>,
    ) -> Result<Self, VerificationError> {
        if g2.source != [vk.delta_g2, vk.gamma_g2, vk.beta_g2] {
            return Err(VerificationError::InvalidVerifyingKey);
        }
        Ok(Self { vk, g2 })
    }
}

struct BatchTerms<E: MultiMillerLoop> {
    affine: Vec<E::G1Affine>,
    variable_g2: Vec<E::G2Prepared>,
}

impl<E: MultiMillerLoop> BatchTerms<E> {
    fn with_key<'a>(
        &'a self,
        key: &'a PreparedBatchVerifyingKey<'_, E>,
    ) -> Vec<(&'a E::G1Affine, &'a E::G2Prepared)> {
        self.affine
            .iter()
            .zip(self.variable_g2.iter().chain(&key.g2.terms))
            .collect()
    }
}

impl<E: MultiMillerLoop> Verifier<E>
where
    E::G1: AddAssign<E::G1>,
{
    /// Returns the number of queued proof statements.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns whether this batch contains no proofs.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn check_key(&self, vk: &VerifyingKey<E>) -> Result<(), VerificationError> {
        if self
            .items
            .iter()
            .any(|item| item.inputs.len() + 1 != vk.ic.len())
        {
            Err(VerificationError::InvalidVerifyingKey)
        } else {
            Ok(())
        }
    }

    fn randomize_terms(
        &self,
        rng: &mut impl Rng,
        vk: &VerifyingKey<E>,
        first_unit: bool,
    ) -> BatchTerms<E> {
        let weights = (0..self.items.len())
            .map(|i| {
                if first_unit && i == 0 {
                    E::Fr::ONE
                } else {
                    random_weight(rng)
                }
            })
            .collect::<Vec<_>>();
        let (mut g1, variable_g2, gammas, delta) = weigh(&self.items, &weights, vk.ic.len());
        g1.extend(shared_terms(vk, &gammas, delta));
        let mut affine = vec![E::G1Affine::identity(); g1.len()];
        E::G1::batch_normalize(&g1, &mut affine);
        BatchTerms {
            affine,
            variable_g2,
        }
    }

    /// Verifies a batch using fixed G2 terms cached for its verifying key.
    pub fn verify_prepared<R: Rng + CryptoRng>(
        self,
        mut rng: R,
        key: &PreparedBatchVerifyingKey<'_, E>,
    ) -> Result<(), VerificationError> {
        self.check_key(key.vk)?;
        if self.is_empty() {
            return Ok(());
        }
        let terms = self.randomize_terms(&mut rng, key.vk, true);
        if E::multi_miller_loop(&terms.with_key(key)).final_exponentiation_vartime()
            == E::Gt::identity()
        {
            Ok(())
        } else {
            Err(VerificationError::InvalidProof)
        }
    }

    /// Verifies two proof statements in one Miller loop and final exponentiation.
    ///
    /// Exactly one proof across both batches has unit weight. Every other proof receives an
    /// independent nonzero random weight, preventing cancellation between the two statements.
    pub fn verify_joint_prepared<R: Rng + CryptoRng>(
        self,
        other: Self,
        mut rng: R,
        key: &PreparedBatchVerifyingKey<'_, E>,
        other_key: &PreparedBatchVerifyingKey<'_, E>,
    ) -> Result<(), VerificationError> {
        self.check_key(key.vk)?;
        other.check_key(other_key.vk)?;
        if self.is_empty() {
            return other.verify_prepared(rng, other_key);
        }
        if other.is_empty() {
            return self.verify_prepared(rng, key);
        }
        let first = self.randomize_terms(&mut rng, key.vk, true);
        let second = other.randomize_terms(&mut rng, other_key.vk, false);
        let mut terms = first.with_key(key);
        terms.extend(second.with_key(other_key));
        if E::multi_miller_loop(&terms).final_exponentiation_vartime() == E::Gt::identity() {
            Ok(())
        } else {
            Err(VerificationError::InvalidProof)
        }
    }
}
