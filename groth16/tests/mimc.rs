// For randomness (during paramgen and proof generation)
use rand::{RngExt, SeedableRng, TryCryptoRng, TryRng, rng, rngs::StdRng, seq::IndexedRandom};

// For benchmarking
use std::time::{Duration, Instant};

// Bring in some tools for using finite fiels
use ff::Field;

// We're going to use the BLS12-381 pairing-friendly elliptic curve.
use bls12_381::{Bls12, G1Projective, G2Projective, Scalar};

// We're going to use the Groth16 proving system.
use groth16::{
    PreparedVerifyingKey, Proof, VerifyingKey, batch, create_random_proof,
    generate_random_parameters, prepare_verifying_key, verify_proof,
};

use bellman::VerificationError;

mod common;

use common::*;

#[test]
fn test_mimc() {
    // This may not be cryptographically safe, use
    // `OsRng` (for example) in production software.
    let mut rng = rng();

    // Generate the MiMC round constants
    let constants = (0..MIMC_ROUNDS)
        .map(|_| Scalar::random(&mut rng))
        .collect::<Vec<_>>();

    println!("Creating parameters...");

    // Create parameters for our circuit
    let params = {
        let c = MiMCDemo {
            xl: None,
            xr: None,
            constants: &constants,
        };

        generate_random_parameters::<Bls12, _, _>(c, &mut rng).unwrap()
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    println!("Creating proofs...");

    // Let's benchmark stuff!
    const SAMPLES: u32 = 50;
    let mut total_proving = Duration::new(0, 0);
    let mut total_verifying = Duration::new(0, 0);

    // Just a place to put the proof data, so we can
    // benchmark deserialization.
    let mut proof_vec = vec![];

    for _ in 0..SAMPLES {
        // Generate a random preimage and compute the image
        let xl = Scalar::random(&mut rng);
        let xr = Scalar::random(&mut rng);
        let image = mimc(xl, xr, &constants);

        proof_vec.truncate(0);

        let start = Instant::now();
        {
            // Create an instance of our circuit (with the
            // witness)
            let c = MiMCDemo {
                xl: Some(xl),
                xr: Some(xr),
                constants: &constants,
            };

            // Create a groth16 proof with our parameters.
            let proof = create_random_proof(c, &params, &mut rng).unwrap();

            proof.write(&mut proof_vec).unwrap();
        }

        total_proving += start.elapsed();

        let start = Instant::now();
        let proof = Proof::read(&proof_vec[..]).unwrap();
        // Check the proof
        assert!(verify_proof(&pvk, &proof, &[image]).is_ok());
        total_verifying += start.elapsed();
    }
    let proving_avg = total_proving / SAMPLES;
    let proving_avg =
        proving_avg.subsec_nanos() as f64 / 1_000_000_000f64 + (proving_avg.as_secs() as f64);

    let verifying_avg = total_verifying / SAMPLES;
    let verifying_avg =
        verifying_avg.subsec_nanos() as f64 / 1_000_000_000f64 + (verifying_avg.as_secs() as f64);

    println!("Average proving time: {:?} seconds", proving_avg);
    println!("Average verifying time: {:?} seconds", verifying_avg);
}

#[test]
fn batch_verify() {
    let mut rng = rng();

    let mut batch = batch::Verifier::new();

    // Generate the MiMC round constants
    let constants = (0..MIMC_ROUNDS)
        .map(|_| Scalar::random(&mut rng))
        .collect::<Vec<_>>();

    println!("Creating parameters...");

    // Create parameters for our circuit
    let params = {
        let c = MiMCDemo {
            xl: None,
            xr: None,
            constants: &constants,
        };

        generate_random_parameters::<Bls12, _, _>(c, &mut rng).unwrap()
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    println!("Creating proofs...");

    // Let's benchmark stuff!
    const SAMPLES: u32 = 50;
    let mut total_proving = Duration::new(0, 0);
    let mut total_verifying = Duration::new(0, 0);

    // Just a place to put the proof data, so we can
    // benchmark deserialization.
    let mut proof_vec = vec![];

    for _ in 0..SAMPLES {
        // Generate a random preimage and compute the image
        let xl = Scalar::random(&mut rng);
        let xr = Scalar::random(&mut rng);
        let image = mimc(xl, xr, &constants);

        proof_vec.truncate(0);

        let start = Instant::now();
        {
            // Create an instance of our circuit (with the
            // witness)
            let c = MiMCDemo {
                xl: Some(xl),
                xr: Some(xr),
                constants: &constants,
            };

            // Create a groth16 proof with our parameters.
            let proof = create_random_proof(c, &params, &mut rng).unwrap();

            proof.write(&mut proof_vec).unwrap();
        }

        total_proving += start.elapsed();

        let start = Instant::now();
        let proof = Proof::read(&proof_vec[..]).unwrap();

        // Check the proof
        assert!(verify_proof(&pvk, &proof, &[image]).is_ok());

        total_verifying += start.elapsed();

        // Queue the proof and inputs for batch verification.
        batch.queue((proof, [image].into()));
    }

    let mut batch_verifying = Duration::new(0, 0);
    let batch_start = Instant::now();

    // Verify this batch for this specific verifying key
    assert!(batch.verify(rng, &params.vk).is_ok());

    batch_verifying += batch_start.elapsed();

    let proving_avg = total_proving / SAMPLES;
    let proving_avg =
        proving_avg.subsec_nanos() as f64 / 1_000_000_000f64 + (proving_avg.as_secs() as f64);

    let verifying_avg = total_verifying / SAMPLES;
    let verifying_avg =
        verifying_avg.subsec_nanos() as f64 / 1_000_000_000f64 + (verifying_avg.as_secs() as f64);

    let batch_amortized = batch_verifying / SAMPLES;
    let batch_amortized = batch_amortized.subsec_nanos() as f64 / 1_000_000_000f64
        + (batch_amortized.as_secs() as f64);

    println!("Average proving time: {:?} seconds", proving_avg);
    println!("Average verifying time: {:?} seconds", verifying_avg);
    println!(
        "Amortized batch verifying time: {:?} seconds",
        batch_amortized
    );
}

/// Exercises the `multicore` batch verifier.
///
/// `Verifier::verify_multicore` is gated behind the `multicore` feature, so it is only
/// compiled (and only reachable) when that feature is on. Without a test under the
/// feature, the path can rot unnoticed while the rest of the crate keeps building.
#[test]
#[cfg(feature = "multicore")]
fn batch_verify_multicore() {
    const SAMPLES: u32 = 4;

    let mut rng = rng();

    let constants = (0..MIMC_ROUNDS)
        .map(|_| Scalar::random(&mut rng))
        .collect::<Vec<_>>();

    let params = {
        let c = MiMCDemo {
            xl: None,
            xr: None,
            constants: &constants,
        };

        generate_random_parameters::<Bls12, _, _>(c, &mut rng).unwrap()
    };

    let mut valid = batch::Verifier::new();
    let mut tampered = batch::Verifier::new();

    for _ in 0..SAMPLES {
        let xl = Scalar::random(&mut rng);
        let xr = Scalar::random(&mut rng);
        let image = mimc(xl, xr, &constants);

        let c = MiMCDemo {
            xl: Some(xl),
            xr: Some(xr),
            constants: &constants,
        };
        let proof = create_random_proof(c, &params, &mut rng).unwrap();

        valid.queue((proof.clone(), [image].into()));
        // Same proof, but bound to a public input it does not attest to.
        tampered.queue((proof, [image + Scalar::ONE].into()));
    }

    assert!(valid.verify_multicore(&params.vk).is_ok());
    // Specifically `InvalidProof`, not `InvalidVerifyingKey`: the tampered batch must be
    // rejected by the pairing check, not waved away by the input-length precondition.
    assert!(matches!(
        tampered.verify_multicore(&params.vk),
        Err(VerificationError::InvalidProof)
    ));
}

type Item = (Proof<Bls12>, Vec<Scalar>);

/// Returns a verifying key and a few valid proofs, generated deterministically.
fn batch_fixture() -> (VerifyingKey<Bls12>, PreparedVerifyingKey<Bls12>, Vec<Item>) {
    let mut rng = StdRng::seed_from_u64(0);
    let constants = (0..MIMC_ROUNDS)
        .map(|_| Scalar::random(&mut rng))
        .collect::<Vec<_>>();
    let params = generate_random_parameters::<Bls12, _, _>(
        MiMCDemo {
            xl: None,
            xr: None,
            constants: &constants,
        },
        &mut rng,
    )
    .unwrap();
    let items = (0..4)
        .map(|_| {
            let xl = Scalar::random(&mut rng);
            let xr = Scalar::random(&mut rng);
            let circuit = MiMCDemo {
                xl: Some(xl),
                xr: Some(xr),
                constants: &constants,
            };
            let proof = create_random_proof(circuit, &params, &mut rng).unwrap();
            (proof, vec![mimc(xl, xr, &constants)])
        })
        .collect();
    let pvk = prepare_verifying_key(&params.vk);
    (params.vk, pvk, items)
}

/// Makes an item invalid, in one of four ways selected by `kind`.
fn corrupt((proof, inputs): &mut Item, kind: usize) {
    match kind % 4 {
        0 => proof.a = (proof.a + G1Projective::generator()).into(),
        1 => proof.b = (proof.b + G2Projective::generator()).into(),
        2 => proof.c = (proof.c + G1Projective::generator()).into(),
        _ => inputs[0] += Scalar::ONE,
    }
}

/// Returns whether a batch result is the expected success, or a proof failure.
fn is_expected(result: Result<(), VerificationError>, valid: bool) -> bool {
    matches!(
        (result, valid),
        (Ok(()), true) | (Err(VerificationError::InvalidProof), false)
    )
}

/// Asserts that batch verification accepts `items` exactly when each item verifies alone.
fn assert_batch_matches_single(
    vk: &VerifyingKey<Bls12>,
    pvk: &PreparedVerifyingKey<Bls12>,
    items: &[Item],
    rng: &mut StdRng,
) {
    let valid = items
        .iter()
        .all(|(proof, inputs)| verify_proof(pvk, proof, inputs).is_ok());
    let batch = || {
        let mut batch = batch::Verifier::new();
        for item in items {
            batch.queue(item.clone());
        }
        batch
    };
    assert!(is_expected(batch().verify(rng, vk), valid));
    #[cfg(feature = "multicore")]
    assert!(is_expected(batch().verify_multicore(vk), valid));
}

#[test]
fn batch_rejects_each_invalid_proof() {
    let (vk, pvk, proofs) = batch_fixture();
    let mut rng = StdRng::seed_from_u64(1);
    // Cover empty and singleton batches, and both sides of the multicore chunk
    // boundary. The first item has the unit weight.
    for size in [0, 1, 2, 8, 9, 17] {
        let mut items = proofs
            .iter()
            .cycle()
            .take(size)
            .cloned()
            .collect::<Vec<_>>();
        assert_batch_matches_single(&vk, &pvk, &items, &mut rng);
        for position in 0..size {
            let valid = items[position].clone();
            corrupt(&mut items[position], size + position);
            let (proof, inputs) = &items[position];
            assert!(verify_proof(&pvk, proof, inputs).is_err());
            assert_batch_matches_single(&vk, &pvk, &items, &mut rng);
            items[position] = valid;
        }
    }
}

#[test]
fn batch_matches_single_verification() {
    let (vk, pvk, proofs) = batch_fixture();
    for seed in 0..64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let size = rng.random_range(1..=20);
        let mut items = (0..size)
            .map(|_| proofs.choose(&mut rng).unwrap().clone())
            .collect::<Vec<_>>();
        for _ in 0..rng.random_range(0..3) {
            let position = rng.random_range(0..size);
            corrupt(&mut items[position], rng.random_range(0..4));
        }
        assert_batch_matches_single(&vk, &pvk, &items, &mut rng);
    }
}

#[test]
fn batch_rejects_cancelling_errors() {
    let (vk, pvk, proofs) = batch_fixture();
    let mut rng = StdRng::seed_from_u64(2);
    // These errors cancel if both items get the same weight, for example if each
    // multicore chunk gave its first item the unit weight.
    for (first, second) in [(0, 1), (0, 8), (1, 9), (3, 16)] {
        let mut items = proofs.iter().cycle().take(17).cloned().collect::<Vec<_>>();
        items[first].0.c = (items[first].0.c + G1Projective::generator()).into();
        items[second].0.c = (items[second].0.c - G1Projective::generator()).into();
        assert_batch_matches_single(&vk, &pvk, &items, &mut rng);
    }
}

#[test]
fn batch_accepts_empty_batch() {
    let (mut vk, _, _) = batch_fixture();
    // A key with no `ic` points can be deserialized, and has no valid proofs.
    for ic_len in [1, 0] {
        vk.ic.truncate(ic_len);
        assert!(batch::Verifier::new().verify(rng(), &vk).is_ok());
        #[cfg(feature = "multicore")]
        assert!(batch::Verifier::new().verify_multicore(&vk).is_ok());
    }
}

/// Supplies fixed bytes, so that a test can choose the batch weights.
struct FixedBytes(std::vec::IntoIter<u8>);

impl TryRng for FixedBytes {
    type Error = std::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut bytes = [0; 4];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut bytes = [0; 8];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        for byte in dest {
            *byte = self.0.next().expect("the test supplies every random byte");
        }
        Ok(())
    }
}

impl TryCryptoRng for FixedBytes {}

#[test]
fn batch_retries_zero_weight() {
    let (vk, _, proofs) = batch_fixture();
    for valid in [true, false] {
        let mut items = proofs[..2].to_vec();
        if !valid {
            corrupt(&mut items[1], 3);
        }
        let mut batch = batch::Verifier::new();
        for item in items {
            batch.queue(item);
        }
        // The unit-weight item draws nothing. The other item first draws zero, which
        // would drop it from the batch, then draws the largest weight, 2^128 - 1.
        let mut rng = FixedBytes([[0; 16], [0xff; 16]].concat().into_iter());
        assert!(is_expected(batch.verify(&mut rng, &vk), valid));
        assert_eq!(rng.0.len(), 0);
    }
}

#[test]
#[cfg(feature = "multicore")]
fn adaptive_chunks_match_single_across_thread_counts() {
    let (vk, pvk, proofs) = batch_fixture();
    for threads in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut rng = StdRng::seed_from_u64(3);
                for size in [0, 1, 2, 7, 8, 9, 17, 33] {
                    let mut items = proofs.iter().cycle().take(size).cloned().collect::<Vec<_>>();
                    assert_batch_matches_single(&vk, &pvk, &items, &mut rng);
                    if size > 1 {
                        // Opposite errors cannot cancel across independently weighted chunks.
                        items[0].0.c = (items[0].0.c + G1Projective::generator()).into();
                        items[size - 1].0.c =
                            (items[size - 1].0.c - G1Projective::generator()).into();
                        assert_batch_matches_single(&vk, &pvk, &items, &mut rng);
                    }
                }
            });
    }
}
