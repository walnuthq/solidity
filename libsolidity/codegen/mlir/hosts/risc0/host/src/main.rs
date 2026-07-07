// RISC Zero host for the MLIR-ladder landmine guest: sends the EVM-semantics
// test vectors, executes the guest (executor = no proving; set PROVE=1 for a
// receipt), checks the journal against the expected values, and reports the
// zkVM cycle count - the metric the SolcRISCV pipeline optimizes.

use evm_risc0_methods::EVM_GUEST_ELF;
use risc0_zkvm::{ExecutorEnv, ExecutorImpl};

const MIN: &str = "8000000000000000000000000000000000000000000000000000000000000000";
const MAX: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

struct Vector {
    func: u32,
    args: Vec<&'static str>,
    expected: &'static str,
}

fn vectors() -> Vec<Vector> {
    let v = |func, args: Vec<&'static str>, expected| Vector { func, args, expected };
    vec![
        v(0, vec!["7", "2"], "3"),
        v(0, vec!["5", "0"], "0"),
        v(1, vec![MIN, MAX], MIN),
        v(
            1,
            vec!["fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa", "2"],
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd",
        ),
        v(2, vec!["9", "4"], "1"),
        v(2, vec!["7", "0"], "0"),
        v(
            3,
            vec!["fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff9", "2"],
            MAX,
        ),
        v(4, vec!["3", "c8"], "c21a937a76f3432ffd73d97e447606b683ecf6f6e4a7ae225bfaff1eaaf8b0a1"),
        v(4, vec!["2", "100"], "0"),
        v(5, vec!["4", "1"], "10"),
        v(5, vec!["12c", "1"], "0"),
        v(6, vec!["4", "100"], "10"),
        v(6, vec!["12c", MAX], "0"),
        v(7, vec!["ff", MIN], MAX),
        v(7, vec!["12c", MIN], MAX),
        v(8, vec!["0", "aa00000000000000000000000000000000000000000000000000000000000000"], "aa"),
        v(8, vec!["1f", "bb"], "bb"),
        v(8, vec!["20", MAX], "0"),
        v(9, vec!["0", "ff"], MAX),
        v(9, vec!["0", "7f"], "7f"),
        v(11, vec![MAX, MAX, "7"], "2"),
        v(11, vec!["1", "2", "0"], "0"),
        v(12, vec![MAX, MAX, "7"], "1"),
        v(10, vec!["2", "5"], "b"),
    ]
}

/// Big-endian hex string -> 32-byte little-endian-limb word (LLVM i256 layout).
fn hex_to_le(hex: &str) -> [u8; 32] {
    let mut be = [0u8; 32];
    let padded = format!("{:0>64}", hex);
    for i in 0..32 {
        be[i] = u8::from_str_radix(&padded[2 * i..2 * i + 2], 16).unwrap();
    }
    be.reverse();
    be
}

fn main() {
    let vecs = vectors();

    let mut input: Vec<u8> = Vec::new();
    input.extend((vecs.len() as u32).to_le_bytes());
    for vector in &vecs {
        input.extend(vector.func.to_le_bytes());
        input.extend((vector.args.len() as u32).to_le_bytes());
        for arg in &vector.args {
            input.extend(hex_to_le(arg));
        }
    }

    let env = ExecutorEnv::builder().write(&input).unwrap().build().unwrap();

    let mut exec = ExecutorImpl::from_elf(env, EVM_GUEST_ELF).expect("guest ELF loads");
    let session = exec.run().expect("guest execution succeeds");
    println!(
        "zkVM execution: {} user cycles, {} total cycles",
        session.user_cycles, session.total_cycles
    );

    let journal = session.journal.expect("journal present").bytes;
    assert_eq!(journal.len(), vecs.len() * 32, "journal size");

    let mut failures = 0;
    for (i, vector) in vecs.iter().enumerate() {
        let got = &journal[i * 32..(i + 1) * 32];
        let want = hex_to_le(vector.expected);
        if got != want {
            println!("[FAIL] vector {i} (func {}): journal mismatch", vector.func);
            failures += 1;
        }
    }
    println!(
        "{}",
        if failures == 0 {
            format!("[PASS] all {} landmine vectors exact inside the RISC Zero zkVM", vecs.len())
        } else {
            format!("[FAIL] {failures} vector mismatches")
        }
    );

    if std::env::var("PROVE").is_ok() {
        use risc0_zkvm::default_prover;
        let env = ExecutorEnv::builder().write(&input).unwrap().build().unwrap();
        let receipt = default_prover().prove(env, EVM_GUEST_ELF).expect("proving succeeds").receipt;
        receipt.verify(evm_risc0_methods::EVM_GUEST_ID).expect("receipt verifies");
        println!("[PASS] receipt generated and verified");
    }

    std::process::exit(if failures == 0 { 0 } else { 1 });
}
