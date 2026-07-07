// RISC Zero guest running MLIR-ladder-compiled EVM landmine functions.
//
// The contract code arrives as a static library of RV32IM objects emitted by
// the solc MLIR pipeline (Target/RISCV) with by-pointer `__test_*` wrappers;
// this guest is the thin ERHI-precursor shim: it reads test vectors from the
// host, calls the compiled functions, and commits every 32-byte result to
// the journal. The host owns the expected values.
//
// Input format (one Vec<u8> via env::read):
//   [u32 count] then per vector: [u32 func_index][u32 argc][argc * 32 bytes]
// Words are 32-byte little-endian-limb i256 values (LLVM memory layout).

use risc0_zkvm::guest::env;

extern "C" {
    // Two-arg wrappers.
    fn __test_div2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_sdiv2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_mod2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_smod2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_exp2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_shl2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_shr2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_sar2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_byte2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_se2(out: *mut u8, a: *const u8, b: *const u8);
    fn __test_addmul(out: *mut u8, a: *const u8, b: *const u8);
    // Three-arg wrappers.
    fn __test_am3(out: *mut u8, a: *const u8, b: *const u8, c: *const u8);
    fn __test_mm3(out: *mut u8, a: *const u8, b: *const u8, c: *const u8);
}

fn main() {
    let input: Vec<u8> = env::read();
    let mut cursor = 0usize;

    let read_u32 = |data: &[u8], pos: &mut usize| -> u32 {
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    };

    let count = read_u32(&input, &mut cursor);
    for _ in 0..count {
        let func = read_u32(&input, &mut cursor);
        let argc = read_u32(&input, &mut cursor) as usize;
        let mut args = [[0u8; 32]; 3];
        for arg in args.iter_mut().take(argc) {
            arg.copy_from_slice(&input[cursor..cursor + 32]);
            cursor += 32;
        }
        let mut out = [0u8; 32];

        unsafe {
            let o = out.as_mut_ptr();
            let a = args[0].as_ptr();
            let b = args[1].as_ptr();
            let c = args[2].as_ptr();
            match func {
                0 => __test_div2(o, a, b),
                1 => __test_sdiv2(o, a, b),
                2 => __test_mod2(o, a, b),
                3 => __test_smod2(o, a, b),
                4 => __test_exp2(o, a, b),
                5 => __test_shl2(o, a, b),
                6 => __test_shr2(o, a, b),
                7 => __test_sar2(o, a, b),
                8 => __test_byte2(o, a, b),
                9 => __test_se2(o, a, b),
                10 => __test_addmul(o, a, b),
                11 => __test_am3(o, a, b, c),
                12 => __test_mm3(o, a, b, c),
                _ => panic!("unknown function index {func}"),
            }
        }
        env::commit_slice(&out);
    }
}
