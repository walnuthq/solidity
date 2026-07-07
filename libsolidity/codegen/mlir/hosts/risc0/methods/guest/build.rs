// Links the MLIR-ladder-compiled contract objects (contract code + evm-rt)
// into the zkVM guest. The archive is produced by ../../build-guest-lib.sh.
fn main() {
    let lib_dir = std::env::var("EVM_CONTRACT_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/../../lib", std::env::var("CARGO_MANIFEST_DIR").unwrap()));
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=static=evmcontract");
    println!("cargo:rerun-if-env-changed=EVM_CONTRACT_LIB_DIR");
    println!("cargo:rerun-if-changed={lib_dir}/libevmcontract.a");
}
