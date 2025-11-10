fn main() {
    if std::env::var("CARGO_FEATURE_FFI_EXAMPLES").is_ok() {
        println!("cargo:rerun-if-changed=examples/c/min.c");
        cc::Build::new().file("examples/c/min.c").warnings(false).compile("mind_ffi_examples");
    }
}
