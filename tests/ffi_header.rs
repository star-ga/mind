#![cfg(feature = "ffi-c")]

#[test]
fn header_contains_expected_symbols() {
    let header = mind::ffi::header::generate_header();
    assert!(header.contains("MindTensor"));
    assert!(header.contains("mind_infer"));
}
