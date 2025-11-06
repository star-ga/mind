#[cfg(feature = "autodiff")]
#[test]
fn grad_placeholder_runs() {
    let g = mind::autodiff::grad(|| 40 + 2);
    let (v, d) = g();
    assert_eq!(v, 42);
    assert!(d.contains("placeholder"));
}
