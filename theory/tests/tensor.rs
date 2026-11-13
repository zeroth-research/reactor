//! Tests for `PyTensor` backed by torch.

#![cfg(feature = "torch")]

use pyo3_tch::tch;
use theory::PyTensor;

#[test]
fn small_tensors_print_entirely_on_one_line() {
    let t: PyTensor = tch::Tensor::from_slice(&[1.5f64, 2.0, 3.0])
        .reshape([1, 3])
        .into();
    assert_eq!(format!("{t}"), "[[1.5, 2, 3]]");

    let b: PyTensor = tch::Tensor::from_slice(&[true, false]).into();
    assert_eq!(format!("{b}"), "[true, false]");

    let s: PyTensor = tch::Tensor::from(4i64).into();
    assert_eq!(format!("{s}"), "4");
}

#[test]
fn large_tensors_abbreviate_with_dots() {
    let t: PyTensor = tch::Tensor::arange(24, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape([2, 12])
        .into();
    assert_eq!(
        format!("{t}"),
        "[[0, 1, 2, ..., 9, 10, 11], [12, 13, 14, ..., 21, 22, 23]]"
    );

    // an oversized outer dimension truncates as well
    let v: PyTensor = tch::Tensor::arange(20, (tch::Kind::Int64, tch::Device::Cpu)).into();
    assert_eq!(format!("{v}"), "[0, 1, 2, ..., 17, 18, 19]");
}

#[test]
fn output_never_spans_multiple_lines() {
    let t: PyTensor = tch::Tensor::rand([4, 4, 4], (tch::Kind::Float, tch::Device::Cpu)).into();
    assert!(!format!("{t}").contains('\n'));
}
