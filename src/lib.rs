use pyo3::prelude::*;
// use pyo3::types::*;
use numpy::*;
use ndarray::*;

/// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

/// Some RLGym functions that were converted into Rust
#[pymodule]
fn rlgym_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(quat_to_rot_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(norm_func, m)?)?;
    Ok(())
}

/// Quat to rot matrix calculation for RLGym using list
#[pyfunction]
#[pyo3(text_signature = "Takes quat list and outputs matrix in ndarray type")]
fn quat_to_rot_mtx(nums: Vec<f64>, py: Python) -> PyResult<&PyArray2<f64>> {
    let mut theta = Array2::<f64>::zeros((3, 3));

    let norm_vec: Vec<f64> = nums.clone()
                                 .into_iter()
                                 .map(|x: f64| x.powf(2.))
                                 .collect();
    let norm: f64 = norm_vec.iter()
                            .sum();
    let s: f64 = 1.0 / norm;

    let w: &f64 = &nums[0];
    let x: &f64 = &nums[1];
    let y: &f64 = &nums[2];
    let z: &f64 = &nums[3];

    // front direction
    theta[[0, 0]] = 1. - 2. * s * (y * y + z * z);
    theta[[1, 0]] = 2. * s * (x * y + z * w);
    theta[[2, 0]] = 2. * s * (x * z - y * w);

    // left direction
    theta[[0, 1]] = 2. * s * (x * y - z * w);
    theta[[1, 1]] = 1. - 2. * s * (x * x + z * z);
    theta[[2, 1]] = 2. * s * (y * z + x * w);

    // up direction
    theta[[0, 2]] = 2. * s * (x * z + y * w);
    theta[[1, 2]] = 2. * s * (y * z - x * w);
    theta[[2, 2]] = 1. - 2. * s * (x * x + y * y);

    // let theta_arr = theta.to_pyarray(py);

    Ok(theta.to_pyarray(py))
}

/// Norm func that takes list
#[pyfunction]
#[pyo3(text_signature = "takes list and norms it")]
fn norm_func(nums: Vec<f64>) -> PyResult<f64> {
    let norm_val: f64 = nums.clone()
                                 .into_iter()
                                 .map(|x: f64| x.powf(2.))
                                 .collect::<Vec<f64>>()
                                 .iter()
                                 .sum::<f64>()
                                 .sqrt();
    Ok(norm_val)
}