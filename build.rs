
#[cfg(not(feature="intel-mkl"))]
fn main() {
    // if not mkl we need to specify we ask for lapacke
    println!("cargo:rustc-link-lib=lapacke");
}