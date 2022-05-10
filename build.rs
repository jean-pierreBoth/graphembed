
#[cfg(not(feature="intel-mkl-static"))]
fn main() {
    // if not mkl we need to specify we ask for lapacke
    println!("cargo:rustc-link-lib=lapacke");
}

#[cfg(feature="intel-mkl-static")]
fn main() {
}
