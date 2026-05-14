fn main() {
    vergen::EmitBuilder::builder()
        .all_git()
        .all_build()
        .all_cargo()
        .all_rustc()
        .emit()
        .unwrap();
}
