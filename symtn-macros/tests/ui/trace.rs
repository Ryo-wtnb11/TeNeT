use symtn_macros::contract;

fn main() {
    let _ = contract!([i] := a[i, j, j]);
}
