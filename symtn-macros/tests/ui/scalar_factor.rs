use symtn_macros::contract;

fn main() {
    let _ = contract!([i; j] := 2.0 * a[i; j]);
}
