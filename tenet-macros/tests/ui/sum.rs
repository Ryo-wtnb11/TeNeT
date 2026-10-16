use tenet_macros::contract;

fn main() {
    let _ = contract!([i; j] := a[i; j] + b[i; j]);
}
