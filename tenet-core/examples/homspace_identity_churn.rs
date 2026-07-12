use std::time::Duration;

use tenet_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1Irrep};

fn main() {
    let count = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let sleep_seconds = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);

    for charge in 0..count {
        let sector = U1Irrep::new(charge as i32).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(sector, 1)], false)]);
        let _ = FusionTreeHomSpace::new(leg(), leg()).id();
    }

    println!("pid={} constructed_homspaces={count}", std::process::id());
    std::thread::sleep(Duration::from_secs(sleep_seconds));
}
