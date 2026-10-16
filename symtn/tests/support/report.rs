// An error as a chain reporter prints it: each `Display` along `source()`,
// outermost first. A facade wrapper names only its own layer, so a test that
// looks for a cause's wording reads the whole chain.

#[allow(dead_code)]
pub fn report(error: &(dyn std::error::Error + 'static)) -> String {
    std::iter::successors(Some(error), |&error| error.source())
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

/// [`report`] as a method, for `result.unwrap_err().report()`.
#[allow(dead_code)]
pub trait Report {
    fn report(&self) -> String;
}

impl<E: std::error::Error + 'static> Report for E {
    fn report(&self) -> String {
        report(self)
    }
}
