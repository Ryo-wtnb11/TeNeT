use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemporaryLabel(String);

impl TemporaryLabel {
    pub fn new(label: impl Into<String>) -> Self {
        Self(label.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<char> for TemporaryLabel {
    fn from(value: char) -> Self {
        Self(value.to_string())
    }
}

impl From<&str> for TemporaryLabel {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl Display for TemporaryLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(usize);

impl TensorId {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorAxis {
    tensor: TensorId,
    axis: usize,
}

impl TensorAxis {
    pub fn new(tensor: TensorId, axis: usize) -> Self {
        Self { tensor, axis }
    }

    pub fn tensor(self) -> TensorId {
        self.tensor
    }

    pub fn axis(self) -> usize {
        self.axis
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabelOccurrence {
    label: TemporaryLabel,
    axis: TensorAxis,
}

impl LabelOccurrence {
    pub fn new(label: TemporaryLabel, axis: TensorAxis) -> Self {
        Self { label, axis }
    }

    pub fn label(&self) -> &TemporaryLabel {
        &self.label
    }

    pub fn axis(&self) -> TensorAxis {
        self.axis
    }
}
