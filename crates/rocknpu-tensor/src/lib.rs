use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    RowMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    pub rows: usize,
    pub cols: usize,
}

impl MatrixShape {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
    pub fn elements(self) -> Result<usize, TensorError> {
        self.rows
            .checked_mul(self.cols)
            .ok_or(TensorError::SizeOverflow)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ZeroDimension,
    SizeOverflow,
    LengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "tensor dimensions must be non-zero"),
            Self::SizeOverflow => write!(f, "tensor element count overflow"),
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "tensor length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}
impl std::error::Error for TensorError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    shape: MatrixShape,
    layout: TensorLayout,
    values: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn from_vec(rows: usize, cols: usize, values: Vec<T>) -> Result<Self, TensorError> {
        if rows == 0 || cols == 0 {
            return Err(TensorError::ZeroDimension);
        }
        let shape = MatrixShape::new(rows, cols);
        let expected = shape.elements()?;
        if values.len() != expected {
            return Err(TensorError::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            shape,
            layout: TensorLayout::RowMajor,
            values,
        })
    }
    pub const fn shape(&self) -> MatrixShape {
        self.shape
    }
    pub const fn rows(&self) -> usize {
        self.shape.rows
    }
    pub const fn cols(&self) -> usize {
        self.shape.cols
    }
    pub const fn layout(&self) -> TensorLayout {
        self.layout
    }
    pub fn values(&self) -> &[T] {
        &self.values
    }
    pub fn into_vec(self) -> Vec<T> {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_validates_shape_and_length() {
        assert!(Matrix::<u8>::from_vec(2, 3, vec![0; 6]).is_ok());
        assert!(matches!(
            Matrix::<u8>::from_vec(2, 3, vec![0; 5]),
            Err(TensorError::LengthMismatch { .. })
        ));
        assert!(matches!(
            Matrix::<u8>::from_vec(0, 3, vec![]),
            Err(TensorError::ZeroDimension)
        ));
    }
}
