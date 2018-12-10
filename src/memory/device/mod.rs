use crate::error::CudaResult;
use cuda_sys::cuda::CUstream;

mod device_box;
mod device_buffer;
mod device_slice;

pub use self::device_box::*;
pub use self::device_buffer::*;
pub use self::device_slice::*;

/// Sealed trait implemented by types which can be the source or destination when copying data
/// to/from the device or from one device allocation to another.
pub trait CopyDestination<O: ?Sized>: crate::private::Sealed {
    /// Copy data from `source`. `source` must be the same size as `self`.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    fn copy_from(&mut self, source: &O) -> CudaResult<()>;

    /// Copy data to `dest`. `dest` must be the same size as `self`.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    fn copy_to(&self, dest: &mut O) -> CudaResult<()>;
}

/// Sealed trait implemented by types which can be the source or destination when copying data
/// asynchronously to/from the device or from one device allocation to another.
pub trait AsyncCopyDestination<O: ?Sized>: crate::private::Sealed {
    /// Asynchronously copy data from `source`. `source` must be the same size as `self`.
    ///
    /// [TODO] Why unsafe
    ///
    /// [TODO] Also host memory must be page-locked
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    unsafe fn async_copy_from(&mut self, source: &O, stream: CUstream) -> CudaResult<()>;

    /// Asynchronously copy data to `dest`. `dest` must be the same size as `self`.
    ///
    /// [TODO] Why unsafe
    ///
    /// [TODO] Also host memory must be page-locked
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, return the error.
    unsafe fn async_copy_to(&self, dest: &mut O, stream: CUstream) -> CudaResult<()>;
}
