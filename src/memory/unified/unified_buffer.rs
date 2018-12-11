use crate::error::*;
use crate::memory::malloc::{cuda_free_unified, cuda_malloc_unified};
use crate::memory::{DeviceCopy, UnifiedPointer};
use std::convert::{AsMut, AsRef};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

/// Fixed-size buffer in unified memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more details on unified memory.
#[derive(Debug)]
pub struct UnifiedBuffer<T: DeviceCopy> {
    buf: UnifiedPointer<T>,
    capacity: usize,
}
impl<T: DeviceCopy + Clone> UnifiedBuffer<T> {
    /// Allocate a new unified buffer large enough to hold `size` `T`'s and initialized with
    /// clones of `value`.
    ///
    /// # Errors:
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
    /// buffer[0] = 1;
    /// ```
    pub fn new(value: &T, size: usize) -> CudaResult<Self> {
        unsafe {
            let mut uninit = UnifiedBuffer::uninitialized(size)?;
            for x in 0..size {
                *uninit.get_unchecked_mut(x) = value.clone();
            }
            Ok(uninit)
        }
    }

    /// Allocate a new unified buffer of the same size as `slice`, initialized with a clone of
    /// the data in `slice`.
    ///
    /// # Errors:
    ///
    /// If the allocation fails, returns the error from CUDA.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let values = [0u64; 5];
    /// let mut buffer = UnifiedBuffer::from_slice(&values).unwrap();
    /// buffer[0] = 1;
    /// ```
    pub fn from_slice(slice: &[T]) -> CudaResult<Self> {
        unsafe {
            let mut uninit = UnifiedBuffer::uninitialized(slice.len())?;
            for (i, x) in slice.iter().enumerate() {
                *uninit.get_unchecked_mut(i) = x.clone();
            }
            Ok(uninit)
        }
    }
}
impl<T: DeviceCopy> UnifiedBuffer<T> {
    /// Allocate a new unified buffer large enough to hold `size` `T`'s, but without
    /// initializing the contents.
    ///
    /// # Errors:
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety:
    ///
    /// The caller must ensure that the contents of the buffer are initialized before reading from
    /// the buffer.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut buffer = unsafe { UnifiedBuffer::uninitialized(5).unwrap() };
    /// for i in buffer.iter_mut() {
    ///     *i = 0u64;
    /// }
    /// ```
    pub unsafe fn uninitialized(size: usize) -> CudaResult<Self> {
        let bytes = size
            .checked_mul(mem::size_of::<T>())
            .ok_or(CudaError::InvalidMemoryAllocation)?;

        let ptr = if bytes > 0 {
            cuda_malloc_unified(bytes)?
        } else {
            UnifiedPointer::wrap(ptr::NonNull::dangling().as_ptr() as *mut T)
        };
        Ok(UnifiedBuffer {
            buf: ptr,
            capacity: size,
        })
    }

    /// Extracts a slice containing the entire buffer.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
    /// let sum : u64 = buffer.as_slice().iter().sum();
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Extracts a mutable slice of the entire buffer.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
    /// for i in buffer.as_mut_slice() {
    ///     *i = 12u64;
    /// }
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Returns a `UnifiedPointer<T>` to the buffer.
    ///
    /// The caller must ensure that the buffer outlives the returned pointer, or it will end up
    /// pointing to garbage.
    ///
    /// Modifying the buffer is guaranteed not to cause its buffer to be reallocated, so pointers
    /// cannot be invalidated in that manner, but other types may be added in the future which can
    /// reallocate.
    pub fn as_unified_ptr(&mut self) -> UnifiedPointer<T> {
        self.buf
    }

    /// Creates a `UnifiedBuffer<T>` directly from the raw components of another unified buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `UnifiedBuffer` or
    /// [`cuda_malloc_unified`](fn.cuda_malloc_unified.html).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the CUDA driver's
    /// internal data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `UnifiedBuffer<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use std::mem;
    /// use rustacuda::memory::*;
    ///
    /// let mut buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
    /// let ptr = buffer.as_unified_ptr();
    /// let size = buffer.len();
    ///
    /// mem::forget(buffer);
    ///
    /// let buffer = unsafe { UnifiedBuffer::from_raw_parts(ptr, size) };
    /// ```
    pub unsafe fn from_raw_parts(ptr: UnifiedPointer<T>, capacity: usize) -> UnifiedBuffer<T> {
        UnifiedBuffer { buf: ptr, capacity }
    }

    /// Destroy a `UnifiedBuffer`, returning an error.
    ///
    /// Deallocating unified memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    ///
    /// # Example:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = UnifiedBuffer::from_slice(&[10u32, 20, 30]).unwrap();
    /// match UnifiedBuffer::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, buf)) => {
    ///         println!("Failed to destroy buffer: {:?}", e);
    ///         // Do something with buf
    ///     },
    /// }
    /// ```
    pub fn drop(mut uni_buf: UnifiedBuffer<T>) -> DropResult<UnifiedBuffer<T>> {
        if uni_buf.buf.is_null() {
            return Ok(());
        }

        if uni_buf.capacity > 0 && mem::size_of::<T>() > 0 {
            let capacity = uni_buf.capacity;
            let ptr = mem::replace(&mut uni_buf.buf, UnifiedPointer::null());
            unsafe {
                match cuda_free_unified(ptr) {
                    Ok(()) => {
                        mem::forget(uni_buf);
                        Ok(())
                    }
                    Err(e) => Err((e, UnifiedBuffer::from_raw_parts(ptr, capacity))),
                }
            }
        } else {
            Ok(())
        }
    }
}

impl<T: DeviceCopy> AsRef<[T]> for UnifiedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}
impl<T: DeviceCopy> AsMut<[T]> for UnifiedBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<T: DeviceCopy> Deref for UnifiedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf.as_raw();
            slice::from_raw_parts(p, self.capacity)
        }
    }
}
impl<T: DeviceCopy> DerefMut for UnifiedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf.as_raw_mut();
            slice::from_raw_parts_mut(ptr, self.capacity)
        }
    }
}
impl<T: DeviceCopy> Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        if self.buf.is_null() {
            return;
        }

        if self.capacity > 0 && mem::size_of::<T>() > 0 {
            // No choice but to panic if this fails.
            unsafe {
                let ptr = mem::replace(&mut self.buf, UnifiedPointer::null());
                cuda_free_unified(ptr).expect("Failed to deallocate CUDA unified memory.");
            }
        }
        self.capacity = 0;
    }
}

#[cfg(test)]
mod test_unified_buffer {
    use super::*;
    use std::mem;

    #[derive(Clone, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_new() {
        let _context = crate::quick_init().unwrap();
        let val = 0u64;
        let mut buffer = UnifiedBuffer::new(&val, 5).unwrap();
        buffer[0] = 1;
    }

    #[test]
    fn test_from_slice() {
        let _context = crate::quick_init().unwrap();
        let values = [0u64; 10];
        let mut buffer = UnifiedBuffer::from_slice(&values).unwrap();
        for i in buffer[0..3].iter_mut() {
            *i = 10;
        }
    }

    #[test]
    fn from_raw_parts() {
        let _context = crate::quick_init().unwrap();
        let mut buffer = UnifiedBuffer::new(&0u64, 5).unwrap();
        buffer[2] = 1;
        let ptr = buffer.as_unified_ptr();
        let len = buffer.len();
        mem::forget(buffer);

        let buffer = unsafe { UnifiedBuffer::from_raw_parts(ptr, len) };
        assert_eq!(&[0u64, 0, 1, 0, 0], buffer.as_slice());
        drop(buffer);
    }

    #[test]
    fn zero_length_buffer() {
        let _context = crate::quick_init().unwrap();
        let buffer = UnifiedBuffer::new(&0u64, 0).unwrap();
        drop(buffer);
    }

    #[test]
    fn zero_size_type() {
        let _context = crate::quick_init().unwrap();
        let buffer = UnifiedBuffer::new(&ZeroSizedType, 10).unwrap();
        drop(buffer);
    }

    #[test]
    fn overflows_usize() {
        let _context = crate::quick_init().unwrap();
        let err = UnifiedBuffer::new(&0u64, ::std::usize::MAX - 1).unwrap_err();
        assert_eq!(CudaError::InvalidMemoryAllocation, err);
    }
}
