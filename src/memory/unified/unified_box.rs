use crate::error::*;
use crate::memory::malloc::{cuda_free_unified, cuda_malloc_unified};
use crate::memory::{DeviceCopy, UnifiedPointer};
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::convert::{AsMut, AsRef};
use std::fmt::{self, Display, Pointer};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::{Deref, DerefMut};

/// A pointer type for heap-allocation in CUDA unified memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more information on unified
/// memory. Should behave equivalently to `std::boxed::Box`, except that the allocated memory can be
/// seamlessly shared between host and device.
#[derive(Debug)]
pub struct UnifiedBox<T: DeviceCopy> {
    ptr: UnifiedPointer<T>,
}
impl<T: DeviceCopy> UnifiedBox<T> {
    /// Allocate unified memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, returns that error.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let five = UnifiedBox::new(5).unwrap();
    /// ```
    pub fn new(val: T) -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(UnifiedBox {
                ptr: UnifiedPointer::null(),
            })
        } else {
            let mut ubox = unsafe { UnifiedBox::uninitialized()? };
            *ubox = val;
            Ok(ubox)
        }
    }

    /// Allocate unified memory without initializing it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety:
    ///
    /// Since the backing memory is not initialized, this function is not safe. The caller must
    /// ensure that the backing memory is set to a valid value before it is read, else undefined
    /// behavior may occur.
    ///
    /// # Errors:
    ///
    /// If a CUDA error occurs, returns that error.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut five = unsafe{ UnifiedBox::uninitialized().unwrap() };
    /// *five = 5u64;
    /// ```
    pub unsafe fn uninitialized() -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(UnifiedBox {
                ptr: UnifiedPointer::null(),
            })
        } else {
            let ptr = cuda_malloc_unified(1)?;
            Ok(UnifiedBox { ptr })
        }
    }

    /// Constructs a UnifiedBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// UnifiedBox. The UnifiedBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call.
    ///
    /// # Safety:
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = UnifiedBox::new(5).unwrap();
    /// let ptr = UnifiedBox::into_unified(x).as_raw_mut();
    /// let x = unsafe { UnifiedBox::from_raw(ptr) };
    /// ```
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        UnifiedBox {
            ptr: UnifiedPointer::wrap(ptr),
        }
    }

    /// Constructs a UnifiedBox from a UnifiedPointer.
    ///
    /// After calling this function, the pointer and the memory it points to is owned by the
    /// UnifiedBox. The UnifiedBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call, such as one taken from `UnifiedBox::into_unified`.
    ///
    /// # Safety:
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = UnifiedBox::new(5).unwrap();
    /// let ptr = UnifiedBox::into_unified(x);
    /// let x = unsafe { UnifiedBox::from_unified(ptr) };
    /// ```
    pub unsafe fn from_unified(ptr: UnifiedPointer<T>) -> Self {
        UnifiedBox { ptr }
    }

    /// Consumes the UnifiedBox, returning the wrapped UnifiedPointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the UnifiedBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new UnifiedBox using the `UnifiedBox::from_unified` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `UnifiedBox::into_unified(b)` instead of `b.into_unified()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = UnifiedBox::new(5).unwrap();
    /// let ptr = UnifiedBox::into_unified(x);
    /// # unsafe { UnifiedBox::from_unified(ptr) };
    /// ```
    #[allow(clippy::wrong_self_convention)]
    pub fn into_unified(mut b: UnifiedBox<T>) -> UnifiedPointer<T> {
        let ptr = mem::replace(&mut b.ptr, UnifiedPointer::null());
        mem::forget(b);
        ptr
    }

    /// Returns the contained unified pointer without consuming the box.
    ///
    /// This is useful for passing the box to a kernel launch.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let mut x = UnifiedBox::new(5).unwrap();
    /// let ptr = x.as_unified_ptr();
    /// println!("{:p}", ptr);
    /// ```
    pub fn as_unified_ptr(&mut self) -> UnifiedPointer<T> {
        self.ptr
    }

    /// Consumes and leaks the UnifiedBox, returning a mutable reference, &'a mut T. Note that the type T
    /// must outlive the chosen lifetime 'a. If the type has only static references, or none at all,
    /// this may be chosen to be 'static.
    ///
    /// This is mainly useful for data that lives for the remainder of the program's life. Dropping
    /// the returned reference will cause a memory leak. If this is not acceptable, the reference
    /// should be wrapped with the UnifiedBox::from_raw function to produce a new UnifiedBox. This UnifiedBox can then
    /// be dropped, which will properly destroy T and release the allocated memory.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `UnifiedBox::leak(b)` instead of `b.leak()` This is so that there is no conflict with
    /// a method on the inner type.
    pub fn leak<'a>(b: UnifiedBox<T>) -> &'a mut T
    where
        T: 'a,
    {
        unsafe { &mut *UnifiedBox::into_unified(b).as_raw_mut() }
    }

    /// Destroy a `UnifiedBox`, returning an error.
    ///
    /// Deallocating unified memory can return errors from previous asynchronous work. This function
    /// destroys the given box and returns the error and the un-destroyed box on failure.
    ///
    /// # Example:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = UnifiedBox::new(5).unwrap();
    /// match UnifiedBox::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, uni_box)) => {
    ///         println!("Failed to destroy box: {:?}", e);
    ///         // Do something with uni_box
    ///     },
    /// }
    /// ```
    pub fn drop(mut uni_box: UnifiedBox<T>) -> DropResult<UnifiedBox<T>> {
        if uni_box.ptr.is_null() {
            return Ok(());
        }

        let ptr = mem::replace(&mut uni_box.ptr, UnifiedPointer::null());
        unsafe {
            match cuda_free_unified(ptr) {
                Ok(()) => {
                    mem::forget(uni_box);
                    Ok(())
                }
                Err(e) => Err((e, UnifiedBox { ptr })),
            }
        }
    }
}
impl<T: DeviceCopy> Drop for UnifiedBox<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let ptr = mem::replace(&mut self.ptr, UnifiedPointer::null());
            // No choice but to panic if this fails.
            unsafe {
                cuda_free_unified(ptr).expect("Failed to deallocate CUDA Unified memory.");
            }
        }
    }
}

impl<T: DeviceCopy> Borrow<T> for UnifiedBox<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> BorrowMut<T> for UnifiedBox<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl<T: DeviceCopy> AsRef<T> for UnifiedBox<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> AsMut<T> for UnifiedBox<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl<T: DeviceCopy> Deref for UnifiedBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.ptr.as_raw() }
    }
}
impl<T: DeviceCopy> DerefMut for UnifiedBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.as_raw_mut() }
    }
}
impl<T: Display + DeviceCopy> Display for UnifiedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}
impl<T: DeviceCopy> Pointer for UnifiedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<T: DeviceCopy + PartialEq> PartialEq for UnifiedBox<T> {
    fn eq(&self, other: &UnifiedBox<T>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}
impl<T: DeviceCopy + Eq> Eq for UnifiedBox<T> {}
impl<T: DeviceCopy + PartialOrd> PartialOrd for UnifiedBox<T> {
    fn partial_cmp(&self, other: &UnifiedBox<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    fn lt(&self, other: &UnifiedBox<T>) -> bool {
        PartialOrd::lt(&**self, &**other)
    }
    fn le(&self, other: &UnifiedBox<T>) -> bool {
        PartialOrd::le(&**self, &**other)
    }
    fn ge(&self, other: &UnifiedBox<T>) -> bool {
        PartialOrd::ge(&**self, &**other)
    }
    fn gt(&self, other: &UnifiedBox<T>) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}
impl<T: DeviceCopy + Ord> Ord for UnifiedBox<T> {
    fn cmp(&self, other: &UnifiedBox<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
impl<T: DeviceCopy + Hash> Hash for UnifiedBox<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

#[cfg(test)]
mod test_unified_box {
    use super::*;

    #[derive(Clone, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_allocate_and_free() {
        let _context = crate::quick_init().unwrap();
        let mut x = UnifiedBox::new(5u64).unwrap();
        *x = 10;
        assert_eq!(10, *x);
        drop(x);
    }

    #[test]
    fn test_allocates_for_non_zst() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(5u64).unwrap();
        let ptr = UnifiedBox::into_unified(x);
        assert!(!ptr.is_null());
        let _ = unsafe { UnifiedBox::from_unified(ptr) };
    }

    #[test]
    fn test_doesnt_allocate_for_zero_sized_type() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(ZeroSizedType).unwrap();
        let ptr = UnifiedBox::into_unified(x);
        assert!(ptr.is_null());
        let _ = unsafe { UnifiedBox::from_unified(ptr) };
    }

    #[test]
    fn test_into_from_unified() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(5u64).unwrap();
        let ptr = UnifiedBox::into_unified(x);
        let _ = unsafe { UnifiedBox::from_unified(ptr) };
    }

    #[test]
    fn test_equality() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(5u64).unwrap();
        let y = UnifiedBox::new(5u64).unwrap();
        let z = UnifiedBox::new(0u64).unwrap();
        assert_eq!(x, y);
        assert!(x != z);
    }

    #[test]
    fn test_ordering() {
        let _context = crate::quick_init().unwrap();
        let x = UnifiedBox::new(1u64).unwrap();
        let y = UnifiedBox::new(2u64).unwrap();

        assert!(x < y);
    }
}
