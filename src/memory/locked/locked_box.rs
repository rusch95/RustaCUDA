use crate::error::*;
use crate::memory::malloc::{cuda_free_locked, cuda_malloc_locked};
use crate::memory::DeviceCopy;
use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::convert::{AsMut, AsRef};
use std::fmt::{self, Display, Pointer};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ptr;
use std::ops::{Deref, DerefMut};

/// A pointer type for heap-allocation in locked memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more information on unified
/// memory. Should behave equivalently to `std::boxed::Box`, except that the allocated memory can be
/// seamlessly shared between host and device.
#[derive(Debug)]
pub struct LockedBox<T: DeviceCopy> {
    ptr: *mut T,
}
impl<T: DeviceCopy> LockedBox<T> {
    /// Allocate locked memory and place val into it.
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
    /// let five = LockedBox::new(5).unwrap();
    /// ```
    pub fn new(val: T) -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(LockedBox {
                ptr: ptr::null_mut(),
            })
        } else {
            let mut ubox = unsafe { LockedBox::uninitialized()? };
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
    /// let mut five = unsafe{ LockedBox::uninitialized().unwrap() };
    /// *five = 5u64;
    /// ```
    pub unsafe fn uninitialized() -> CudaResult<Self> {
        if mem::size_of::<T>() == 0 {
            Ok(LockedBox {
                ptr: ptr::null_mut(),
            })
        } else {
            let ptr = cuda_malloc_locked(1)?;
            Ok(LockedBox { ptr })
        }
    }

    /// Constructs a LockedBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// LockedBox. The LockedBox destructor will free the allocated memory, but will not call the destructor
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
    /// let x = LockedBox::new(5).unwrap();
    /// let ptr = LockedBox::into_unified(x).as_raw_mut();
    /// let x = unsafe { LockedBox::from_raw(ptr) };
    /// ```
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        LockedBox {
            ptr,
        }
    }

    /// Consumes the LockedBox, returning the wrapped LockedPointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the LockedBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new LockedBox using the `LockedBox::from_unified` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `LockedBox::into_unified(b)` instead of `b.into_unified()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = LockedBox::new(5).unwrap();
    /// let ptr = LockedBox::into_unified(x);
    /// # unsafe { LockedBox::from_unified(ptr) };
    /// ```
    #[allow(clippy::wrong_self_convention)]
    pub fn into_raw(mut b: LockedBox<T>) -> *mut T {
        let ptr = mem::replace(&mut b.ptr, ptr::null_mut());
        mem::forget(b);
        ptr
    }

    pub fn as_raw(&self) -> *const T {
        self.ptr
    }

    pub fn as_raw_mut(&mut self) -> *mut T {
        self.ptr
    }

    /// Consumes and leaks the LockedBox, returning a mutable reference, &'a mut T. Note that the type T
    /// must outlive the chosen lifetime 'a. If the type has only static references, or none at all,
    /// this may be chosen to be 'static.
    ///
    /// This is mainly useful for data that lives for the remainder of the program's life. Dropping
    /// the returned reference will cause a memory leak. If this is not acceptable, the reference
    /// should be wrapped with the LockedBox::from_raw function to produce a new LockedBox. This LockedBox can then
    /// be dropped, which will properly destroy T and release the allocated memory.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `LockedBox::leak(b)` instead of `b.leak()` This is so that there is no conflict with
    /// a method on the inner type.
    pub fn leak<'a>(b: LockedBox<T>) -> &'a mut T
    where
        T: 'a,
    {
        unsafe { &mut *LockedBox::into_raw(b) }
    }

    /// Destroy a `LockedBox`, returning an error.
    ///
    /// Deallocating unified memory can return errors from previous asynchronous work. This function
    /// destroys the given box and returns the error and the un-destroyed box on failure.
    ///
    /// # Example:
    ///
    /// ```
    /// # let _context = rustacuda::quick_init().unwrap();
    /// use rustacuda::memory::*;
    /// let x = LockedBox::new(5).unwrap();
    /// match LockedBox::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, uni_box)) => {
    ///         println!("Failed to destroy box: {:?}", e);
    ///         // Do something with uni_box
    ///     },
    /// }
    /// ```
    pub fn drop(mut uni_box: LockedBox<T>) -> DropResult<LockedBox<T>> {
        if uni_box.ptr.is_null() {
            return Ok(());
        }

        let ptr = mem::replace(&mut uni_box.ptr, ptr::null_mut());
        unsafe {
            match cuda_free_locked(ptr) {
                Ok(()) => {
                    mem::forget(uni_box);
                    Ok(())
                }
                Err(e) => Err((e, LockedBox { ptr })),
            }
        }
    }
}
impl<T: DeviceCopy> Drop for LockedBox<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let ptr = mem::replace(&mut self.ptr, ptr::null_mut());
            // No choice but to panic if this fails.
            unsafe {
                cuda_free_locked(ptr).expect("Failed to deallocate CUDA Locked memory.");
            }
        }
    }
}

impl<T: DeviceCopy> Borrow<T> for LockedBox<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> BorrowMut<T> for LockedBox<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl<T: DeviceCopy> AsRef<T> for LockedBox<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}
impl<T: DeviceCopy> AsMut<T> for LockedBox<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}
impl<T: DeviceCopy> Deref for LockedBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.ptr }
    }
}
impl<T: DeviceCopy> DerefMut for LockedBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr }
    }
}
impl<T: Display + DeviceCopy> Display for LockedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}
impl<T: DeviceCopy> Pointer for LockedBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}
impl<T: DeviceCopy + PartialEq> PartialEq for LockedBox<T> {
    fn eq(&self, other: &LockedBox<T>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}
impl<T: DeviceCopy + Eq> Eq for LockedBox<T> {}
impl<T: DeviceCopy + PartialOrd> PartialOrd for LockedBox<T> {
    fn partial_cmp(&self, other: &LockedBox<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    fn lt(&self, other: &LockedBox<T>) -> bool {
        PartialOrd::lt(&**self, &**other)
    }
    fn le(&self, other: &LockedBox<T>) -> bool {
        PartialOrd::le(&**self, &**other)
    }
    fn ge(&self, other: &LockedBox<T>) -> bool {
        PartialOrd::ge(&**self, &**other)
    }
    fn gt(&self, other: &LockedBox<T>) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}
impl<T: DeviceCopy + Ord> Ord for LockedBox<T> {
    fn cmp(&self, other: &LockedBox<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
impl<T: DeviceCopy + Hash> Hash for LockedBox<T> {
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
        let mut x = LockedBox::new(5u64).unwrap();
        *x = 10;
        assert_eq!(10, *x);
        drop(x);
    }

    #[test]
    fn test_allocates_for_non_zst() {
        let _context = crate::quick_init().unwrap();
        let x = LockedBox::new(5u64).unwrap();
        let ptr = LockedBox::into_raw(x);
        assert!(!ptr.is_null());
        let _ = unsafe { LockedBox::from_raw(ptr) };
    }

    #[test]
    fn test_doesnt_allocate_for_zero_sized_type() {
        let _context = crate::quick_init().unwrap();
        let x = LockedBox::new(ZeroSizedType).unwrap();
        let ptr = LockedBox::into_raw(x);
        assert!(ptr.is_null());
        let _ = unsafe { LockedBox::from_raw(ptr) };
    }

    #[test]
    fn test_into_from_unified() {
        let _context = crate::quick_init().unwrap();
        let x = LockedBox::new(5u64).unwrap();
        let ptr = LockedBox::into_raw(x);
        let _ = unsafe { LockedBox::from_raw(ptr) };
    }

    #[test]
    fn test_equality() {
        let _context = crate::quick_init().unwrap();
        let x = LockedBox::new(5u64).unwrap();
        let y = LockedBox::new(5u64).unwrap();
        let z = LockedBox::new(0u64).unwrap();
        assert_eq!(x, y);
        assert!(x != z);
    }

    #[test]
    fn test_ordering() {
        let _context = crate::quick_init().unwrap();
        let x = LockedBox::new(1u64).unwrap();
        let y = LockedBox::new(2u64).unwrap();

        assert!(x < y);
    }
}
