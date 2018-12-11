use crate::error::CudaError;
use cuda_sys::cuda::CUstream;
use futures::{Async, Future, Poll};
use std::cell::Cell;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Promise<'a, F>
where
    F: FnOnce(&Executor<'a>),
{
    inner: CUstream,
    f: F,
    phantom: PhantomData<Cell<&'a ()>>,
}

impl<'a, F> Promise<'a, F>
where
    F: FnOnce(&Executor<'a>),
{
    pub(crate) fn new(stream: CUstream, f: F) -> Promise<'a, F> {
        Promise {
            inner: stream,
            f: f,
            phantom: PhantomData,
        }
    }

    fn execute(self, executor: &Executor<'a>) {
        (self.f)(executor);
    }
}

impl<'a, F> Future for Promise<'a, F>
where
    F: FnOnce(&Executor),
{
    type Item = Async<()>;
    type Error = CudaError;
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        Ok(Async::NotReady)
    }
}

#[derive(Debug)]
pub struct Executor<'a> {
    inner: CUstream,
    phantom: PhantomData<Cell<&'a ()>>,
}

impl<'a> Executor<'a> {
    pub(crate) fn from_stream(stream: CUstream) -> Executor<'a> {
        Executor {
            inner: stream,
            phantom: PhantomData,
        }
    }

    fn copy(&self, srcs: &'a [i32], dsts: &'a mut [i32]) {
        for (src, dst) in srcs.iter().zip(dsts) {
            *dst = *src;
        }
    }
}
