#[cfg(test)]
mod test_async_futures {
    use super::*;
    use crate::stream::{Stream, StreamFlags};

    #[test]
    fn test_host_to_device() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    }

    #[test]
    fn test_device_to_host() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    }

    #[test]
    fn test_device_to_device() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    }

    #[test]
    fn test_roundtrip() {
        let _context = crate::quick_init().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        // To device
        // Kernel
        // Back to host
    }

    #[test]
    fn test_multistream_roundtrip() {
        let _context = crate::quick_init().unwrap();
        let stream1 = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let stream2 = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    }

    #[test]
    fn test_multistream_shared_buffer() {
        let _context = crate::quick_init().unwrap();
        let stream1 = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let stream2 = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    }

    // Doesn't compile checks
}
