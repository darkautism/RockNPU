#![forbid(unsafe_op_in_unsafe_fn)]

use rocket_uapi as uapi;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::ptr::NonNull;
use std::slice;

pub struct RocketDevice {
    file: File,
}

impl RocketDevice {
    pub fn open() -> io::Result<Self> {
        Self::open_path("/dev/accel/accel0")
    }
    pub fn open_path(path: &str) -> io::Result<Self> {
        Ok(Self {
            file: OpenOptions::new().read(true).write(true).open(path)?,
        })
    }
    pub fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
    pub fn alloc_buffer(&self, size: usize) -> io::Result<RocketBuffer<'_>> {
        let allocation = alloc_mapped_bo(self.fd(), size)?;
        Ok(RocketBuffer {
            device: self,
            ptr: allocation.ptr,
            requested_size: size,
            handle: allocation.handle,
            dma_address: allocation.dma_address,
            mmap_offset: allocation.mmap_offset,
        })
    }

    /// Allocate a BO that owns a cloned fd for the same DRM file context.
    ///
    /// `File::try_clone` uses `dup`, so GEM handles created on this device stay
    /// valid through the owned buffer even when no Rust borrow of `RocketDevice`
    /// is retained. This is intended for long-lived resident/prepacked weights.
    pub fn alloc_owned_buffer(&self, size: usize) -> io::Result<RocketOwnedBuffer> {
        let file = self.file.try_clone()?;
        let allocation = alloc_mapped_bo(self.fd(), size)?;
        Ok(RocketOwnedBuffer {
            file,
            ptr: allocation.ptr,
            requested_size: size,
            handle: allocation.handle,
            dma_address: allocation.dma_address,
            mmap_offset: allocation.mmap_offset,
        })
    }
    pub fn submit(&self, tasks: &[uapi::Task], inputs: &[u32], outputs: &[u32]) -> io::Result<()> {
        if tasks.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Rocket job requires at least one task",
            ));
        }
        let job =
            uapi::Job {
                tasks: tasks.as_ptr() as usize as u64,
                in_bo_handles: inputs.as_ptr() as usize as u64,
                out_bo_handles: outputs.as_ptr() as usize as u64,
                task_count: tasks
                    .len()
                    .try_into()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many tasks"))?,
                task_struct_size: core::mem::size_of::<uapi::Task>() as u32,
                in_bo_handle_count: inputs.len().try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "too many input BOs")
                })?,
                out_bo_handle_count: outputs.len().try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "too many output BOs")
                })?,
            };
        let submit = uapi::Submit {
            jobs: (&job as *const uapi::Job) as usize as u64,
            job_count: 1,
            job_struct_size: core::mem::size_of::<uapi::Job>() as u32,
            reserved: 0,
        };
        uapi::submit(self.fd(), &submit)
    }
}

struct MappedBo {
    ptr: NonNull<u8>,
    handle: u32,
    dma_address: u64,
    mmap_offset: u64,
}

fn alloc_mapped_bo(fd: RawFd, size: usize) -> io::Result<MappedBo> {
    let size32 = u32::try_from(size).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Rocket BO size exceeds u32 UAPI",
        )
    })?;
    let mut arg = uapi::CreateBo {
        size: size32,
        ..Default::default()
    };
    uapi::create_bo(fd, &mut arg)?;
    let map_len = size.max(1);
    // SAFETY: offset is returned by CREATE_BO for mmap on this DRM fd; mapping
    // is released by the owning RocketBuffer/RocketOwnedBuffer Drop.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            map_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            arg.offset as libc::off_t,
        )
    };
    if ptr == libc::MAP_FAILED {
        let err = io::Error::last_os_error();
        let _ = uapi::gem_close(
            fd,
            &uapi::GemClose {
                handle: arg.handle,
                pad: 0,
            },
        );
        return Err(err);
    }
    Ok(MappedBo {
        ptr: NonNull::new(ptr.cast()).expect("mmap returned null"),
        handle: arg.handle,
        dma_address: arg.dma_address,
        mmap_offset: arg.offset,
    })
}

pub struct RocketBuffer<'a> {
    device: &'a RocketDevice,
    ptr: NonNull<u8>,
    requested_size: usize,
    handle: u32,
    dma_address: u64,
    mmap_offset: u64,
}
impl RocketBuffer<'_> {
    pub fn handle(&self) -> u32 {
        self.handle
    }
    pub fn dma_address(&self) -> u64 {
        self.dma_address
    }
    pub fn mmap_offset(&self) -> u64 {
        self.mmap_offset
    }
    pub fn len(&self) -> usize {
        self.requested_size
    }
    pub fn is_empty(&self) -> bool {
        self.requested_size == 0
    }
    pub fn prep(&self, absolute_timeout_ns: i64) -> io::Result<()> {
        uapi::prep_bo(
            self.device.fd(),
            &uapi::PrepBo {
                handle: self.handle,
                reserved: 0,
                timeout_ns: absolute_timeout_ns,
            },
        )
    }
    pub fn fini(&self) -> io::Result<()> {
        uapi::fini_bo(
            self.device.fd(),
            &uapi::FiniBo {
                handle: self.handle,
                reserved: 0,
            },
        )
    }
    pub fn prep_relative(&self, timeout_ns: i64) -> io::Result<()> {
        let deadline = if timeout_ns <= 0 {
            timeout_ns
        } else {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: ts points to valid writable timespec.
            if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } != 0 {
                return Err(io::Error::last_os_error());
            }
            let now = (ts.tv_sec as i128) * 1_000_000_000i128 + ts.tv_nsec as i128;
            i64::try_from(now + timeout_ns as i128)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "timeout overflow"))?
        };
        self.prep(deadline)
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: mapping remains valid for self lifetime and requested_size <= mapping length.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.requested_size) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: &mut self guarantees unique Rust access to mapped bytes for this call.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.requested_size) }
    }
}
impl Drop for RocketBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: ptr/map length were returned by mmap in alloc_buffer and are unmapped exactly once here.
        let _ = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.requested_size.max(1)) };
        let _ = uapi::gem_close(
            self.device.fd(),
            &uapi::GemClose {
                handle: self.handle,
                pad: 0,
            },
        );
    }
}

pub struct RocketOwnedBuffer {
    file: File,
    ptr: NonNull<u8>,
    requested_size: usize,
    handle: u32,
    dma_address: u64,
    mmap_offset: u64,
}

impl RocketOwnedBuffer {
    pub fn handle(&self) -> u32 {
        self.handle
    }
    pub fn dma_address(&self) -> u64 {
        self.dma_address
    }
    pub fn mmap_offset(&self) -> u64 {
        self.mmap_offset
    }
    pub fn len(&self) -> usize {
        self.requested_size
    }
    pub fn is_empty(&self) -> bool {
        self.requested_size == 0
    }
    pub fn prep(&self, absolute_timeout_ns: i64) -> io::Result<()> {
        uapi::prep_bo(
            self.file.as_raw_fd(),
            &uapi::PrepBo {
                handle: self.handle,
                reserved: 0,
                timeout_ns: absolute_timeout_ns,
            },
        )
    }
    pub fn prep_relative(&self, timeout_ns: i64) -> io::Result<()> {
        let deadline = if timeout_ns <= 0 {
            timeout_ns
        } else {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: ts points to a valid writable timespec.
            if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } != 0 {
                return Err(io::Error::last_os_error());
            }
            let now = (ts.tv_sec as i128) * 1_000_000_000i128 + ts.tv_nsec as i128;
            i64::try_from(now + timeout_ns as i128)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "timeout overflow"))?
        };
        self.prep(deadline)
    }
    pub fn fini(&self) -> io::Result<()> {
        uapi::fini_bo(
            self.file.as_raw_fd(),
            &uapi::FiniBo {
                handle: self.handle,
                reserved: 0,
            },
        )
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: mapping remains valid for self lifetime and requested_size <= mapping length.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.requested_size) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: &mut self guarantees unique Rust access to mapped bytes for this call.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.requested_size) }
    }
}

impl Drop for RocketOwnedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr/map length were returned by mmap in alloc_owned_buffer and are unmapped exactly once here.
        let _ = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.requested_size.max(1)) };
        let _ = uapi::gem_close(
            self.file.as_raw_fd(),
            &uapi::GemClose {
                handle: self.handle,
                pad: 0,
            },
        );
    }
}

pub use uapi::Task;
