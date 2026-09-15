#![forbid(unsafe_op_in_unsafe_fn)]

use std::io;
use std::os::fd::RawFd;

const IOC_NRBITS: u32 = 8;
const IOC_TYPEBITS: u32 = 8;
const IOC_SIZEBITS: u32 = 14;
const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const DRM_IOCTL_BASE: u32 = b'd' as u32;
const DRM_COMMAND_BASE: u32 = 0x40;

const fn ioc(dir: u32, ty: u32, nr: u32, size: usize) -> libc::c_ulong {
    ((dir << IOC_DIRSHIFT)
        | (ty << IOC_TYPESHIFT)
        | (nr << IOC_NRSHIFT)
        | ((size as u32) << IOC_SIZESHIFT)) as libc::c_ulong
}
const fn drm_iow<T>(nr: u32) -> libc::c_ulong {
    ioc(IOC_WRITE, DRM_IOCTL_BASE, nr, core::mem::size_of::<T>())
}
const fn drm_iowr<T>(nr: u32) -> libc::c_ulong {
    ioc(
        IOC_READ | IOC_WRITE,
        DRM_IOCTL_BASE,
        nr,
        core::mem::size_of::<T>(),
    )
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CreateBo {
    pub size: u32,
    pub handle: u32,
    pub dma_address: u64,
    pub offset: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PrepBo {
    pub handle: u32,
    pub reserved: u32,
    pub timeout_ns: i64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FiniBo {
    pub handle: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Task {
    pub regcmd: u32,
    pub regcmd_count: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Job {
    pub tasks: u64,
    pub in_bo_handles: u64,
    pub out_bo_handles: u64,
    pub task_count: u32,
    pub task_struct_size: u32,
    pub in_bo_handle_count: u32,
    pub out_bo_handle_count: u32,
}

/// Rocket interface >= 1.1 appends per-job flags after the stock v1 job.
/// Keep `Job` itself at the Linux v6.18 40-byte ABI so ordinary submits stay
/// byte-for-byte compatible with stock kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct JobFlagged {
    pub job: Job,
    pub flags: u32,
    pub reserved: u32,
}

pub const JOB_BATCHED: u32 = 1 << 0;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Submit {
    pub jobs: u64,
    pub job_count: u32,
    pub job_struct_size: u32,
    pub reserved: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GemClose {
    pub handle: u32,
    pub pad: u32,
}

pub const IOCTL_CREATE_BO: libc::c_ulong = drm_iowr::<CreateBo>(DRM_COMMAND_BASE + 0x00);
pub const IOCTL_SUBMIT: libc::c_ulong = drm_iow::<Submit>(DRM_COMMAND_BASE + 0x01);
pub const IOCTL_PREP_BO: libc::c_ulong = drm_iow::<PrepBo>(DRM_COMMAND_BASE + 0x02);
pub const IOCTL_FINI_BO: libc::c_ulong = drm_iow::<FiniBo>(DRM_COMMAND_BASE + 0x03);
pub const IOCTL_GEM_CLOSE: libc::c_ulong = drm_iow::<GemClose>(0x09);

fn cvt(rc: libc::c_int) -> io::Result<()> {
    if rc == -1 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub fn create_bo(fd: RawFd, arg: &mut CreateBo) -> io::Result<()> {
    // SAFETY: arg is a valid repr(C) buffer matching Linux v6.18 drm_rocket_create_bo.
    cvt(unsafe { libc::ioctl(fd, IOCTL_CREATE_BO, arg as *mut CreateBo) })
}
pub fn prep_bo(fd: RawFd, arg: &PrepBo) -> io::Result<()> {
    // SAFETY: arg remains valid for the duration of ioctl and contains no userspace pointers.
    cvt(unsafe { libc::ioctl(fd, IOCTL_PREP_BO, arg as *const PrepBo) })
}
pub fn fini_bo(fd: RawFd, arg: &FiniBo) -> io::Result<()> {
    // SAFETY: arg remains valid for the duration of ioctl and contains no userspace pointers.
    cvt(unsafe { libc::ioctl(fd, IOCTL_FINI_BO, arg as *const FiniBo) })
}
pub fn gem_close(fd: RawFd, arg: &GemClose) -> io::Result<()> {
    // SAFETY: arg remains valid for the duration of ioctl.
    cvt(unsafe { libc::ioctl(fd, IOCTL_GEM_CLOSE, arg as *const GemClose) })
}
pub fn submit(fd: RawFd, arg: &Submit) -> io::Result<()> {
    // SAFETY: caller must keep all pointer-bearing job/task/handle arrays alive for this synchronous ioctl copy-in.
    cvt(unsafe { libc::ioctl(fd, IOCTL_SUBMIT, arg as *const Submit) })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uapi_layouts_match_linux_v6_18() {
        assert_eq!(core::mem::size_of::<CreateBo>(), 24);
        assert_eq!(core::mem::size_of::<PrepBo>(), 16);
        assert_eq!(core::mem::size_of::<FiniBo>(), 8);
        assert_eq!(core::mem::size_of::<Task>(), 8);
        assert_eq!(core::mem::size_of::<Job>(), 40);
        assert_eq!(core::mem::size_of::<JobFlagged>(), 48);
        assert_eq!(core::mem::offset_of!(JobFlagged, flags), 40);
        assert_eq!(core::mem::size_of::<Submit>(), 24);
        assert_eq!(core::mem::size_of::<GemClose>(), 8);
    }
}
