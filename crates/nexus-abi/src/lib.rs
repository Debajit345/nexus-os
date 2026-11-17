#![cfg_attr(not(any(test, feature = "std")), no_std)]

//! Nexus OS ABI definitions shared between the microkernel and user-space
//! personality servers (Native, POSIX, Windows NT, and Darwin/XNU).

/// Capability handle encoding:
/// - bits 0..32: slot index in the capability table
/// - bits 32..56: generation counter (prevents use-after-free / ABA)
/// - bits 56..64: reserved / flags
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CapHandle(u64);

impl CapHandle {
    pub const INVALID: Self = Self(0);

    #[inline]
    pub const fn new(slot: u32, generation: u32) -> Self {
        let gen_masked = (generation as u64) & 0x00FF_FFFF;
        Self((slot as u64) | (gen_masked << 32))
    }

    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn as_raw(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub const fn slot(self) -> u32 {
        (self.0 & 0xFFFF_FFFF) as u32
    }

    #[inline]
    pub const fn generation(self) -> u32 {
        ((self.0 >> 32) & 0x00FF_FFFF) as u32
    }
}

/// Capability access rights bitmask.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rights(u32);

impl Rights {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const EXECUTE: Self = Self(1 << 2);
    pub const DUPLICATE: Self = Self(1 << 3);
    pub const TRANSFER: Self = Self(1 << 4);
    pub const WAIT: Self = Self(1 << 5);
    pub const SIGNAL: Self = Self(1 << 6);
    pub const MAP: Self = Self(1 << 7);
    pub const ADMIN: Self = Self(1 << 31);
    pub const ALL: Self = Self(
        Self::READ.0
            | Self::WRITE.0
            | Self::EXECUTE.0
            | Self::DUPLICATE.0
            | Self::TRANSFER.0
            | Self::WAIT.0
            | Self::SIGNAL.0
            | Self::MAP.0
            | Self::ADMIN.0,
    );

    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline]
    pub const fn is_subset_of(self, superset: Self) -> bool {
        (self.0 & superset.0) == self.0
    }
}

/// Kernel object types managed by the capability table.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ObjectType {
    Null = 0,
    Process = 1,
    Thread = 2,
    Vmo = 3,
    Vmar = 4,
    Channel = 5,
    Event = 6,
    Timer = 7,
    File = 8,
    Socket = 9,
    Device = 10,
    PersonalityContext = 11,
}

impl ObjectType {
    pub const fn from_u16(val: u16) -> Option<Self> {
        match val {
            0 => Some(Self::Null),
            1 => Some(Self::Process),
            2 => Some(Self::Thread),
            3 => Some(Self::Vmo),
            4 => Some(Self::Vmar),
            5 => Some(Self::Channel),
            6 => Some(Self::Event),
            7 => Some(Self::Timer),
            8 => Some(Self::File),
            9 => Some(Self::Socket),
            10 => Some(Self::Device),
            11 => Some(Self::PersonalityContext),
            _ => None,
        }
    }
}

/// Execution personality for a process or thread.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Personality {
    Native = 0,
    Posix = 1,
    Nt = 2,
    Darwin = 3,
}

impl Personality {
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Native),
            1 => Some(Self::Posix),
            2 => Some(Self::Nt),
            3 => Some(Self::Darwin),
            _ => None,
        }
    }
}

/// Unified syscall numbers across Native, POSIX, NT, and Darwin personalities.
#[repr(u64)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SyscallNumber {
    // Native capability & microkernel core (0x0000..0x0FFF)
    CapClose = 0x0001,
    CapDuplicate = 0x0002,
    CapTransfer = 0x0003,
    ChannelCreate = 0x0010,
    ChannelSend = 0x0011,
    ChannelRecv = 0x0012,
    VmoCreate = 0x0020,
    VmoMap = 0x0021,
    VmoUnmap = 0x0022,
    ThreadSpawn = 0x0030,
    ThreadExit = 0x0031,
    ThreadYield = 0x0032,
    ProcessSpawn = 0x0040,
    ProcessExit = 0x0041,
    WaitOne = 0x0050,
    Signal = 0x0051,
    DebugLog = 0x00FF,

    // POSIX personality bridge (0x1000..0x1FFF)
    PosixRead = 0x1000,
    PosixWrite = 0x1001,
    PosixOpenAt = 0x1002,
    PosixClose = 0x1003,
    PosixMmap = 0x1009,
    PosixFork = 0x1039,
    PosixExecve = 0x103B,

    // Windows NT personality bridge (0x2000..0x2FFF)
    NtCreateFile = 0x2001,
    NtReadFile = 0x2002,
    NtWriteFile = 0x2003,
    NtClose = 0x2004,
    NtAllocateVirtualMemory = 0x2018,
    NtCreateProcessEx = 0x204D,

    // Darwin / XNU personality bridge (0x3000..0x3FFF)
    DarwinMachMsg = 0x3001,
    DarwinBsdRead = 0x3003,
    DarwinBsdWrite = 0x3004,
    DarwinBsdOpen = 0x3005,
    DarwinBsdClose = 0x3006,
    DarwinTaskSelf = 0x301C,
}

impl SyscallNumber {
    pub const fn from_u64(val: u64) -> Option<Self> {
        match val {
            0x0001 => Some(Self::CapClose),
            0x0002 => Some(Self::CapDuplicate),
            0x0003 => Some(Self::CapTransfer),
            0x0010 => Some(Self::ChannelCreate),
            0x0011 => Some(Self::ChannelSend),
            0x0012 => Some(Self::ChannelRecv),
            0x0020 => Some(Self::VmoCreate),
            0x0021 => Some(Self::VmoMap),
            0x0022 => Some(Self::VmoUnmap),
            0x0030 => Some(Self::ThreadSpawn),
            0x0031 => Some(Self::ThreadExit),
            0x0032 => Some(Self::ThreadYield),
            0x0040 => Some(Self::ProcessSpawn),
            0x0041 => Some(Self::ProcessExit),
            0x0050 => Some(Self::WaitOne),
            0x0051 => Some(Self::Signal),
            0x00FF => Some(Self::DebugLog),

            0x1000 => Some(Self::PosixRead),
            0x1001 => Some(Self::PosixWrite),
            0x1002 => Some(Self::PosixOpenAt),
            0x1003 => Some(Self::PosixClose),
            0x1009 => Some(Self::PosixMmap),
            0x1039 => Some(Self::PosixFork),
            0x103B => Some(Self::PosixExecve),

            0x2001 => Some(Self::NtCreateFile),
            0x2002 => Some(Self::NtReadFile),
            0x2003 => Some(Self::NtWriteFile),
            0x2004 => Some(Self::NtClose),
            0x2018 => Some(Self::NtAllocateVirtualMemory),
            0x204D => Some(Self::NtCreateProcessEx),

            0x3001 => Some(Self::DarwinMachMsg),
            0x3003 => Some(Self::DarwinBsdRead),
            0x3004 => Some(Self::DarwinBsdWrite),
            0x3005 => Some(Self::DarwinBsdOpen),
            0x3006 => Some(Self::DarwinBsdClose),
            0x301C => Some(Self::DarwinTaskSelf),
            _ => None,
        }
    }

    pub const fn personality(self) -> Personality {
        let raw = self as u64;
        if raw < 0x1000 {
            Personality::Native
        } else if raw < 0x2000 {
            Personality::Posix
        } else if raw < 0x3000 {
            Personality::Nt
        } else {
            Personality::Darwin
        }
    }
}

/// Standard ABI status/error codes returned by syscalls.
#[repr(i64)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbiError {
    Ok = 0,
    InvalidHandle = -1,
    AccessDenied = -2,
    InvalidArg = -3,
    NotFound = -4,
    NoMemory = -5,
    WouldBlock = -6,
    NotSupported = -7,
    AlreadyExists = -8,
    BadState = -9,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_handle_roundtrip() {
        let h = CapHandle::new(42, 7);
        assert!(h.is_valid());
        assert_eq!(h.slot(), 42);
        assert_eq!(h.generation(), 7);
        assert_eq!(CapHandle::from_raw(h.as_raw()), h);
        assert!(!CapHandle::INVALID.is_valid());
    }

    #[test]
    fn rights_attenuation() {
        let rw = Rights::READ.union(Rights::WRITE);
        assert!(rw.contains(Rights::READ));
        assert!(rw.contains(Rights::WRITE));
        assert!(!rw.contains(Rights::EXECUTE));
        assert!(Rights::READ.is_subset_of(rw));
        assert!(!Rights::ALL.is_subset_of(rw));
        assert_eq!(rw.intersect(Rights::READ), Rights::READ);
    }

    #[test]
    fn object_type_and_personality_roundtrip() {
        assert_eq!(ObjectType::from_u16(5), Some(ObjectType::Channel));
        assert_eq!(ObjectType::from_u16(999), None);
        assert_eq!(Personality::from_u8(1), Some(Personality::Posix));
        assert_eq!(Personality::from_u8(2), Some(Personality::Nt));
        assert_eq!(Personality::from_u8(3), Some(Personality::Darwin));
        assert_eq!(Personality::from_u8(99), None);
    }

    #[test]
    fn syscall_personality_mapping() {
        assert_eq!(
            SyscallNumber::from_u64(0x0010),
            Some(SyscallNumber::ChannelCreate)
        );
        assert_eq!(
            SyscallNumber::ChannelCreate.personality(),
            Personality::Native
        );
        assert_eq!(SyscallNumber::PosixOpenAt.personality(), Personality::Posix);
        assert_eq!(SyscallNumber::NtCreateFile.personality(), Personality::Nt);
        assert_eq!(
            SyscallNumber::DarwinMachMsg.personality(),
            Personality::Darwin
        );
        assert_eq!(SyscallNumber::from_u64(0xDEAD_BEEF), None);
    }
}
