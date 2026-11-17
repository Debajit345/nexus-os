use alloc::string::String;

/// Hardware and OS metadata exposed to shell builtins (`uname`, `free`, `lscpu`, `lspci`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemInfo {
    pub os_name: String,
    pub version: String,
    pub arch: String,
    pub cpu_vendor: String,
    pub total_ram_mib: u64,
    pub free_ram_mib: u64,
    pub online_cpus: u32,
    pub pci_count: u32,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            os_name: String::from("NexusOS"),
            version: String::from("0.1.0"),
            arch: String::from("x86_64"),
            cpu_vendor: String::from("GenuineIntel"),
            total_ram_mib: 512,
            free_ram_mib: 384,
            online_cpus: 1,
            pci_count: 0,
        }
    }
}
