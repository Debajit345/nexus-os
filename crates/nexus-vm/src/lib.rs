#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use nexus_abi::Personality;

#[derive(Clone, Debug)]
pub struct VmConfig {
    pub name: String,
    pub vcpus: u8,
    pub ram_mib: u32,
    pub personality: Personality,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VcpuState {
    Created,
    Running,
    Paused,
    Halted,
}

#[derive(Clone, Debug)]
pub struct GuestMemoryRegion {
    pub guest_phys: u64,
    pub size: usize,
}

#[derive(Clone, Debug)]
pub struct ContainerSpec {
    pub name: String,
    pub rootfs: String,
    pub personality: Personality,
}

struct VmInstance {
    config: VmConfig,
    state: VcpuState,
}

#[derive(Default)]
pub struct IsolationManager {
    next_id: u32,
    vms: BTreeMap<u32, VmInstance>,
    containers: Vec<ContainerSpec>,
}

impl IsolationManager {
    pub const fn new() -> Self {
        Self {
            next_id: 1,
            vms: BTreeMap::new(),
            containers: Vec::new(),
        }
    }

    pub fn create_vm(&mut self, config: VmConfig) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.vms.insert(
            id,
            VmInstance {
                config,
                state: VcpuState::Created,
            },
        );
        id
    }

    pub fn start_vm(&mut self, id: u32) -> Result<(), &'static str> {
        let vm = self.vms.get_mut(&id).ok_or("unknown vm")?;
        vm.state = VcpuState::Running;
        Ok(())
    }

    pub fn pause_vm(&mut self, id: u32) -> Result<(), &'static str> {
        let vm = self.vms.get_mut(&id).ok_or("unknown vm")?;
        vm.state = VcpuState::Paused;
        Ok(())
    }

    pub fn vm_state(&self, id: u32) -> Option<VcpuState> {
        self.vms.get(&id).map(|v| v.state)
    }

    pub fn create_container(&mut self, spec: ContainerSpec) {
        self.containers.push(spec);
    }

    pub fn list_vms(&self) -> Vec<(u32, &str, VcpuState)> {
        self.vms
            .iter()
            .map(|(&id, v)| (id, v.config.name.as_str(), v.state))
            .collect()
    }

    pub fn container_count(&self) -> usize {
        self.containers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vm_and_container_lifecycle() {
        let mut mgr = IsolationManager::new();
        let id = mgr.create_vm(VmConfig {
            name: String::from("win11-compat"),
            vcpus: 4,
            ram_mib: 4096,
            personality: Personality::Nt,
        });
        assert_eq!(mgr.vm_state(id), Some(VcpuState::Created));
        mgr.start_vm(id).unwrap();
        assert_eq!(mgr.vm_state(id), Some(VcpuState::Running));
        mgr.pause_vm(id).unwrap();
        assert_eq!(mgr.vm_state(id), Some(VcpuState::Paused));
        assert_eq!(mgr.list_vms().len(), 1);

        mgr.create_container(ContainerSpec {
            name: String::from("linux-dev"),
            rootfs: String::from("/var/containers/linux"),
            personality: Personality::Posix,
        });
        assert_eq!(mgr.container_count(), 1);
    }
}
