//! Kernel capability table skeleton.

use nexus_abi::{AbiError, CapHandle, ObjectType, Rights};

pub const MAX_CAPABILITIES: usize = 256;

#[derive(Copy, Clone, Debug)]
pub struct CapEntry {
    pub occupied: bool,
    pub generation: u32,
    pub obj_type: ObjectType,
    pub rights: Rights,
    pub object_ptr: usize,
}

impl CapEntry {
    const fn empty() -> Self {
        Self {
            occupied: false,
            generation: 1,
            obj_type: ObjectType::Null,
            rights: Rights::NONE,
            object_ptr: 0,
        }
    }
}

pub struct CapTable {
    slots: [CapEntry; MAX_CAPABILITIES],
    active_count: usize,
}

impl CapTable {
    pub const fn new() -> Self {
        Self {
            slots: [CapEntry::empty(); MAX_CAPABILITIES],
            active_count: 0,
        }
    }

    pub fn count(&self) -> usize {
        self.active_count
    }

    pub fn insert(
        &mut self,
        obj_type: ObjectType,
        rights: Rights,
        object_ptr: usize,
    ) -> Result<CapHandle, AbiError> {
        for (idx, slot) in self.slots.iter_mut().enumerate() {
            if !slot.occupied {
                slot.occupied = true;
                slot.obj_type = obj_type;
                slot.rights = rights;
                slot.object_ptr = object_ptr;
                self.active_count += 1;
                return Ok(CapHandle::new(idx as u32, slot.generation));
            }
        }
        Err(AbiError::NoMemory)
    }

    pub fn lookup(&self, handle: CapHandle, required: Rights) -> Result<&CapEntry, AbiError> {
        if !handle.is_valid() {
            return Err(AbiError::InvalidHandle);
        }
        let idx = handle.slot() as usize;
        let entry = self.slots.get(idx).ok_or(AbiError::InvalidHandle)?;
        if !entry.occupied || entry.generation != handle.generation() {
            return Err(AbiError::InvalidHandle);
        }
        if !entry.rights.contains(required) {
            return Err(AbiError::AccessDenied);
        }
        Ok(entry)
    }

    pub fn duplicate(
        &mut self,
        handle: CapHandle,
        new_rights: Rights,
    ) -> Result<CapHandle, AbiError> {
        let entry = *self.lookup(handle, Rights::DUPLICATE)?;
        if !new_rights.is_subset_of(entry.rights) {
            return Err(AbiError::AccessDenied);
        }
        self.insert(entry.obj_type, new_rights, entry.object_ptr)
    }

    pub fn close(&mut self, handle: CapHandle) -> Result<(), AbiError> {
        if !handle.is_valid() {
            return Err(AbiError::InvalidHandle);
        }
        let idx = handle.slot() as usize;
        let entry = self.slots.get_mut(idx).ok_or(AbiError::InvalidHandle)?;
        if !entry.occupied || entry.generation != handle.generation() {
            return Err(AbiError::InvalidHandle);
        }
        entry.occupied = false;
        entry.generation = entry.generation.wrapping_add(1).max(1);
        entry.obj_type = ObjectType::Null;
        entry.rights = Rights::NONE;
        entry.object_ptr = 0;
        self.active_count = self.active_count.saturating_sub(1);
        Ok(())
    }
}
