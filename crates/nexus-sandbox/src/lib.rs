#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Permission {
    FsRead,
    FsWrite,
    NetConnect,
    GpuAccess,
    AudioAccess,
    IpcSend,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PolicyDecision {
    Allow,
    Deny,
}

#[derive(Clone, Debug)]
pub struct AppPolicy {
    pub app_id: String,
    pub allowed_paths: Vec<String>,
    pub permissions: Vec<Permission>,
}

pub struct SandboxEnforcer;

impl SandboxEnforcer {
    pub fn check_permission(policy: &AppPolicy, perm: Permission) -> PolicyDecision {
        if policy.permissions.contains(&perm) {
            PolicyDecision::Allow
        } else {
            PolicyDecision::Deny
        }
    }

    pub fn check_path(policy: &AppPolicy, path: &str, write: bool) -> PolicyDecision {
        let needed = if write {
            Permission::FsWrite
        } else {
            Permission::FsRead
        };
        if !policy.permissions.contains(&needed) {
            return PolicyDecision::Deny;
        }
        if path.contains("..") {
            return PolicyDecision::Deny;
        }
        for prefix in &policy.allowed_paths {
            if path.starts_with(prefix.as_str()) {
                return PolicyDecision::Allow;
            }
        }
        PolicyDecision::Deny
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn sandbox_path_and_permission_checks() {
        let policy = AppPolicy {
            app_id: String::from("org.nexus.finder"),
            allowed_paths: vec![String::from("/home/user"), String::from("/tmp")],
            permissions: vec![Permission::FsRead, Permission::GpuAccess],
        };
        assert_eq!(
            SandboxEnforcer::check_permission(&policy, Permission::GpuAccess),
            PolicyDecision::Allow
        );
        assert_eq!(
            SandboxEnforcer::check_permission(&policy, Permission::NetConnect),
            PolicyDecision::Deny
        );
        assert_eq!(
            SandboxEnforcer::check_path(&policy, "/home/user/docs", false),
            PolicyDecision::Allow
        );
        assert_eq!(
            SandboxEnforcer::check_path(&policy, "/home/user/../etc/shadow", false),
            PolicyDecision::Deny
        );
        assert_eq!(
            SandboxEnforcer::check_path(&policy, "/home/user/docs", true),
            PolicyDecision::Deny
        );
    }
}
