#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use nexus_desktop::ThemeMode;
use nexus_shell::{FsError, ShellFs, SystemInfo};

pub struct FinderApp {
    pub cwd: String,
}

impl FinderApp {
    pub fn new(start_dir: &str) -> Self {
        Self {
            cwd: String::from(start_dir),
        }
    }

    pub fn list(&self, fs: &dyn ShellFs) -> Result<Vec<String>, FsError> {
        fs.list_dir(&self.cwd)
    }

    pub fn navigate(&mut self, path: &str, fs: &dyn ShellFs) -> Result<(), FsError> {
        let _ = fs.list_dir(path)?;
        self.cwd = String::from(path);
        Ok(())
    }
}

pub struct ActivityMonitorApp;

impl ActivityMonitorApp {
    pub fn summary(sys: &SystemInfo) -> String {
        format!(
            "{} {} ({}) | CPU: {} ({} cores) | RAM: {}/{} MiB free | PCI: {}",
            sys.os_name,
            sys.version,
            sys.arch,
            sys.cpu_vendor,
            sys.online_cpus,
            sys.free_ram_mib,
            sys.total_ram_mib,
            sys.pci_count
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PowerMode {
    PowerSaver,
    Balanced,
    Performance,
}

pub struct SettingsApp {
    pub theme: ThemeMode,
    pub power_mode: PowerMode,
    pub resolution: (u32, u32),
}

impl Default for SettingsApp {
    fn default() -> Self {
        Self {
            theme: ThemeMode::Dark,
            power_mode: PowerMode::Balanced,
            resolution: (1920, 1080),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_shell::MemShellFs;

    #[test]
    fn finder_monitor_and_settings() {
        let mut fs = MemShellFs::new();
        fs.mkdir("/Applications").unwrap();
        let mut finder = FinderApp::new("/");
        assert!(finder
            .list(&fs)
            .unwrap()
            .contains(&String::from("Applications")));
        finder.navigate("/Applications", &fs).unwrap();
        assert_eq!(finder.cwd, "/Applications");

        let sys = SystemInfo {
            os_name: String::from("Nexus OS"),
            version: String::from("0.2.0"),
            arch: String::from("x86_64"),
            cpu_vendor: String::from("Amd"),
            total_ram_mib: 16384,
            free_ram_mib: 15000,
            online_cpus: 16,
            pci_count: 12,
        };
        let summary = ActivityMonitorApp::summary(&sys);
        assert!(summary.contains("16 cores"));

        let settings = SettingsApp {
            theme: ThemeMode::Light,
            power_mode: PowerMode::Performance,
            ..Default::default()
        };
        assert_eq!(settings.theme, ThemeMode::Light);
    }
}
