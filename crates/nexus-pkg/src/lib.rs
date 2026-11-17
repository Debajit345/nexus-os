#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Multi-OS package manifest, dependency resolver, and registry for Nexus OS.
//!
//! Supports native Nexus OS packages as well as foreign-personality packages
//! targeting Linux ELF, Windows PE, and macOS `.app` bundles.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use nexus_abi::Personality;

/// Supported package binary/bundle formats across Nexus OS personalities.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PackageFormat {
    /// Native Nexus OS capability-based package.
    NexusNative,
    /// Linux ELF64 executable/package (POSIX personality).
    LinuxElf,
    /// Windows PE32+ executable/package (NT personality).
    WindowsPe,
    /// macOS `.app` Mach-O bundle (Darwin/XNU personality).
    MacOsApp,
}

impl PackageFormat {
    /// Returns the default Nexus OS execution personality for this package format.
    #[inline]
    pub const fn personality(self) -> Personality {
        match self {
            Self::NexusNative => Personality::Native,
            Self::LinuxElf => Personality::Posix,
            Self::WindowsPe => Personality::Nt,
            Self::MacOsApp => Personality::Darwin,
        }
    }

    /// Returns a human-readable identifier for this package format.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NexusNative => "nexus-native",
            Self::LinuxElf => "linux-elf",
            Self::WindowsPe => "windows-pe",
            Self::MacOsApp => "macos-app",
        }
    }
}

/// Metadata describing an installable package in Nexus OS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageManifest {
    pub name: String,
    pub version: String,
    pub format: PackageFormat,
    pub personality: Personality,
    pub entry_bin: String,
    pub dependencies: Vec<String>,
    pub sha256_hex: String,
}

impl PackageManifest {
    /// Creates a new [`PackageManifest`] with the personality inferred from `format`.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        format: PackageFormat,
        entry_bin: impl Into<String>,
        dependencies: Vec<String>,
        sha256_hex: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            format,
            personality: format.personality(),
            entry_bin: entry_bin.into(),
            dependencies,
            sha256_hex: sha256_hex.into(),
        }
    }

    /// Validates basic manifest invariants.
    pub fn validate(&self) -> Result<(), PkgError> {
        if self.name.is_empty() {
            return Err(PkgError::InvalidManifest("package name must not be empty"));
        }
        if self.version.is_empty() {
            return Err(PkgError::InvalidManifest(
                "package version must not be empty",
            ));
        }
        if self.entry_bin.is_empty() {
            return Err(PkgError::InvalidManifest(
                "package entry_bin must not be empty",
            ));
        }
        if !self.sha256_hex.is_empty()
            && (self.sha256_hex.len() != 64
                || !self.sha256_hex.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err(PkgError::InvalidManifest(
                "sha256_hex must be empty or 64 hexadecimal characters",
            ));
        }
        Ok(())
    }
}

/// Errors returned by [`PackageRegistry`] operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PkgError {
    /// Manifest failed validation.
    InvalidManifest(&'static str),
    /// A package with the same name is already registered.
    AlreadyRegistered(String),
    /// Requested package was not found in the registry.
    NotFound(String),
    /// A required dependency is not registered.
    MissingDependency { package: String, dependency: String },
    /// A dependency cycle was detected (contains the cycle path).
    DependencyCycle(Vec<String>),
    /// Attempted to remove a package that is not currently installed.
    NotInstalled(String),
    /// Attempted to remove a package that other installed packages depend on.
    HasDependents {
        package: String,
        dependents: Vec<String>,
    },
}

impl fmt::Display for PkgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            Self::AlreadyRegistered(name) => write!(f, "package already registered: {name}"),
            Self::NotFound(name) => write!(f, "package not found: {name}"),
            Self::MissingDependency {
                package,
                dependency,
            } => write!(
                f,
                "package '{package}' is missing dependency '{dependency}'"
            ),
            Self::DependencyCycle(cycle) => {
                write!(f, "dependency cycle detected: {}", cycle.join(" -> "))
            }
            Self::NotInstalled(name) => write!(f, "package is not installed: {name}"),
            Self::HasDependents {
                package,
                dependents,
            } => write!(
                f,
                "cannot remove '{package}'; required by installed packages: {}",
                dependents.join(", ")
            ),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Visited,
}

/// In-memory package registry with topological dependency resolution and install tracking.
#[derive(Clone, Debug, Default)]
pub struct PackageRegistry {
    manifests: BTreeMap<String, PackageManifest>,
    installed: BTreeSet<String>,
}

impl PackageRegistry {
    /// Creates an empty [`PackageRegistry`].
    pub const fn new() -> Self {
        Self {
            manifests: BTreeMap::new(),
            installed: BTreeSet::new(),
        }
    }

    /// Registers a package manifest.
    pub fn register(&mut self, manifest: PackageManifest) -> Result<(), PkgError> {
        manifest.validate()?;
        if self.manifests.contains_key(&manifest.name) {
            return Err(PkgError::AlreadyRegistered(manifest.name));
        }
        self.manifests.insert(manifest.name.clone(), manifest);
        Ok(())
    }

    /// Returns the manifest for a registered package, if present.
    pub fn get(&self, name: &str) -> Option<&PackageManifest> {
        self.manifests.get(name)
    }

    /// Returns all registered package manifests in deterministic name order.
    pub fn list_registered(&self) -> Vec<&PackageManifest> {
        self.manifests.values().collect()
    }

    /// Resolves the topological installation order for `name` and all its transitive
    /// dependencies, placing dependencies before the packages that require them.
    ///
    /// Detects missing dependencies and dependency cycles.
    pub fn resolve_install_order(&self, name: &str) -> Result<Vec<String>, PkgError> {
        if !self.manifests.contains_key(name) {
            return Err(PkgError::NotFound(String::from(name)));
        }

        let mut state: BTreeMap<String, VisitState> = BTreeMap::new();
        let mut stack: Vec<String> = Vec::new();
        let mut order: Vec<String> = Vec::new();

        self.dfs_resolve(name, &mut state, &mut stack, &mut order)?;
        Ok(order)
    }

    fn dfs_resolve(
        &self,
        current: &str,
        state: &mut BTreeMap<String, VisitState>,
        stack: &mut Vec<String>,
        order: &mut Vec<String>,
    ) -> Result<(), PkgError> {
        if let Some(&visit_state) = state.get(current) {
            match visit_state {
                VisitState::Visited => return Ok(()),
                VisitState::Visiting => {
                    let start = stack.iter().position(|n| n == current).unwrap_or(0);
                    let mut cycle = stack[start..].to_vec();
                    cycle.push(String::from(current));
                    return Err(PkgError::DependencyCycle(cycle));
                }
            }
        }

        let manifest = self
            .manifests
            .get(current)
            .ok_or_else(|| PkgError::NotFound(String::from(current)))?;

        state.insert(String::from(current), VisitState::Visiting);
        stack.push(String::from(current));

        for dep in &manifest.dependencies {
            if !self.manifests.contains_key(dep.as_str()) {
                return Err(PkgError::MissingDependency {
                    package: String::from(current),
                    dependency: dep.clone(),
                });
            }
            self.dfs_resolve(dep, state, stack, order)?;
        }

        stack.pop();
        state.insert(String::from(current), VisitState::Visited);
        order.push(String::from(current));
        Ok(())
    }

    /// Installs `name` and any uninstalled transitive dependencies in topological order.
    ///
    /// Returns the list of newly installed package names in the order they were installed.
    pub fn install(&mut self, name: &str) -> Result<Vec<String>, PkgError> {
        let order = self.resolve_install_order(name)?;
        let mut newly_installed = Vec::new();
        for pkg in order {
            if self.installed.insert(pkg.clone()) {
                newly_installed.push(pkg);
            }
        }
        Ok(newly_installed)
    }

    /// Removes an installed package if no other installed package depends on it.
    pub fn remove(&mut self, name: &str) -> Result<(), PkgError> {
        if !self.installed.contains(name) {
            return Err(PkgError::NotInstalled(String::from(name)));
        }

        let mut dependents = Vec::new();
        for installed_name in &self.installed {
            if installed_name == name {
                continue;
            }
            if let Some(manifest) = self.manifests.get(installed_name) {
                if manifest.dependencies.iter().any(|d| d == name) {
                    dependents.push(installed_name.clone());
                }
            }
        }

        if !dependents.is_empty() {
            return Err(PkgError::HasDependents {
                package: String::from(name),
                dependents,
            });
        }

        self.installed.remove(name);
        Ok(())
    }

    /// Returns all currently installed package manifests in deterministic name order.
    pub fn list_installed(&self) -> Vec<&PackageManifest> {
        self.installed
            .iter()
            .filter_map(|name| self.manifests.get(name))
            .collect()
    }

    /// Returns `true` if `name` is currently installed.
    pub fn is_installed(&self, name: &str) -> bool {
        self.installed.contains(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample_sha(fill: char) -> String {
        core::iter::repeat_n(fill, 64).collect()
    }

    fn make_pkg(name: &str, format: PackageFormat, deps: Vec<&str>) -> PackageManifest {
        PackageManifest::new(
            name,
            "1.0.0",
            format,
            alloc::format!("/bin/{name}"),
            deps.into_iter().map(String::from).collect(),
            sample_sha('a'),
        )
    }

    #[test]
    fn multi_os_formats_and_personalities() {
        let native = make_pkg("nexus-init", PackageFormat::NexusNative, vec![]);
        let linux = make_pkg("bash", PackageFormat::LinuxElf, vec![]);
        let win = make_pkg("notepad", PackageFormat::WindowsPe, vec![]);
        let mac = make_pkg("Finder", PackageFormat::MacOsApp, vec![]);

        assert_eq!(native.personality, Personality::Native);
        assert_eq!(linux.personality, Personality::Posix);
        assert_eq!(win.personality, Personality::Nt);
        assert_eq!(mac.personality, Personality::Darwin);

        assert_eq!(PackageFormat::NexusNative.as_str(), "nexus-native");
        assert_eq!(PackageFormat::LinuxElf.as_str(), "linux-elf");
        assert_eq!(PackageFormat::WindowsPe.as_str(), "windows-pe");
        assert_eq!(PackageFormat::MacOsApp.as_str(), "macos-app");

        let mut reg = PackageRegistry::new();
        reg.register(native).unwrap();
        reg.register(linux).unwrap();
        reg.register(win).unwrap();
        reg.register(mac).unwrap();

        assert_eq!(reg.list_registered().len(), 4);
        assert_eq!(reg.install("Finder").unwrap(), vec!["Finder"]);
        assert_eq!(reg.install("notepad").unwrap(), vec!["notepad"]);
        assert!(reg.is_installed("Finder"));
        assert!(reg.is_installed("notepad"));
        assert!(!reg.is_installed("bash"));
    }

    #[test]
    fn topological_dependency_resolution() {
        let mut reg = PackageRegistry::new();
        reg.register(make_pkg("libc", PackageFormat::LinuxElf, vec![]))
            .unwrap();
        reg.register(make_pkg("libssl", PackageFormat::LinuxElf, vec!["libc"]))
            .unwrap();
        reg.register(make_pkg("zlib", PackageFormat::LinuxElf, vec!["libc"]))
            .unwrap();
        reg.register(make_pkg(
            "curl",
            PackageFormat::LinuxElf,
            vec!["libssl", "zlib"],
        ))
        .unwrap();

        let order = reg.resolve_install_order("curl").unwrap();
        assert_eq!(order, vec!["libc", "libssl", "zlib", "curl"]);

        // Installing curl installs all transitive dependencies in order.
        let installed = reg.install("curl").unwrap();
        assert_eq!(installed, vec!["libc", "libssl", "zlib", "curl"]);

        // Re-installing curl is idempotent and installs nothing new.
        assert!(reg.install("curl").unwrap().is_empty());
    }

    #[test]
    fn cycle_and_missing_dependency_detection() {
        let mut reg = PackageRegistry::new();
        reg.register(make_pkg("a", PackageFormat::NexusNative, vec!["b"]))
            .unwrap();
        reg.register(make_pkg("b", PackageFormat::NexusNative, vec!["c"]))
            .unwrap();
        reg.register(make_pkg("c", PackageFormat::NexusNative, vec!["a"]))
            .unwrap();

        let err = reg.resolve_install_order("a").unwrap_err();
        assert_eq!(
            err,
            PkgError::DependencyCycle(vec![
                String::from("a"),
                String::from("b"),
                String::from("c"),
                String::from("a"),
            ])
        );

        // Self-cycle
        reg.register(make_pkg(
            "self_dep",
            PackageFormat::WindowsPe,
            vec!["self_dep"],
        ))
        .unwrap();
        assert_eq!(
            reg.resolve_install_order("self_dep").unwrap_err(),
            PkgError::DependencyCycle(vec![String::from("self_dep"), String::from("self_dep")])
        );

        // Missing dependency
        reg.register(make_pkg("broken", PackageFormat::MacOsApp, vec!["ghost"]))
            .unwrap();
        assert_eq!(
            reg.resolve_install_order("broken").unwrap_err(),
            PkgError::MissingDependency {
                package: String::from("broken"),
                dependency: String::from("ghost"),
            }
        );

        // Unknown root package
        assert_eq!(
            reg.resolve_install_order("nonexistent").unwrap_err(),
            PkgError::NotFound(String::from("nonexistent"))
        );
    }

    #[test]
    fn install_and_remove_lifecycle() {
        let mut reg = PackageRegistry::new();
        reg.register(make_pkg("core", PackageFormat::NexusNative, vec![]))
            .unwrap();
        reg.register(make_pkg("app", PackageFormat::NexusNative, vec!["core"]))
            .unwrap();

        reg.install("app").unwrap();
        assert!(reg.is_installed("core"));
        assert!(reg.is_installed("app"));
        assert_eq!(reg.list_installed().len(), 2);

        // Cannot remove `core` while `app` is installed.
        assert_eq!(
            reg.remove("core").unwrap_err(),
            PkgError::HasDependents {
                package: String::from("core"),
                dependents: vec![String::from("app")],
            }
        );

        // Remove `app` first, then `core`.
        reg.remove("app").unwrap();
        assert!(!reg.is_installed("app"));
        reg.remove("core").unwrap();
        assert!(!reg.is_installed("core"));
        assert!(reg.list_installed().is_empty());

        // Removing an uninstalled package fails.
        assert_eq!(
            reg.remove("core").unwrap_err(),
            PkgError::NotInstalled(String::from("core"))
        );
    }

    #[test]
    fn manifest_validation_and_duplicate_registration() {
        let mut reg = PackageRegistry::new();
        let bad_name = PackageManifest::new(
            "",
            "1.0.0",
            PackageFormat::NexusNative,
            "/bin/x",
            vec![],
            "",
        );
        assert!(matches!(
            reg.register(bad_name),
            Err(PkgError::InvalidManifest(_))
        ));

        let bad_sha = PackageManifest::new(
            "ok",
            "1.0.0",
            PackageFormat::NexusNative,
            "/bin/ok",
            vec![],
            "not-hex",
        );
        assert!(matches!(
            reg.register(bad_sha),
            Err(PkgError::InvalidManifest(_))
        ));

        let valid = make_pkg("ok", PackageFormat::NexusNative, vec![]);
        reg.register(valid.clone()).unwrap();
        assert_eq!(
            reg.register(valid).unwrap_err(),
            PkgError::AlreadyRegistered(String::from("ok"))
        );
    }
}
