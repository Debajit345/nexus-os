use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use nexus_abi::Personality;
use nexus_compat::{detect_format, CompatError};

use crate::config::NexusConfig;

/// Declarative specification for a managed user-space service.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceSpec {
    pub name: String,
    pub binary_path: String,
    pub personality: Personality,
    pub dependencies: Vec<String>,
    pub restart_on_exit: bool,
    pub sandbox_strict: bool,
}

impl ServiceSpec {
    /// Creates a new service specification with default flags (`Native` personality,
    /// no dependencies, no automatic restart, non-strict sandbox).
    pub fn new(name: impl Into<String>, binary_path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            binary_path: binary_path.into(),
            personality: Personality::Native,
            dependencies: Vec::new(),
            restart_on_exit: false,
            sandbox_strict: false,
        }
    }
}

/// Lifecycle state of a managed service.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ServiceState {
    Stopped,
    Starting,
    Running,
    Failed,
    Restarting,
}

/// Errors returned by [`ServiceManager`] operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServiceError {
    DuplicateService(String),
    UnknownService(String),
    MissingDependency { service: String, dependency: String },
    DependencyCycle(Vec<String>),
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateService(name) => write!(f, "duplicate service: {name}"),
            Self::UnknownService(name) => write!(f, "unknown service: {name}"),
            Self::MissingDependency {
                service,
                dependency,
            } => write!(
                f,
                "service '{service}' depends on unknown service '{dependency}'"
            ),
            Self::DependencyCycle(cycle) => {
                write!(
                    f,
                    "dependency cycle detected involving: {}",
                    cycle.join(", ")
                )
            }
        }
    }
}

#[derive(Clone, Debug)]
struct ServiceEntry {
    spec: ServiceSpec,
    state: ServiceState,
    restart_count: u32,
    last_exit_code: Option<i32>,
}

/// Infers the execution personality of a binary image from its header magic bytes.
pub fn infer_personality_from_binary(bytes: &[u8]) -> Result<Personality, CompatError> {
    detect_format(bytes).map(|fmt| fmt.personality())
}

/// User-space service manager responsible for dependency-ordered startup,
/// lifecycle tracking, and crash-restart handling.
#[derive(Clone, Debug, Default)]
pub struct ServiceManager {
    services: BTreeMap<String, ServiceEntry>,
    registration_order: Vec<String>,
}

impl ServiceManager {
    /// Creates an empty service manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a service manager and registers all services from a parsed [`NexusConfig`].
    pub fn from_config(config: &NexusConfig) -> Result<Self, ServiceError> {
        let mut manager = Self::new();
        for spec in &config.services {
            manager.register(spec.clone())?;
        }
        Ok(manager)
    }

    /// Registers a new service specification in the `Stopped` state.
    pub fn register(&mut self, spec: ServiceSpec) -> Result<(), ServiceError> {
        if self.services.contains_key(&spec.name) {
            return Err(ServiceError::DuplicateService(spec.name));
        }
        let name = spec.name.clone();
        self.services.insert(
            name.clone(),
            ServiceEntry {
                spec,
                state: ServiceState::Stopped,
                restart_count: 0,
                last_exit_code: None,
            },
        );
        self.registration_order.push(name);
        Ok(())
    }

    /// Computes a deterministic topological startup order (dependencies before dependents)
    /// and transitions all registered services to `Running`.
    ///
    /// Returns the ordered list of service names that were started.
    pub fn start_all(&mut self) -> Result<Vec<String>, ServiceError> {
        let order = self.topological_order()?;

        for name in &order {
            if let Some(entry) = self.services.get_mut(name) {
                if entry.state != ServiceState::Running {
                    entry.state = ServiceState::Starting;
                    entry.state = ServiceState::Running;
                }
            }
        }

        Ok(order)
    }

    /// Starts or completes a restart for a single registered service.
    pub fn start(&mut self, name: &str) -> Result<ServiceState, ServiceError> {
        let deps = self
            .services
            .get(name)
            .ok_or_else(|| ServiceError::UnknownService(name.to_string()))?
            .spec
            .dependencies
            .clone();

        for dep in &deps {
            if !self.services.contains_key(dep) {
                return Err(ServiceError::MissingDependency {
                    service: name.to_string(),
                    dependency: dep.clone(),
                });
            }
        }

        let entry = self
            .services
            .get_mut(name)
            .ok_or_else(|| ServiceError::UnknownService(name.to_string()))?;
        entry.state = ServiceState::Starting;
        entry.state = ServiceState::Running;
        Ok(entry.state)
    }

    /// Stops a registered service, transitioning it to `Stopped`.
    pub fn stop(&mut self, name: &str) -> Result<(), ServiceError> {
        let entry = self
            .services
            .get_mut(name)
            .ok_or_else(|| ServiceError::UnknownService(name.to_string()))?;
        entry.state = ServiceState::Stopped;
        Ok(())
    }

    /// Notifies the manager that a service process exited with `exit_code`.
    ///
    /// - If `restart_on_exit` is `true`, transitions to `Restarting` and increments `restart_count`.
    /// - Otherwise, transitions to `Stopped` on `exit_code == 0` or `Failed` on non-zero exit.
    pub fn notify_exit(
        &mut self,
        name: &str,
        exit_code: i32,
    ) -> Result<ServiceState, ServiceError> {
        let entry = self
            .services
            .get_mut(name)
            .ok_or_else(|| ServiceError::UnknownService(name.to_string()))?;

        entry.last_exit_code = Some(exit_code);
        if entry.spec.restart_on_exit {
            entry.restart_count = entry.restart_count.saturating_add(1);
            entry.state = ServiceState::Restarting;
        } else if exit_code == 0 {
            entry.state = ServiceState::Stopped;
        } else {
            entry.state = ServiceState::Failed;
        }

        Ok(entry.state)
    }

    /// Returns the current [`ServiceState`] of `name`, if registered.
    pub fn status(&self, name: &str) -> Option<ServiceState> {
        self.services.get(name).map(|entry| entry.state)
    }

    /// Returns the number of services currently in [`ServiceState::Running`].
    pub fn running_count(&self) -> usize {
        self.services
            .values()
            .filter(|entry| entry.state == ServiceState::Running)
            .count()
    }

    /// Returns the [`ServiceSpec`] for `name`, if registered.
    pub fn spec(&self, name: &str) -> Option<&ServiceSpec> {
        self.services.get(name).map(|entry| &entry.spec)
    }

    /// Returns how many times `name` has entered `Restarting`.
    pub fn restart_count(&self, name: &str) -> Option<u32> {
        self.services.get(name).map(|entry| entry.restart_count)
    }

    /// Returns the last recorded exit code for `name`, if any.
    pub fn last_exit_code(&self, name: &str) -> Option<i32> {
        self.services
            .get(name)
            .and_then(|entry| entry.last_exit_code)
    }

    /// Returns the total number of registered services.
    pub fn len(&self) -> usize {
        self.services.len()
    }

    /// Returns `true` if no services are registered.
    pub fn is_empty(&self) -> bool {
        self.services.is_empty()
    }

    fn topological_order(&self) -> Result<Vec<String>, ServiceError> {
        let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
        let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for name in &self.registration_order {
            in_degree.insert(name.clone(), 0);
            dependents.insert(name.clone(), Vec::new());
        }

        for name in &self.registration_order {
            let entry = &self.services[name];
            let mut seen_deps = BTreeSet::new();
            for dep in &entry.spec.dependencies {
                if !self.services.contains_key(dep) {
                    return Err(ServiceError::MissingDependency {
                        service: name.clone(),
                        dependency: dep.clone(),
                    });
                }
                if seen_deps.insert(dep.as_str()) {
                    if let Some(deg) = in_degree.get_mut(name) {
                        *deg += 1;
                    }
                    if let Some(list) = dependents.get_mut(dep) {
                        list.push(name.clone());
                    }
                }
            }
        }

        let mut queue = VecDeque::new();
        for name in &self.registration_order {
            if in_degree.get(name).copied().unwrap_or(0) == 0 {
                queue.push_back(name.clone());
            }
        }

        let mut order = Vec::with_capacity(self.services.len());
        while let Some(current) = queue.pop_front() {
            order.push(current.clone());
            if let Some(children) = dependents.get(&current) {
                for child in children {
                    if let Some(deg) = in_degree.get_mut(child) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(child.clone());
                        }
                    }
                }
            }
        }

        if order.len() != self.services.len() {
            let cycle_nodes: Vec<String> = self
                .registration_order
                .iter()
                .filter(|name| in_degree.get(*name).copied().unwrap_or(0) > 0)
                .cloned()
                .collect();
            return Err(ServiceError::DependencyCycle(cycle_nodes));
        }

        Ok(order)
    }
}
