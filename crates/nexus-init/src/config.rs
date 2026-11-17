use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use nexus_abi::Personality;

use crate::service::ServiceSpec;

/// Parsed representation of `/etc/nexus.conf`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NexusConfig {
    pub hostname: Option<String>,
    pub default_personality: Option<Personality>,
    pub sandbox_default: bool,
    pub system: BTreeMap<String, String>,
    pub services: Vec<ServiceSpec>,
    pub sections: BTreeMap<String, BTreeMap<String, String>>,
}

/// Errors returned by [`parse_config`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    InvalidSection {
        line: usize,
        text: String,
    },
    InvalidKeyValue {
        line: usize,
        text: String,
    },
    InvalidBool {
        line: usize,
        key: String,
        value: String,
    },
    InvalidPersonality {
        line: usize,
        value: String,
    },
    MissingBinaryPath {
        service: String,
    },
    DuplicateService {
        service: String,
    },
    EmptyServiceName {
        line: usize,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSection { line, text } => {
                write!(f, "line {line}: invalid section header '{text}'")
            }
            Self::InvalidKeyValue { line, text } => {
                write!(f, "line {line}: expected 'key = value', got '{text}'")
            }
            Self::InvalidBool { line, key, value } => {
                write!(f, "line {line}: invalid boolean for '{key}': '{value}'")
            }
            Self::InvalidPersonality { line, value } => {
                write!(f, "line {line}: unknown personality '{value}'")
            }
            Self::MissingBinaryPath { service } => {
                write!(f, "service '{service}' is missing required 'binary_path'")
            }
            Self::DuplicateService { service } => {
                write!(f, "duplicate service section '{service}'")
            }
            Self::EmptyServiceName { line } => {
                write!(f, "line {line}: empty service name in section header")
            }
        }
    }
}

#[derive(Clone, Debug)]
enum ActiveSection {
    System,
    Service(usize),
    Other(String),
}

#[derive(Clone, Debug)]
struct RawService {
    name: String,
    binary_path: Option<String>,
    personality: Option<Personality>,
    dependencies: Vec<String>,
    restart_on_exit: Option<bool>,
    sandbox_strict: Option<bool>,
}

impl RawService {
    fn new(name: String) -> Self {
        Self {
            name,
            binary_path: None,
            personality: None,
            dependencies: Vec::new(),
            restart_on_exit: None,
            sandbox_strict: None,
        }
    }
}

/// Parses a declarative `/etc/nexus.conf` configuration string into a [`NexusConfig`].
pub fn parse_config(input: &str) -> Result<NexusConfig, ConfigError> {
    let mut config = NexusConfig::default();
    let mut raw_services: Vec<RawService> = Vec::new();
    let mut seen_services: BTreeSet<String> = BTreeSet::new();
    let mut active = ActiveSection::System;

    for (idx, raw_line) in input.lines().enumerate() {
        let line_num = idx + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with('[') {
            if !line.ends_with(']') {
                return Err(ConfigError::InvalidSection {
                    line: line_num,
                    text: line.to_string(),
                });
            }
            let header = line[1..line.len() - 1].trim();
            if header.is_empty() {
                return Err(ConfigError::InvalidSection {
                    line: line_num,
                    text: line.to_string(),
                });
            }

            if header.eq_ignore_ascii_case("system") || header.eq_ignore_ascii_case("nexus") {
                active = ActiveSection::System;
            } else if let Some(svc_name) = strip_service_prefix(header) {
                let svc_name = svc_name.trim();
                if svc_name.is_empty() {
                    return Err(ConfigError::EmptyServiceName { line: line_num });
                }
                if !seen_services.insert(svc_name.to_string()) {
                    return Err(ConfigError::DuplicateService {
                        service: svc_name.to_string(),
                    });
                }
                let svc_idx = raw_services.len();
                raw_services.push(RawService::new(svc_name.to_string()));
                active = ActiveSection::Service(svc_idx);
            } else {
                config.sections.entry(header.to_string()).or_default();
                active = ActiveSection::Other(header.to_string());
            }
            continue;
        }

        let Some((raw_key, raw_val)) = line.split_once('=') else {
            return Err(ConfigError::InvalidKeyValue {
                line: line_num,
                text: line.to_string(),
            });
        };

        let key = raw_key.trim();
        if key.is_empty() {
            return Err(ConfigError::InvalidKeyValue {
                line: line_num,
                text: line.to_string(),
            });
        }
        let val = unquote(raw_val.trim());

        match &active {
            ActiveSection::System => {
                config.system.insert(key.to_string(), val.to_string());
                if key.eq_ignore_ascii_case("hostname") {
                    config.hostname = Some(val.to_string());
                } else if key.eq_ignore_ascii_case("default_personality") {
                    config.default_personality = Some(parse_personality(line_num, val)?);
                } else if key.eq_ignore_ascii_case("sandbox_default") {
                    config.sandbox_default = parse_bool(line_num, key, val)?;
                }
            }
            ActiveSection::Service(svc_idx) => {
                let svc = &mut raw_services[*svc_idx];
                if key.eq_ignore_ascii_case("binary_path") || key.eq_ignore_ascii_case("binary") {
                    svc.binary_path = Some(val.to_string());
                } else if key.eq_ignore_ascii_case("personality") {
                    svc.personality = Some(parse_personality(line_num, val)?);
                } else if key.eq_ignore_ascii_case("dependencies")
                    || key.eq_ignore_ascii_case("depends")
                    || key.eq_ignore_ascii_case("after")
                {
                    for dep in val.split(',') {
                        let dep = dep.trim();
                        if !dep.is_empty() {
                            svc.dependencies.push(dep.to_string());
                        }
                    }
                } else if key.eq_ignore_ascii_case("restart_on_exit")
                    || key.eq_ignore_ascii_case("restart")
                {
                    svc.restart_on_exit = Some(parse_bool(line_num, key, val)?);
                } else if key.eq_ignore_ascii_case("sandbox_strict")
                    || key.eq_ignore_ascii_case("sandbox")
                {
                    svc.sandbox_strict = Some(parse_bool(line_num, key, val)?);
                }
            }
            ActiveSection::Other(section_name) => {
                if let Some(map) = config.sections.get_mut(section_name) {
                    map.insert(key.to_string(), val.to_string());
                }
            }
        }
    }

    let default_personality = config.default_personality.unwrap_or(Personality::Native);
    let default_sandbox = config.sandbox_default;

    for raw in raw_services {
        let binary_path = raw.binary_path.filter(|p| !p.is_empty()).ok_or_else(|| {
            ConfigError::MissingBinaryPath {
                service: raw.name.clone(),
            }
        })?;

        config.services.push(ServiceSpec {
            name: raw.name,
            binary_path,
            personality: raw.personality.unwrap_or(default_personality),
            dependencies: raw.dependencies,
            restart_on_exit: raw.restart_on_exit.unwrap_or(false),
            sandbox_strict: raw.sandbox_strict.unwrap_or(default_sandbox),
        });
    }

    Ok(config)
}

fn strip_service_prefix(header: &str) -> Option<&str> {
    let prefix = "service.";
    if header.len() >= prefix.len() && header[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&header[prefix.len()..])
    } else {
        None
    }
}

fn strip_comment(line: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    for (idx, ch) in line.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '#' | ';' if !in_single && !in_double => return &line[..idx],
            _ => {}
        }
    }
    line
}

fn unquote(val: &str) -> &str {
    if val.len() >= 2
        && ((val.starts_with('"') && val.ends_with('"'))
            || (val.starts_with('\'') && val.ends_with('\'')))
    {
        &val[1..val.len() - 1]
    } else {
        val
    }
}

fn parse_bool(line: usize, key: &str, val: &str) -> Result<bool, ConfigError> {
    if val.eq_ignore_ascii_case("true") || val.eq_ignore_ascii_case("yes") || val == "1" {
        Ok(true)
    } else if val.eq_ignore_ascii_case("false") || val.eq_ignore_ascii_case("no") || val == "0" {
        Ok(false)
    } else {
        Err(ConfigError::InvalidBool {
            line,
            key: key.to_string(),
            value: val.to_string(),
        })
    }
}

fn parse_personality(line: usize, val: &str) -> Result<Personality, ConfigError> {
    if val.eq_ignore_ascii_case("native") {
        Ok(Personality::Native)
    } else if val.eq_ignore_ascii_case("posix") || val.eq_ignore_ascii_case("linux") {
        Ok(Personality::Posix)
    } else if val.eq_ignore_ascii_case("nt") || val.eq_ignore_ascii_case("windows") {
        Ok(Personality::Nt)
    } else if val.eq_ignore_ascii_case("darwin") || val.eq_ignore_ascii_case("macos") {
        Ok(Personality::Darwin)
    } else {
        Err(ConfigError::InvalidPersonality {
            line,
            value: val.to_string(),
        })
    }
}
