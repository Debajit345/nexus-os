#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room init system, declarative `/etc/nexus.conf` parser, and
//! dependency-aware service manager for Nexus OS.

extern crate alloc;

pub mod config;
pub mod service;

pub use config::{parse_config, ConfigError, NexusConfig};
pub use service::{
    infer_personality_from_binary, ServiceError, ServiceManager, ServiceSpec, ServiceState,
};

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;
    use nexus_abi::Personality;

    use super::*;

    #[test]
    fn parse_declarative_nexus_conf() {
        let conf = r#"
# Nexus OS declarative system configuration
[system]
hostname = "nexus-workstation"
default_personality = native
sandbox_default = true
log_level = info

[service.ipcd]
binary_path = /sbin/ipcd
personality = native
restart_on_exit = true
sandbox_strict = true

[service.vfsd]
binary_path = "/sbin/vfsd"
personality = posix
dependencies = ipcd
restart_on_exit = yes

[service.win32_bridge]
binary = /sbin/ntbridge.exe
personality = nt
dependencies = ipcd, vfsd
restart = false
sandbox = false

[network]
dhcp = true
"#;

        let parsed = parse_config(conf).expect("valid config should parse");
        assert_eq!(parsed.hostname.as_deref(), Some("nexus-workstation"));
        assert_eq!(parsed.default_personality, Some(Personality::Native));
        assert!(parsed.sandbox_default);
        assert_eq!(
            parsed.system.get("log_level").map(String::as_str),
            Some("info")
        );
        assert_eq!(
            parsed
                .sections
                .get("network")
                .and_then(|m| m.get("dhcp"))
                .map(String::as_str),
            Some("true")
        );

        assert_eq!(parsed.services.len(), 3);

        let ipcd = &parsed.services[0];
        assert_eq!(ipcd.name, "ipcd");
        assert_eq!(ipcd.binary_path, "/sbin/ipcd");
        assert_eq!(ipcd.personality, Personality::Native);
        assert!(ipcd.dependencies.is_empty());
        assert!(ipcd.restart_on_exit);
        assert!(ipcd.sandbox_strict);

        let vfsd = &parsed.services[1];
        assert_eq!(vfsd.name, "vfsd");
        assert_eq!(vfsd.personality, Personality::Posix);
        assert_eq!(vfsd.dependencies, vec!["ipcd"]);
        assert!(vfsd.restart_on_exit);
        assert!(vfsd.sandbox_strict); // inherited from sandbox_default

        let bridge = &parsed.services[2];
        assert_eq!(bridge.name, "win32_bridge");
        assert_eq!(bridge.binary_path, "/sbin/ntbridge.exe");
        assert_eq!(bridge.personality, Personality::Nt);
        assert_eq!(bridge.dependencies, vec!["ipcd", "vfsd"]);
        assert!(!bridge.restart_on_exit);
        assert!(!bridge.sandbox_strict);
    }

    #[test]
    fn parse_config_rejects_invalid_inputs() {
        let missing_bin = "[service.broken]\npersonality = posix\n";
        assert_eq!(
            parse_config(missing_bin),
            Err(ConfigError::MissingBinaryPath {
                service: String::from("broken")
            })
        );

        let bad_bool = "[service.svc]\nbinary_path = /bin/svc\nrestart_on_exit = maybe\n";
        assert!(matches!(
            parse_config(bad_bool),
            Err(ConfigError::InvalidBool { line: 3, .. })
        ));

        let bad_personality = "[service.svc]\nbinary_path = /bin/svc\npersonality = plan9\n";
        assert!(matches!(
            parse_config(bad_personality),
            Err(ConfigError::InvalidPersonality { line: 3, .. })
        ));

        let dup_service = "[service.a]\nbinary_path = /bin/a\n[service.a]\nbinary_path = /bin/a2\n";
        assert_eq!(
            parse_config(dup_service),
            Err(ConfigError::DuplicateService {
                service: String::from("a")
            })
        );
    }

    #[test]
    fn dependency_ordered_startup_and_from_config() {
        let conf = r#"
[service.shell]
binary_path = /bin/nsh
personality = posix
dependencies = compositor, vfsd

[service.compositor]
binary_path = /sbin/nexus-gfx
dependencies = ipcd

[service.vfsd]
binary_path = /sbin/vfsd
dependencies = ipcd

[service.ipcd]
binary_path = /sbin/ipcd
"#;

        let parsed = parse_config(conf).expect("config should parse");
        let mut mgr = ServiceManager::from_config(&parsed).expect("registration should succeed");
        assert_eq!(mgr.len(), 4);
        assert_eq!(mgr.running_count(), 0);
        assert_eq!(mgr.status("shell"), Some(ServiceState::Stopped));

        let order = mgr.start_all().expect("topological startup should succeed");
        assert_eq!(mgr.running_count(), 4);
        assert_eq!(mgr.status("shell"), Some(ServiceState::Running));

        let pos = |name: &str| order.iter().position(|s| s == name).unwrap();
        assert!(pos("ipcd") < pos("compositor"));
        assert!(pos("ipcd") < pos("vfsd"));
        assert!(pos("compositor") < pos("shell"));
        assert!(pos("vfsd") < pos("shell"));
    }

    #[test]
    fn cycle_detection_and_missing_dependency() {
        let mut mgr = ServiceManager::new();
        let mut a = ServiceSpec::new("a", "/bin/a");
        a.dependencies.push(String::from("b"));
        let mut b = ServiceSpec::new("b", "/bin/b");
        b.dependencies.push(String::from("c"));
        let mut c = ServiceSpec::new("c", "/bin/c");
        c.dependencies.push(String::from("a"));

        mgr.register(a).unwrap();
        mgr.register(b).unwrap();
        mgr.register(c).unwrap();

        let err = mgr.start_all().unwrap_err();
        assert!(matches!(err, ServiceError::DependencyCycle(nodes) if nodes.len() == 3));
        assert_eq!(mgr.running_count(), 0);

        // Self-cycle
        let mut self_mgr = ServiceManager::new();
        let mut s = ServiceSpec::new("self_svc", "/bin/self");
        s.dependencies.push(String::from("self_svc"));
        self_mgr.register(s).unwrap();
        assert!(matches!(
            self_mgr.start_all(),
            Err(ServiceError::DependencyCycle(_))
        ));

        // Missing dependency
        let mut missing_mgr = ServiceManager::new();
        let mut m = ServiceSpec::new("app", "/bin/app");
        m.dependencies.push(String::from("nonexistent"));
        missing_mgr.register(m).unwrap();
        assert_eq!(
            missing_mgr.start_all(),
            Err(ServiceError::MissingDependency {
                service: String::from("app"),
                dependency: String::from("nonexistent"),
            })
        );
    }

    #[test]
    fn crash_restart_and_stop_handling() {
        let mut mgr = ServiceManager::new();

        let mut daemon = ServiceSpec::new("daemon", "/sbin/daemon");
        daemon.restart_on_exit = true;

        let oneshot = ServiceSpec::new("oneshot", "/sbin/oneshot");

        mgr.register(daemon).unwrap();
        mgr.register(oneshot).unwrap();
        mgr.start_all().unwrap();
        assert_eq!(mgr.running_count(), 2);

        // Crash of restartable service -> Restarting, restart_count incremented
        let state = mgr.notify_exit("daemon", 139).unwrap();
        assert_eq!(state, ServiceState::Restarting);
        assert_eq!(mgr.status("daemon"), Some(ServiceState::Restarting));
        assert_eq!(mgr.restart_count("daemon"), Some(1));
        assert_eq!(mgr.last_exit_code("daemon"), Some(139));
        assert_eq!(mgr.running_count(), 1);

        // Complete restart -> Running again
        assert_eq!(mgr.start("daemon").unwrap(), ServiceState::Running);
        assert_eq!(mgr.running_count(), 2);

        // Crash of non-restartable service -> Failed
        let state = mgr.notify_exit("oneshot", 1).unwrap();
        assert_eq!(state, ServiceState::Failed);
        assert_eq!(mgr.status("oneshot"), Some(ServiceState::Failed));
        assert_eq!(mgr.running_count(), 1);

        // Explicit stop -> Stopped
        mgr.stop("daemon").unwrap();
        assert_eq!(mgr.status("daemon"), Some(ServiceState::Stopped));
        assert_eq!(mgr.running_count(), 0);
    }

    #[test]
    fn infer_personality_from_magic_bytes() {
        assert_eq!(
            infer_personality_from_binary(&[0x7F, b'E', b'L', b'F']),
            Ok(Personality::Posix)
        );
        assert_eq!(
            infer_personality_from_binary(&[b'M', b'Z', 0x90, 0x00]),
            Ok(Personality::Nt)
        );
        assert_eq!(
            infer_personality_from_binary(&[0xCF, 0xFA, 0xED, 0xFE]),
            Ok(Personality::Darwin)
        );
    }
}
