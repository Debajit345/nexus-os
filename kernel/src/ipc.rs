//! Mach-inspired zero-copy IPC channels/ports and NT/io_uring-inspired async completion queues.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;

use nexus_abi::AbiError;

use crate::mm::SpinLock;

/// Zero-copy IPC message transferred across a channel endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub sender: u64,
    pub tag: u64,
    pub data: Vec<u8>,
    pub caps: Vec<u64>,
}

impl Message {
    pub fn new(sender: u64, tag: u64, data: Vec<u8>, caps: Vec<u64>) -> Self {
        Self {
            sender,
            tag,
            data,
            caps,
        }
    }
}

/// One endpoint of a bidirectional Mach-style channel/port pair.
struct Endpoint {
    peer: u64,
    inbox: VecDeque<Message>,
    closed: bool,
}

struct ChannelRegistry {
    endpoints: BTreeMap<u64, Endpoint>,
    next_id: u64,
}

impl ChannelRegistry {
    const fn new() -> Self {
        Self {
            endpoints: BTreeMap::new(),
            next_id: 1,
        }
    }

    fn create_channel(&mut self) -> (u64, u64) {
        let ep_a = self.next_id;
        let ep_b = self.next_id + 1;
        self.next_id += 2;

        self.endpoints.insert(
            ep_a,
            Endpoint {
                peer: ep_b,
                inbox: VecDeque::new(),
                closed: false,
            },
        );
        self.endpoints.insert(
            ep_b,
            Endpoint {
                peer: ep_a,
                inbox: VecDeque::new(),
                closed: false,
            },
        );

        (ep_a, ep_b)
    }

    fn send(&mut self, endpoint_id: u64, msg: Message) -> Result<(), AbiError> {
        let peer_id = {
            let ep = self
                .endpoints
                .get(&endpoint_id)
                .ok_or(AbiError::InvalidHandle)?;
            if ep.closed {
                return Err(AbiError::InvalidHandle);
            }
            ep.peer
        };

        let peer = self
            .endpoints
            .get_mut(&peer_id)
            .ok_or(AbiError::InvalidHandle)?;
        if peer.closed {
            return Err(AbiError::InvalidHandle);
        }
        peer.inbox.push_back(msg);
        Ok(())
    }

    fn recv(&mut self, endpoint_id: u64) -> Option<Message> {
        let ep = self.endpoints.get_mut(&endpoint_id)?;
        ep.inbox.pop_front()
    }

    fn close(&mut self, endpoint_id: u64) -> Result<(), AbiError> {
        let ep = self
            .endpoints
            .get_mut(&endpoint_id)
            .ok_or(AbiError::InvalidHandle)?;
        ep.closed = true;
        ep.inbox.clear();
        Ok(())
    }

    fn pending_count(&self, endpoint_id: u64) -> usize {
        self.endpoints
            .get(&endpoint_id)
            .map(|ep| ep.inbox.len())
            .unwrap_or(0)
    }
}

static CHANNELS: SpinLock<ChannelRegistry> = SpinLock::new(ChannelRegistry::new());

/// Create a connected pair of bidirectional IPC channel endpoints `(ep_a, ep_b)`.
pub fn create_channel() -> (u64, u64) {
    CHANNELS.lock().create_channel()
}

/// Send a message from `endpoint_id` to its peer endpoint (zero-copy move of `Message`).
pub fn send(endpoint_id: u64, msg: Message) -> Result<(), AbiError> {
    CHANNELS.lock().send(endpoint_id, msg)
}

/// Receive the next queued message on `endpoint_id`, if any.
pub fn recv(endpoint_id: u64) -> Option<Message> {
    CHANNELS.lock().recv(endpoint_id)
}

/// Close an IPC channel endpoint.
#[allow(dead_code)]
pub fn close_endpoint(endpoint_id: u64) -> Result<(), AbiError> {
    CHANNELS.lock().close(endpoint_id)
}

/// Return the number of queued messages waiting on `endpoint_id`.
#[allow(dead_code)]
pub fn pending_count(endpoint_id: u64) -> usize {
    CHANNELS.lock().pending_count(endpoint_id)
}

/// Asynchronous I/O or IPC completion event (NT IOCP / io_uring style).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CompletionEvent {
    pub user_data: u64,
    pub status: i64,
    pub bytes: usize,
}

/// Ring/FIFO completion queue for asynchronous operations.
pub struct CompletionQueue {
    events: VecDeque<CompletionEvent>,
}

impl CompletionQueue {
    pub const fn new() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }

    pub fn post(&mut self, event: CompletionEvent) {
        self.events.push_back(event);
    }

    pub fn poll(&mut self) -> Option<CompletionEvent> {
        self.events.pop_front()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

struct CompletionRegistry {
    queues: BTreeMap<u64, CompletionQueue>,
    next_id: u64,
}

impl CompletionRegistry {
    const fn new() -> Self {
        Self {
            queues: BTreeMap::new(),
            next_id: 1,
        }
    }

    fn create_queue(&mut self) -> u64 {
        let qid = self.next_id;
        self.next_id += 1;
        self.queues.insert(qid, CompletionQueue::new());
        qid
    }

    fn post(&mut self, qid: u64, event: CompletionEvent) -> Result<(), AbiError> {
        let q = self.queues.get_mut(&qid).ok_or(AbiError::InvalidHandle)?;
        q.post(event);
        Ok(())
    }

    fn poll(&mut self, qid: u64) -> Option<CompletionEvent> {
        self.queues.get_mut(&qid)?.poll()
    }

    fn close(&mut self, qid: u64) -> Result<(), AbiError> {
        self.queues
            .remove(&qid)
            .map(|_| ())
            .ok_or(AbiError::InvalidHandle)
    }

    fn len(&self, qid: u64) -> usize {
        self.queues.get(&qid).map(|q| q.len()).unwrap_or(0)
    }
}

static COMPLETION_QUEUES: SpinLock<CompletionRegistry> = SpinLock::new(CompletionRegistry::new());

/// Create a new asynchronous completion queue and return its queue ID.
pub fn create_queue() -> u64 {
    COMPLETION_QUEUES.lock().create_queue()
}

/// Post a completion event to queue `qid`.
pub fn post(qid: u64, event: CompletionEvent) -> Result<(), AbiError> {
    COMPLETION_QUEUES.lock().post(qid, event)
}

/// Poll the next completion event from queue `qid`, if available.
pub fn poll(qid: u64) -> Option<CompletionEvent> {
    COMPLETION_QUEUES.lock().poll(qid)
}

/// Close and destroy completion queue `qid`.
#[allow(dead_code)]
pub fn close_queue(qid: u64) -> Result<(), AbiError> {
    COMPLETION_QUEUES.lock().close(qid)
}

/// Return the number of pending completion events in queue `qid`.
#[allow(dead_code)]
pub fn queue_len(qid: u64) -> usize {
    COMPLETION_QUEUES.lock().len(qid)
}
