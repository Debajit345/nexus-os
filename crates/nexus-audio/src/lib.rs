#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

//! Clean-room `no_std` + `alloc` DSP audio mixer crate for Nexus OS.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Supported PCM sample formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Signed 16-bit integer PCM.
    I16,
    /// 32-bit floating-point PCM normalized to `[-1.0, 1.0]`.
    F32,
}

/// An audio buffer storing normalized `f32` samples along with stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: SampleFormat,
    samples: Vec<f32>,
}

impl AudioBuffer {
    /// Creates a silent buffer with the given frame count.
    pub fn silence(sample_rate: u32, channels: u16, frames: usize, format: SampleFormat) -> Self {
        let total = frames.saturating_mul(channels as usize);
        Self {
            sample_rate,
            channels,
            format,
            samples: vec![0.0; total],
        }
    }

    /// Creates an `AudioBuffer` from signed 16-bit PCM samples.
    pub fn from_i16(sample_rate: u32, channels: u16, data: &[i16]) -> Self {
        let samples = data.iter().map(|&s| (s as f32) / 32768.0).collect();
        Self {
            sample_rate,
            channels,
            format: SampleFormat::I16,
            samples,
        }
    }

    /// Creates an `AudioBuffer` from 32-bit floating-point PCM samples.
    pub fn from_f32(sample_rate: u32, channels: u16, data: &[f32]) -> Self {
        Self {
            sample_rate,
            channels,
            format: SampleFormat::F32,
            samples: data.to_vec(),
        }
    }

    /// Returns the normalized `f32` samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Returns the total number of samples across all channels.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` if the buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Converts the buffer into signed 16-bit PCM samples with clamping.
    pub fn to_i16_vec(&self) -> Vec<i16> {
        self.samples.iter().map(|&s| f32_to_i16(s)).collect()
    }
}

/// A single channel feeding into [`AudioMixer`].
#[derive(Debug, Clone, PartialEq)]
pub struct MixerChannel {
    pub id: u32,
    pub gain: f32,
    pub pan: f32,
    pub muted: bool,
    pub samples: Vec<i16>,
}

impl MixerChannel {
    /// Creates a new unmuted channel with unity gain (`1.0`) and center pan (`0.0`).
    pub fn new(id: u32, samples: Vec<i16>) -> Self {
        Self {
            id,
            gain: 1.0,
            pan: 0.0,
            muted: false,
            samples,
        }
    }

    /// Creates a channel from an [`AudioBuffer`].
    pub fn from_buffer(id: u32, buffer: &AudioBuffer) -> Self {
        Self::new(id, buffer.to_i16_vec())
    }

    /// Sets the linear gain multiplier.
    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }

    /// Sets the stereo pan in `[-1.0, 1.0]` (`-1.0` = left, `0.0` = center, `1.0` = right).
    pub fn with_pan(mut self, pan: f32) -> Self {
        self.pan = pan.clamp(-1.0, 1.0);
        self
    }

    /// Sets whether the channel is muted.
    pub fn with_muted(mut self, muted: bool) -> Self {
        self.muted = muted;
        self
    }
}

/// Multi-channel DSP audio mixer with gain, stereo panning, mute, and soft-clipping.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioMixer {
    channels: Vec<MixerChannel>,
    master_gain: f32,
    soft_clip: bool,
}

impl Default for AudioMixer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioMixer {
    /// Creates an empty mixer with unity master gain and soft-clipping enabled.
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
            master_gain: 1.0,
            soft_clip: true,
        }
    }

    /// Adds a channel to the mixer.
    pub fn add_channel(&mut self, channel: MixerChannel) {
        self.channels.push(channel);
    }

    /// Removes a channel by `id`, returning it if found.
    pub fn remove_channel(&mut self, id: u32) -> Option<MixerChannel> {
        let pos = self.channels.iter().position(|c| c.id == id)?;
        Some(self.channels.remove(pos))
    }

    /// Returns a mutable reference to a channel by `id`.
    pub fn channel_mut(&mut self, id: u32) -> Option<&mut MixerChannel> {
        self.channels.iter_mut().find(|c| c.id == id)
    }

    /// Returns all registered channels.
    pub fn channels(&self) -> &[MixerChannel] {
        &self.channels
    }

    /// Sets the master output gain.
    pub fn set_master_gain(&mut self, gain: f32) {
        self.master_gain = gain;
    }

    /// Enables or disables soft-clipping (when disabled, hard clamping is used).
    pub fn set_soft_clip(&mut self, enabled: bool) {
        self.soft_clip = enabled;
    }

    /// Mixes all unmuted channels into `out` (interleaved stereo or mono `i16` samples).
    ///
    /// Applies per-channel gain, stereo pan (even indices = left, odd indices = right),
    /// master gain, and soft-clipping / clamping to `[i16::MIN, i16::MAX]`.
    pub fn mix_into(&self, out: &mut [i16]) {
        if out.is_empty() {
            return;
        }

        let mut accum = vec![0.0_f32; out.len()];

        for ch in &self.channels {
            if ch.muted || ch.gain == 0.0 {
                continue;
            }
            let pan = ch.pan.clamp(-1.0, 1.0);
            let left_pan = (1.0 - pan).min(1.0);
            let right_pan = (1.0 + pan).min(1.0);

            let count = out.len().min(ch.samples.len());
            for (i, acc) in accum.iter_mut().enumerate().take(count) {
                let pan_gain = if i % 2 == 0 { left_pan } else { right_pan };
                let normalized = (ch.samples[i] as f32) / 32768.0;
                *acc += normalized * ch.gain * pan_gain;
            }
        }

        for (dst, &sum) in out.iter_mut().zip(accum.iter()) {
            let scaled = sum * self.master_gain;
            let shaped = if self.soft_clip {
                soft_clip_sample(scaled)
            } else {
                scaled.clamp(-1.0, 1.0)
            };
            *dst = f32_to_i16(shaped);
        }
    }
}

/// Smooth `no_std` soft-clipper: linear in `[-0.5, 0.5]`, smoothly saturating toward `[-1.0, 1.0]`.
fn soft_clip_sample(x: f32) -> f32 {
    if x.abs() <= 0.5 {
        x
    } else {
        // Rational tanh-like saturation for |x| > 0.5, continuous at +/-0.5?
        // Use x / (1.0 + (x.abs() - 0.5))? At x=0.5 -> 0.5/1.0 = 0.5, as x->inf -> 1.0!
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let ax = x.abs();
        let compressed = ax / (1.0 + (ax - 0.5));
        (sign * compressed).clamp(-1.0, 1.0)
    }
}

fn f32_to_i16(sample: f32) -> i16 {
    let clamped = sample.clamp(-1.0, 1.0);
    let scaled = clamped * 32767.0;
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    rounded.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_channel_mixing_sums_samples() {
        let mut mixer = AudioMixer::new();
        mixer.set_soft_clip(false);
        mixer.add_channel(MixerChannel::new(1, vec![4000, -4000, 2000, -2000]));
        mixer.add_channel(MixerChannel::new(2, vec![4000, -4000, 1000, 2000]));

        let mut out = [0i16; 4];
        mixer.mix_into(&mut out);

        assert!((out[0] - 8000).abs() <= 2);
        assert!((out[1] + 8000).abs() <= 2);
        assert!((out[2] - 3000).abs() <= 2);
        assert!(out[3].abs() <= 2);
    }

    #[test]
    fn muted_channel_is_excluded() {
        let mut mixer = AudioMixer::new();
        mixer.set_soft_clip(false);
        mixer.add_channel(MixerChannel::new(1, vec![5000, 5000]));
        mixer.add_channel(MixerChannel::new(2, vec![10000, 10000]).with_muted(true));

        let mut out = [0i16; 2];
        mixer.mix_into(&mut out);

        assert!((out[0] - 5000).abs() <= 2);
        assert!((out[1] - 5000).abs() <= 2);
    }

    #[test]
    fn clipping_prevents_overflow_in_both_modes() {
        let mut mixer = AudioMixer::new();
        mixer.add_channel(MixerChannel::new(1, vec![30000, -30000]));
        mixer.add_channel(MixerChannel::new(2, vec![30000, -30000]));

        // Soft-clipping mode
        let mut soft_out = [0i16; 2];
        mixer.mix_into(&mut soft_out);
        assert!(soft_out[0] > 20000);
        assert!(soft_out[1] < -20000);

        // Hard-clamping mode
        mixer.set_soft_clip(false);
        let mut hard_out = [0i16; 2];
        mixer.mix_into(&mut hard_out);
        assert_eq!(hard_out[0], i16::MAX);
        assert_eq!(hard_out[1], -32767);
    }

    #[test]
    fn stereo_pan_routes_left_and_right() {
        let mut mixer = AudioMixer::new();
        mixer.set_soft_clip(false);
        mixer.add_channel(MixerChannel::new(1, vec![8000, 8000]).with_pan(-1.0));

        let mut out = [0i16; 2];
        mixer.mix_into(&mut out);
        assert!((out[0] - 8000).abs() <= 2);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn audio_buffer_conversions() {
        let buf = AudioBuffer::from_i16(48000, 2, &[16384, -16384]);
        assert_eq!(buf.format, SampleFormat::I16);
        assert_eq!(buf.len(), 2);
        let back = buf.to_i16_vec();
        assert!((back[0] - 16384).abs() <= 2);
        assert!((back[1] + 16384).abs() <= 2);
    }
}
