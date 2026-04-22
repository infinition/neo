use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// Precise media timestamp with rational time base.
///
/// Avoids floating-point drift that plagues video processing.
/// All timestamps are stored as integer ticks of a rational time base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp {
    /// Tick count in the given time base.
    pub pts: i64,
    /// Time base numerator (e.g., 1 for 1/90000).
    pub num: u32,
    /// Time base denominator (e.g., 90000 for 1/90000).
    pub den: u32,
}

impl Timestamp {
    pub const ZERO: Self = Self { pts: 0, num: 1, den: 90_000 };

    pub fn new(pts: i64, num: u32, den: u32) -> Self {
        Self { pts, num, den }
    }

    /// Create from seconds.
    pub fn from_secs_f64(secs: f64, den: u32) -> Self {
        Self {
            pts: (secs * den as f64) as i64,
            num: 1,
            den,
        }
    }

    /// Convert to seconds as f64.
    pub fn as_secs_f64(&self) -> f64 {
        self.pts as f64 * self.num as f64 / self.den as f64
    }

    /// Convert to `Duration`.
    pub fn as_duration(&self) -> Duration {
        Duration::from_secs_f64(self.as_secs_f64())
    }

    /// Rescale to a different time base.
    pub fn rescale(&self, new_num: u32, new_den: u32) -> Self {
        let new_pts = self.pts as i128 * self.num as i128 * new_den as i128
            / (self.den as i128 * new_num as i128);
        Self {
            pts: new_pts as i64,
            num: new_num,
            den: new_den,
        }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_secs = self.as_secs_f64();
        let hours = (total_secs / 3600.0) as u32;
        let mins = ((total_secs % 3600.0) / 60.0) as u32;
        let secs = total_secs % 60.0;
        write!(f, "{:02}:{:02}:{:06.3}", hours, mins, secs)
    }
}
