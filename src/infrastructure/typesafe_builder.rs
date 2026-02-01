//! Builder pattern utilities
//!
//! This module provides utility types for implementing the builder
//! pattern with compile-time validation of required parameters.

use std::mem::MaybeUninit;

/// Represents an unset field in a builder pattern.
///
/// This is used to track at compile time that a required field has
/// not yet been set.
pub struct Unset<T> {
    _value: MaybeUninit<T>,
}

impl<T> std::default::Default for Unset<T> {
    fn default() -> Self {
        Self {
            _value: MaybeUninit::uninit(),
        }
    }
}

impl<T> Unset<T> {
    /// Creates a new unset field.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Represents a set field in a builder pattern.
///
/// This is used to track at compile time that a required field has
/// been set.
#[derive(Clone)]
pub struct Set<T> {
    value: T,
}

impl<T> Set<T> {
    /// Creates a new set field with the given value.
    ///
    /// # Parameters
    /// * `value` - The value to store in this field
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Consumes the wrapper and returns the inner value.
    ///
    /// # Returns
    /// The inner value
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Returns a reference to the inner value.
    ///
    /// # Returns
    /// A reference to the inner value
    pub fn as_ref_inner(&self) -> &T {
        &self.value
    }
}
