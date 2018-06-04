// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::alloc::{Global, Alloc, Layout, LayoutErr, CollectionAllocErr, oom};
use std::marker;
use std::str;
use std::mem::{size_of, needs_drop};
use std::ops::{Deref, DerefMut};
use std::ptr::{self, Unique, NonNull};

use self::BucketState::*;

/// Integer type used for stored hash values.
///
/// No more than bit_width(usize) bits are needed to select a bucket.
///
/// The most significant bit is ours to use for tagging `SafeHash`.
///
/// (Even if we could have usize::MAX bytes allocated for buckets,
/// each bucket stores at least a `HashUint`, so there can be no more than
/// usize::MAX / size_of(usize) buckets.)
type HashUint = u32;

const EMPTY_BUCKET: HashUint = 0;
const EMPTY: u32 = 1;

/// Special `Unique<HashUint>` that uses the lower bit of the pointer
/// to expose a boolean tag.
/// Note: when the pointer is initialized to EMPTY `.ptr()` will return
/// null and the tag functions shouldn't be used.
struct TaggedHashUintPtr(Unique<HashUint>);

// type BucketType = String;
type BucketType = (u32, u32);
// type BucketType = (u32, u32);

impl TaggedHashUintPtr {
    #[inline]
    unsafe fn new(ptr: *mut HashUint) -> Self {
        debug_assert!(ptr as usize & 1 == 0 || ptr as usize == EMPTY as usize);
        TaggedHashUintPtr(Unique::new_unchecked(ptr))
    }

    #[inline]
    fn set_tag(&mut self, value: bool) {
        let mut usize_ptr = self.0.as_ptr() as usize;
        unsafe {
            if value {
                usize_ptr |= 1;
            } else {
                usize_ptr &= !1;
            }
            self.0 = Unique::new_unchecked(usize_ptr as *mut HashUint)
        }
    }

    #[inline]
    fn tag(&self) -> bool {
        (self.0.as_ptr() as usize) & 1 == 1
    }

    #[inline]
    fn ptr(&self) -> *mut HashUint {
        (self.0.as_ptr() as usize & !1) as *mut HashUint
    }
}

/// The raw hashtable, providing safe-ish access to the unzipped and highly
/// optimized arrays of hashes, and key-value pairs.
///
/// This design is a lot faster than the naive
/// `Vec<Option<(u64, K, V)>>`, because we don't pay for the overhead of an
/// option on every element, and we get a generally more cache-aware design.
///
/// Essential invariants of this structure:
///
///   - if `t.hashes[i] == EMPTY_BUCKET`, then `Bucket::at_index(&t, i).raw`
///     points to 'undefined' contents. Don't read from it. This invariant is
///     enforced outside this module with the `EmptyBucket`, `FullBucket`,
///     and `SafeHash` types.
///
///   - An `EmptyBucket` is only constructed at an index with
///     a hash of EMPTY_BUCKET.
///
///   - A `FullBucket` is only constructed at an index with a
///     non-EMPTY_BUCKET hash.
///
///   - A `SafeHash` is only constructed for non-`EMPTY_BUCKET` hash. We get
///     around hashes of zero by changing them to 0x8000_0000_0000_0000,
///     which will likely map to the same bucket, while not being confused
///     with "empty".
///
///   - Both "arrays represented by pointers" are the same length:
///     `capacity`. This is set at creation and never changes. The arrays
///     are unzipped and are more cache aware (scanning through 8 hashes
///     brings in at most 2 cache lines, since they're all right beside each
///     other). This layout may waste space in padding such as in a map from
///     u64 to u8, but is a more cache conscious layout as the key-value pairs
///     are only very shortly probed and the desired value will be in the same
///     or next cache line.
///
/// You can kind of think of this module/data structure as a safe wrapper
/// around just the "table" part of the hashtable. It enforces some
/// invariants at the type level and employs some performance trickery,
/// but in general is just a tricked out `Vec<Option<(u64, K, V)>>`.
///
/// The hashtable also exposes a special boolean tag. The tag defaults to false
/// when the RawTable is created and is accessible with the `tag` and `set_tag`
/// functions.
pub struct RawTable<V> {
    capacity_mask: usize,
    size: usize,
    hashes: TaggedHashUintPtr,
    pub raw_text_data: Vec<u8>,

    // Because K/V do not appear directly in any of the types in the struct,
    // inform rustc that in fact instances of K and V are reachable from here.
    marker: marker::PhantomData<(BucketType, V)>,
}

// An unsafe view of a RawTable bucket
// Valid indexes are within [0..table_capacity)
pub struct RawBucket<V> {
    hash_start: *mut HashUint,
    // We use *const to ensure covariance with respect to K and V
    pair_start: *const (BucketType, V),
    idx: usize,
    _marker: marker::PhantomData<(BucketType, V)>,
}

impl<V> Copy for RawBucket<V> {}
impl<V> Clone for RawBucket<V> {
    fn clone(&self) -> RawBucket<V> {
        *self
    }
}

pub struct Bucket<V, M> {
    raw: RawBucket<V>,
    table: M,
}

impl<V, M: Copy> Copy for Bucket<V, M> {}
impl<V, M: Copy> Clone for Bucket<V, M> {
    fn clone(&self) -> Bucket<V, M> {
        *self
    }
}

pub struct EmptyBucket<V, M> {
    raw: RawBucket<V>,
    table: M,
}

pub struct FullBucket<V, M> {
    raw: RawBucket<V>,
    table: M,
}

pub type FullBucketMut<'table, V> = FullBucket<V, &'table mut RawTable<V>>;

pub enum BucketState<V, M> {
    Empty(EmptyBucket<V, M>),
    Full(FullBucket<V, M>),
}

/// A hash that is not zero, since we use a hash of zero to represent empty
/// buckets.
#[derive(PartialEq, Copy, Clone)]
pub struct SafeHash {
    hash: HashUint,
}

impl SafeHash {
    /// Peek at the hash value, which is guaranteed to be non-zero.
    #[inline(always)]
    pub fn inspect(&self) -> HashUint {
        self.hash
    }

    #[inline(always)]
    pub fn new_u32(hash: u32) -> Self {
        // We need to avoid 0 in order to prevent collisions with
        // EMPTY_HASH. We can maintain our precious uniform distribution
        // of initial indexes by unconditionally setting the MSB,
        // effectively reducing the hashes by one bit.
        //
        // Truncate hash to fit in `HashUint`.
        let hash_bits = size_of::<HashUint>() * 8;
        SafeHash { hash: (1 << (hash_bits - 1)) | (hash as HashUint) }
    }
}

// `replace` casts a `*HashUint` to a `*SafeHash`. Since we statically
// ensure that a `FullBucket` points to an index with a non-zero hash,
// and a `SafeHash` is just a `HashUint` with a different name, this is
// safe.
//
// This test ensures that a `SafeHash` really IS the same size as a
// `HashUint`. If you need to change the size of `SafeHash` (and
// consequently made this test fail), `replace` needs to be
// modified to no longer assume this.
#[test]
fn can_alias_safehash_as_hash() {
    assert_eq!(size_of::<SafeHash>(), size_of::<HashUint>())
}

// RawBucket methods are unsafe as it's possible to
// make a RawBucket point to invalid memory using safe code.
impl<V> RawBucket<V> {
    unsafe fn hash(&self) -> *mut HashUint {
        self.hash_start.offset(self.idx as isize)
    }
    unsafe fn pair(&self) -> *mut (BucketType, V) {
        self.pair_start.offset(self.idx as isize) as *mut (BucketType, V)
    }
}

// Buckets hold references to the table.
impl<V, M> FullBucket<V, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
    /// Borrow a mutable reference to the table.
    pub fn table_mut(&mut self) -> &mut M {
        &mut self.table
    }
    /// Move out the reference to the table.
    pub fn into_table(self) -> M {
        self.table
    }
    /// Get the raw index.
    pub fn index(&self) -> usize {
        self.raw.idx
    }
}

impl<V, M> EmptyBucket<V, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
    /// Borrow a mutable reference to the table.
    pub fn table_mut(&mut self) -> &mut M {
        &mut self.table
    }
}

impl<V, M> Bucket<V, M> {
    /// Get the raw index.
    pub fn index(&self) -> usize {
        self.raw.idx
    }
}

impl<V, M> Deref for FullBucket<V, M>
    where M: Deref<Target = RawTable<V>>
{
    type Target = RawTable<V>;
    fn deref(&self) -> &RawTable<V> {
        &self.table
    }
}

/// `Put` is implemented for types which provide access to a table and cannot be invalidated
///  by filling a bucket. A similar implementation for `Take` is possible.
pub trait Put<V> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<V>;
}


impl<'t, V> Put<V> for &'t mut RawTable<V> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<V> {
        *self
    }
}

impl<V, M> Put<V> for Bucket<V, M>
    where M: Put<V>
{
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<V> {
        self.table.borrow_table_mut()
    }
}

impl<V, M> Put<V> for FullBucket<V, M>
    where M: Put<V>
{
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<V> {
        self.table.borrow_table_mut()
    }
}

impl<V, M: Deref<Target = RawTable<V>>> Bucket<V, M> {
    pub fn new(table: M, hash: SafeHash) -> Bucket<V, M> {
        Bucket::at_index(table, hash.inspect() as usize)
    }

    pub fn at_index(table: M, ib_index: usize) -> Bucket<V, M> {
        // if capacity is 0, then the RawBucket will be populated with bogus pointers.
        // This is an uncommon case though, so avoid it in release builds.
        debug_assert!(table.capacity() > 0,
                      "Table should have capacity at this point");
        let ib_index = ib_index & table.capacity_mask;
        Bucket {
            raw: table.raw_bucket_at(ib_index),
            table,
        }
    }

    pub fn first(table: M) -> Bucket<V, M> {
        Bucket {
            raw: table.raw_bucket_at(0),
            table,
        }
    }

    // "So a few of the first shall be last: for many be called,
    // but few chosen."
    //
    // We'll most likely encounter a few buckets at the beginning that
    // have their initial buckets near the end of the table. They were
    // placed at the beginning as the probe wrapped around the table
    // during insertion. We must skip forward to a bucket that won't
    // get reinserted too early and won't unfairly steal others spot.
    // This eliminates the need for robin hood.
    pub fn head_bucket(table: M) -> Bucket<V, M> {
        let mut bucket = Bucket::first(table);

        loop {
            bucket = match bucket.peek() {
                Full(full) => {
                    if full.displacement() == 0 {
                        // This bucket occupies its ideal spot.
                        // It indicates the start of another "cluster".
                        bucket = full.into_bucket();
                        break;
                    }
                    // Leaving this bucket in the last cluster for later.
                    full.into_bucket()
                }
                Empty(b) => {
                    // Encountered a hole between clusters.
                    b.into_bucket()
                }
            };
            bucket.next();
        }
        bucket
    }

    /// Reads a bucket at a given index, returning an enum indicating whether
    /// it's initialized or not. You need to match on this enum to get
    /// the appropriate types to call most of the other functions in
    /// this module.
    pub fn peek(self) -> BucketState<V, M> {
        match unsafe { *self.raw.hash() } {
            EMPTY_BUCKET => {
                Empty(EmptyBucket {
                    raw: self.raw,
                    table: self.table,
                })
            }
            _ => {
                Full(FullBucket {
                    raw: self.raw,
                    table: self.table,
                })
            }
        }
    }

    /// Modifies the bucket in place to make it point to the next slot.
    pub fn next(&mut self) {
        self.raw.idx = self.raw.idx.wrapping_add(1) & self.table.capacity_mask;
    }

}

impl<V, M: Deref<Target = RawTable<V>>> EmptyBucket<V, M> {
    #[inline]
    pub fn into_bucket(self) -> Bucket<V, M> {
        Bucket {
            raw: self.raw,
            table: self.table,
        }
    }

}

impl<V, M: Deref<Target = RawTable<V>>> EmptyBucket<V, M>
    where M: Put<V>
{
    /// Puts given key and value pair, along with the key's hash,
    /// into this bucket in the hashtable. Note how `self` is 'moved' into
    /// this function, because this slot will no longer be empty when
    /// we return! A `FullBucket` is returned for later use, pointing to
    /// the newly-filled slot in the hashtable.
    ///
    /// Use `make_hash` to construct a `SafeHash` to pass to this function.
    pub fn put(mut self, hash: SafeHash, key: &BucketType, value: V) -> FullBucket<V, M> {
        unsafe {
            *self.raw.hash() = hash.inspect();
            // self.table.borrow_table_mut().raw_text_data.extend(key.as_bytes());
            ptr::write(self.raw.pair(), (*key, value));
            self.table.borrow_table_mut().size += 1;
        }

        FullBucket {
            raw: self.raw,
            table: self.table,
        }
    }
}

impl<V, M: Deref<Target = RawTable<V>>> FullBucket<V, M> {
    #[inline]
    pub fn next(self) -> Bucket<V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    #[inline]
    pub fn into_bucket(self) -> Bucket<V, M> {
        Bucket {
            raw: self.raw,
            table: self.table,
        }
    }

    /// Duplicates the current position. This can be useful for operations
    /// on two or more buckets.
    pub fn stash(self) -> FullBucket<V, Self> {
        FullBucket {
            raw: self.raw,
            table: self,
        }
    }

    /// Get the distance between this bucket and the 'ideal' location
    /// as determined by the key's hash stored in it.
    ///
    /// In the cited blog posts above, this is called the "distance to
    /// initial bucket", or DIB. Also known as "probe count".
    pub fn displacement(&self) -> usize {
        // Calculates the distance one has to travel when going from
        // `hash mod capacity` onwards to `idx mod capacity`, wrapping around
        // if the destination is not reached before the end of the table.
        (self.raw.idx.wrapping_sub(self.hash().inspect() as usize)) & self.table.capacity_mask
    }

    #[inline]
    pub fn hash(&self) -> SafeHash {
        unsafe { SafeHash { hash: *self.raw.hash() } }
    }

    /// Gets references to the key and value at a given index.
    pub fn read(&self) -> (&str, &V) {
        unsafe {
            let pair_ptr = self.raw.pair();
            let text_offsets = (*pair_ptr).0;
            let slice = &self.table.raw_text_data[text_offsets.0 as usize .. text_offsets.0 as usize + text_offsets.1 as usize];
            (str::from_utf8_unchecked(&slice), &(*pair_ptr).1)
            // (&(*pair_ptr).0, &(*pair_ptr).1)
        }
    }
}

// We take a mutable reference to the table instead of accepting anything that
// implements `DerefMut` to prevent fn `take` from being called on `stash`ed
// buckets.
impl<'t, V> FullBucket<V, &'t mut RawTable<V>> {
    /// Removes this bucket's key and value from the hashtable.
    ///
    /// This works similarly to `put`, building an `EmptyBucket` out of the
    /// taken bucket.
    pub fn take(self) -> (EmptyBucket<V, &'t mut RawTable<V>>, BucketType, V) {
        self.table.size -= 1;

        unsafe {
            *self.raw.hash() = EMPTY_BUCKET;
            let (k, v) = ptr::read(self.raw.pair());
            (EmptyBucket {
                 raw: self.raw,
                 table: self.table,
             },
            k,
            v)
        }
    }
}

// This use of `Put` is misleading and restrictive, but safe and sufficient for our use cases
// where `M` is a full bucket or table reference type with mutable access to the table.
impl<V, M> FullBucket<V, M>
    where M: Put<V>
{
    pub fn replace(&mut self, h: SafeHash, k: BucketType, v: V) -> (SafeHash, BucketType, V) {
        unsafe {
            let old_hash = ptr::replace(self.raw.hash() as *mut SafeHash, h);
            let (old_key, old_val) = ptr::replace(self.raw.pair(), (k, v));

            (old_hash, old_key, old_val)
        }
    }
}

impl<V, M> FullBucket<V, M>
    where M: Deref<Target = RawTable<V>> + DerefMut
{
    /// Gets mutable references to the key and value at a given index.
    pub fn read_mut(&mut self) -> (&mut BucketType, &mut V) {
        unsafe {
            let pair_ptr = self.raw.pair();
            (&mut (*pair_ptr).0, &mut (*pair_ptr).1)
        }
    }
}

impl<'t, V, M> FullBucket<V, M>
    where M: Deref<Target = RawTable<V>> + 't
{
    /// Exchange a bucket state for immutable references into the table.
    /// Because the underlying reference to the table is also consumed,
    /// no further changes to the structure of the table are possible;
    /// in exchange for this, the returned references have a longer lifetime
    /// than the references returned by `read()`.
    pub fn into_refs(self) -> (&'t BucketType, &'t V) {
        unsafe {
            let pair_ptr = self.raw.pair();
            (&(*pair_ptr).0, &(*pair_ptr).1)
        }
    }
}

impl<'t, V, M> FullBucket<V, M>
    where M: Deref<Target = RawTable<V>> + DerefMut + 't
{
    /// This works similarly to `into_refs`, exchanging a bucket state
    /// for mutable references into the table.
    pub fn into_mut_refs(self) -> (&'t mut BucketType, &'t mut V) {
        unsafe {
            let pair_ptr = self.raw.pair();
            (&mut (*pair_ptr).0, &mut (*pair_ptr).1)
        }
    }
}


// Returns a Layout which describes the allocation required for a hash table,
// and the offset of the array of (key, value) pairs in the allocation.
fn calculate_layout<V>(capacity: usize) -> Result<(Layout, usize), LayoutErr> {
    let hashes = Layout::array::<HashUint>(capacity)?;
    let pairs = Layout::array::<(BucketType, V)>(capacity)?;
    hashes.extend(pairs)
}

pub(crate) enum Fallibility {
    Fallible,
    Infallible,
}

use self::Fallibility::*;

impl<V> RawTable<V> {
    /// Does not initialize the buckets. The caller should ensure they,
    /// at the very least, set every hash to EMPTY_BUCKET.
    /// Returns an error if it cannot allocate or capacity overflows.
    unsafe fn new_uninitialized_internal(
        capacity: usize,
        fallibility: Fallibility,
    ) -> Result<RawTable<V>, CollectionAllocErr> {
        if capacity == 0 {
            return Ok(RawTable {
                size: 0,
                raw_text_data: vec![],
                capacity_mask: capacity.wrapping_sub(1),
                hashes: TaggedHashUintPtr::new(EMPTY as *mut HashUint),
                marker: marker::PhantomData,
            });
        }

        // Allocating hashmaps is a little tricky. We need to allocate two
        // arrays, but since we know their sizes and alignments up front,
        // we just allocate a single array, and then have the subarrays
        // point into it.
        let (layout, _) = calculate_layout::<V>(capacity)?;
        let buffer = Global.alloc(layout).map_err(|e| match fallibility {
            Infallible => oom(layout),
            Fallible => e,
        })?;

        Ok(RawTable {
            capacity_mask: capacity.wrapping_sub(1),
            size: 0,
            raw_text_data: vec![], // TODO capacity
            hashes: TaggedHashUintPtr::new(buffer.cast().as_ptr()),
            marker: marker::PhantomData,
        })
    }

    /// Does not initialize the buckets. The caller should ensure they,
    /// at the very least, set every hash to EMPTY_BUCKET.
    unsafe fn new_uninitialized(capacity: usize) -> RawTable<V> {
        match Self::new_uninitialized_internal(capacity, Infallible) {
            Err(CollectionAllocErr::CapacityOverflow) => panic!("capacity overflow"),
            Err(CollectionAllocErr::AllocErr) => unreachable!(),
            Ok(table) => { table }
        }
    }

    fn raw_bucket_at(&self, index: usize) -> RawBucket<V> {
        let (_, pairs_offset) = calculate_layout::<V>(self.capacity()).unwrap();
        let buffer = self.hashes.ptr() as *mut u8;
        unsafe {
            RawBucket {
                hash_start: buffer as *mut HashUint,
                pair_start: buffer.add(pairs_offset) as *const (BucketType, V),
                idx: index,
                _marker: marker::PhantomData,
            }
        }
    }

    fn new_internal(
        capacity: usize,
        fallibility: Fallibility,
    ) -> Result<RawTable<V>, CollectionAllocErr> {
        unsafe {
            let ret = RawTable::new_uninitialized_internal(capacity, fallibility)?;
            ptr::write_bytes(ret.hashes.ptr(), 0, capacity);
            Ok(ret)
        }
    }

    /// Tries to create a new raw table from a given capacity. If it cannot allocate,
    /// it returns with AllocErr.
    pub fn try_new(capacity: usize) -> Result<RawTable<V>, CollectionAllocErr> {
        Self::new_internal(capacity, Fallible)
    }

    /// Creates a new raw table from a given capacity. All buckets are
    /// initially empty.
    pub fn new(capacity: usize) -> RawTable<V> {
        match Self::new_internal(capacity, Infallible) {
            Err(CollectionAllocErr::CapacityOverflow) => panic!("capacity overflow"),
            Err(CollectionAllocErr::AllocErr) => unreachable!(),
            Ok(table) => { table }
        }
    }

    /// The hashtable's capacity, similar to a vector's.
    pub fn capacity(&self) -> usize {
        self.capacity_mask.wrapping_add(1)
    }

    /// The number of elements ever `put` in the hashtable, minus the number
    /// of elements ever `take`n.
    pub fn size(&self) -> usize {
        self.size
    }

    fn raw_buckets(&self) -> RawBuckets<V> {
        RawBuckets {
            raw: self.raw_bucket_at(0),
            elems_left: self.size,
            marker: marker::PhantomData,
        }
    }

// TODO :enable Iterators
    // pub fn iter(&self) -> Iter<V> {
    //     Iter {
    //         iter: self.raw_buckets(),
    //     }
    // }

    // pub fn iter_mut(&mut self) -> IterMut<V> {
    //     IterMut {
    //         iter: self.raw_buckets(),
    //         _marker: marker::PhantomData,
    //     }
    // }

    // pub fn into_iter(self) -> IntoIter<V> {
    //     let RawBuckets { raw, elems_left, .. } = self.raw_buckets();
    //     // Replace the marker regardless of lifetime bounds on parameters.
    //     IntoIter {
    //         iter: RawBuckets {
    //             raw,
    //             elems_left,
    //             marker: marker::PhantomData,
    //         },
    //         table: self,
    //     }
    // }
//TODO :enable Iterators

    /// Drops buckets in reverse order. It leaves the table in an inconsistent
    /// state and should only be used for dropping the table's remaining
    /// entries. It's used in the implementation of Drop.
    unsafe fn rev_drop_buckets(&mut self) {
        // initialize the raw bucket past the end of the table
        let mut raw = self.raw_bucket_at(self.capacity());
        let mut elems_left = self.size;

        while elems_left != 0 {
            raw.idx -= 1;

            if *raw.hash() != EMPTY_BUCKET {
                elems_left -= 1;
                ptr::drop_in_place(raw.pair());
            }
        }
    }

    /// Set the table tag
    pub fn set_tag(&mut self, value: bool) {
        self.hashes.set_tag(value)
    }

    /// Get the table tag
    pub fn tag(&self) -> bool {
        self.hashes.tag()
    }
}

/// A raw iterator. The basis for some other iterators in this module. Although
/// this interface is safe, it's not used outside this module.
struct RawBuckets<'a, V> {
    raw: RawBucket<V>,
    elems_left: usize,

    // Strictly speaking, this should be &'a (K,V), but that would
    // require that K:'a, and we often use RawBuckets<'static...> for
    // move iterations, so that messes up a lot of other things. So
    // just use `&'a (K,V)` as this is not a publicly exposed type
    // anyway.
    marker: marker::PhantomData<&'a ()>,
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<'a, V> Clone for RawBuckets<'a, V> {
    fn clone(&self) -> RawBuckets<'a, V> {
        RawBuckets {
            raw: self.raw,
            elems_left: self.elems_left,
            marker: marker::PhantomData,
        }
    }
}


impl<'a, V> Iterator for RawBuckets<'a, V> {
    type Item = RawBucket<V>;

    fn next(&mut self) -> Option<RawBucket<V>> {
        if self.elems_left == 0 {
            return None;
        }

        loop {
            unsafe {
                let item = self.raw;
                self.raw.idx += 1;
                if *item.hash() != EMPTY_BUCKET {
                    self.elems_left -= 1;
                    return Some(item);
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}

impl<'a, V> ExactSizeIterator for RawBuckets<'a, V> {
    fn len(&self) -> usize {
        self.elems_left
    }
}

// TODO enable Iterators

/// Iterator over shared references to entries in a table.
// pub struct Iter<'a, V: 'a> {
//     iter: RawBuckets<'a, V>,
// }

// unsafe impl<'a, V: Sync> Sync for Iter<'a, V> {}
// unsafe impl<'a, V: Sync> Send for Iter<'a, V> {}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
// impl<'a, V> Clone for Iter<'a, V> {
//     fn clone(&self) -> Iter<'a, V> {
//         Iter {
//             iter: self.iter.clone(),
//         }
//     }
// }

/// Iterator over mutable references to entries in a table.
// pub struct IterMut<'a, V: 'a> {
//     iter: RawBuckets<'a, V>,
//     // To ensure invariance with respect to V
//     _marker: marker::PhantomData<&'a mut V>,
// }

// unsafe impl<'a, V: Sync> Sync for IterMut<'a, V> {}
// // Both K: Sync and K: Send are correct for IterMut's Send impl,
// // but Send is the more useful bound
// unsafe impl<'a, V: Send> Send for IterMut<'a, V> {}

// impl<'a, V: 'a> IterMut<'a, V> {
//     pub fn iter(&self) -> Iter<V> {
//         Iter {
//             iter: self.iter.clone(),
//         }
//     }
// }

/// Iterator over the entries in a table, consuming the table.
// pub struct IntoIter<V> {
//     table: RawTable<V>,
//     iter: RawBuckets<'static, V>,
// }

// unsafe impl<V: Sync> Sync for IntoIter<V> {}
// unsafe impl<V: Send> Send for IntoIter<V> {}

// impl<V> IntoIter<V> {
//     pub fn iter(&self) -> Iter<V> {
//         Iter {
//             iter: self.iter.clone(),
//         }
//     }
// }

// impl<'a, V> Iterator for Iter<'a, V> {
//     type Item = (&'a BucketType, &'a V);

//     fn next(&mut self) -> Option<(&'a BucketType, &'a V)> {
//         self.iter.next().map(|raw| unsafe {
//             let pair_ptr = raw.pair();
//             (&(*pair_ptr).0, &(*pair_ptr).1)
//         })
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }

// impl<'a, V> ExactSizeIterator for Iter<'a, V> {
//     fn len(&self) -> usize {
//         self.iter.len()
//     }
// }

// impl<'a, V> Iterator for IterMut<'a, V> {
//     type Item = (&'a BucketType, &'a mut V);

//     fn next(&mut self) -> Option<(&'a BucketType, &'a mut V)> {
//         self.iter.next().map(|raw| unsafe {
//             let pair_ptr = raw.pair();
//             (&(*pair_ptr).0, &mut (*pair_ptr).1)
//         })
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }

// impl<'a, V> ExactSizeIterator for IterMut<'a, V> {
//     fn len(&self) -> usize {
//         self.iter.len()
//     }
// }

// impl<V> Iterator for IntoIter<V> {
//     type Item = (SafeHash, BucketType, V);

//     fn next(&mut self) -> Option<(SafeHash, BucketType, V)> {
//         self.iter.next().map(|raw| {
//             self.table.size -= 1;
//             unsafe {
//                 let (k, v) = ptr::read(raw.pair());
//                 (SafeHash { hash: *raw.hash() }, k, v)
//             }
//         })
//     }

//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.iter.size_hint()
//     }
// }

// impl<V> ExactSizeIterator for IntoIter<V> {
//     fn len(&self) -> usize {
//         self.iter().len()
//     }
// }


impl<V: Clone> Clone for RawTable<V> {
    fn clone(&self) -> RawTable<V> {
        unsafe {
            let cap = self.capacity();
            let mut new_ht = RawTable::new_uninitialized(cap);
            new_ht.raw_text_data = self.raw_text_data.clone();
            let mut new_buckets = new_ht.raw_bucket_at(0);
            let mut buckets = self.raw_bucket_at(0);
            while buckets.idx < cap {
                *new_buckets.hash() = *buckets.hash();
                if *new_buckets.hash() != EMPTY_BUCKET {
                    let pair_ptr = buckets.pair();
                    let kv = ((*pair_ptr).0.clone(), (*pair_ptr).1.clone());
                    ptr::write(new_buckets.pair(), kv);
                }
                buckets.idx += 1;
                new_buckets.idx += 1;
            }

            new_ht.size = self.size();
            new_ht.set_tag(self.tag());

            new_ht
        }
    }
}

unsafe impl<#[may_dangle] V> Drop for RawTable<V> {
    fn drop(&mut self) {
        if self.capacity() == 0 {
            return;
        }

        // This is done in reverse because we've likely partially taken
        // some elements out with `.into_iter()` from the front.
        // Check if the size is 0, so we don't do a useless scan when
        // dropping empty tables such as on resize.
        // Also avoid double drop of elements that have been already moved out.
        unsafe {
            if needs_drop::<(BucketType, V)>() {
                // avoid linear runtime for types that don't need drop
                self.rev_drop_buckets();
            }
        }

        let (layout, _) = calculate_layout::<V>(self.capacity()).unwrap();
        unsafe {
            Global.dealloc(NonNull::new_unchecked(self.hashes.ptr()).as_opaque(), layout);
            // Remember how everything was allocated out of one buffer
            // during initialization? We only need one call to free here.
        }
    }
}
