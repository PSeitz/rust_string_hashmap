#![feature(try_reserve)]
#![feature(ptr_internals)]
#![feature(allocator_api)]
#![feature(dropck_eyepatch)]

#[macro_use]
extern crate log;
extern crate byteorder;

pub mod hasher;
mod table;
pub mod stacker;

use hasher::FnvYoshiBuildHasher;

// use self::Entry::*;
use self::VacantEntryState::*;

use std::alloc::{CollectionAllocErr};
use std::cmp::max;
use std::fmt::{self, Debug};
#[allow(deprecated)]
use std::hash::{BuildHasher};
use std::iter::{FromIterator, FusedIterator};
use std::mem::{self, replace};
use std::ops::{Deref, Index};
// use std::sys;

use table::{Bucket, EmptyBucket, FullBucket, FullBucketMut, RawTable, SafeHash};
use table::Fallibility::{self, Fallible, Infallible};
use table::BucketState::{Empty, Full};

const MIN_NONZERO_RAW_CAPACITY: usize = 32;     // must be a power of two

/// The default behavior of HashMap implements a maximum load factor of 90.9%.
#[derive(Clone)]
struct DefaultResizePolicy;

impl DefaultResizePolicy {
    #[inline]
    fn new() -> DefaultResizePolicy {
        DefaultResizePolicy
    }

    /// A hash map's "capacity" is the number of elements it can hold without
    /// being resized. Its "raw capacity" is the number of slots required to
    /// provide that capacity, accounting for maximum loading. The raw capacity
    /// is always zero or a power of two.
    #[inline]
    fn try_raw_capacity(&self, len: usize) -> Result<usize, CollectionAllocErr> {
        if len == 0 {
            Ok(0)
        } else {
            // 1. Account for loading: `raw_capacity >= len * 1.1`.
            // 2. Ensure it is a power of two.
            // 3. Ensure it is at least the minimum size.
            let mut raw_cap = len.checked_mul(11)
                .map(|l| l / 10)
                .and_then(|l| l.checked_next_power_of_two())
                .ok_or(CollectionAllocErr::CapacityOverflow)?;

            raw_cap = max(MIN_NONZERO_RAW_CAPACITY, raw_cap);
            Ok(raw_cap)
        }
    }

    #[inline]
    fn raw_capacity(&self, len: usize) -> usize {
        self.try_raw_capacity(len).expect("raw_capacity overflow")
    }

    /// The capacity of the given raw capacity.
    #[inline]
    fn capacity(&self, raw_cap: usize) -> usize {
        // This doesn't have to be checked for overflow since allocation size
        // in bytes will overflow earlier than multiplication by 10.
        //
        // As per https://github.com/rust-lang/rust/pull/30991 this is updated
        // to be: (raw_cap * den + den - 1) / num
        (raw_cap * 10 + 10 - 1) / 11
    }
}

// The main performance trick in this hashmap is called Robin Hood Hashing.
// It gains its excellent performance from one essential operation:
//
//    If an insertion collides with an existing element, and that element's
//    "probe distance" (how far away the element is from its ideal location)
//    is higher than how far we've already probed, swap the elements.
//
// This massively lowers variance in probe distance, and allows us to get very
// high load factors with good performance. The 90% load factor I use is rather
// conservative.
//
// > Why a load factor of approximately 90%?
//
// In general, all the distances to initial buckets will converge on the mean.
// At a load factor of α, the odds of finding the target bucket after k
// probes is approximately 1-α^k. If we set this equal to 50% (since we converge
// on the mean) and set k=8 (64-byte cache line / 8-byte hash), α=0.92. I round
// this down to make the math easier on the CPU and avoid its FPU.
// Since on average we start the probing in the middle of a cache line, this
// strategy pulls in two cache lines of hashes on every lookup. I think that's
// pretty good, but if you want to trade off some space, it could go down to one
// cache line on average with an α of 0.84.
//
// > Wait, what? Where did you get 1-α^k from?
//
// On the first probe, your odds of a collision with an existing element is α.
// The odds of doing this twice in a row is approximately α^2. For three times,
// α^3, etc. Therefore, the odds of colliding k times is α^k. The odds of NOT
// colliding after k tries is 1-α^k.
//
// The paper from 1986 cited below mentions an implementation which keeps track
// of the distance-to-initial-bucket histogram. This approach is not suitable
// for modern architectures because it requires maintaining an internal data
// structure. This allows very good first guesses, but we are most concerned
// with guessing entire cache lines, not individual indexes. Furthermore, array
// accesses are no longer linear and in one direction, as we have now. There
// is also memory and cache pressure that this would entail that would be very
// difficult to properly see in a microbenchmark.
//
// ## Future Improvements (FIXME!)
//
// Allow the load factor to be changed dynamically and/or at initialization.
//
// Also, would it be possible for us to reuse storage when growing the
// underlying table? This is exactly the use case for 'realloc', and may
// be worth exploring.
//
// ## Future Optimizations (FIXME!)
//
// Another possible design choice that I made without any real reason is
// parameterizing the raw table over keys and values. Technically, all we need
// is the size and alignment of keys and values, and the code should be just as
// efficient (well, we might need one for power-of-two size and one for not...).
// This has the potential to reduce code bloat in rust executables, without
// really losing anything except 4 words (key size, key alignment, val size,
// val alignment) which can be passed in to every call of a `RawTable` function.
// This would definitely be an avenue worth exploring if people start complaining
// about the size of rust executables.
//
// Annotate exceedingly likely branches in `table::make_hash`
// and `search_hashed` to reduce instruction cache pressure
// and mispredictions once it becomes possible (blocked on issue #11092).
//
// Shrinking the table could simply reallocate in place after moving buckets
// to the first half.
//
// The growth algorithm (fragment of the Proof of Correctness)
// --------------------
//
// The growth algorithm is basically a fast path of the naive reinsertion-
// during-resize algorithm. Other paths should never be taken.
//
// Consider growing a robin hood hashtable of capacity n. Normally, we do this
// by allocating a new table of capacity `2n`, and then individually reinsert
// each element in the old table into the new one. This guarantees that the
// new table is a valid robin hood hashtable with all the desired statistical
// properties. Remark that the order we reinsert the elements in should not
// matter. For simplicity and efficiency, we will consider only linear
// reinsertions, which consist of reinserting all elements in the old table
// into the new one by increasing order of index. However we will not be
// starting our reinsertions from index 0 in general. If we start from index
// i, for the purpose of reinsertion we will consider all elements with real
// index j < i to have virtual index n + j.
//
// Our hash generation scheme consists of generating a 64-bit hash and
// truncating the most significant bits. When moving to the new table, we
// simply introduce a new bit to the front of the hash. Therefore, if an
// elements has ideal index i in the old table, it can have one of two ideal
// locations in the new table. If the new bit is 0, then the new ideal index
// is i. If the new bit is 1, then the new ideal index is n + i. Intuitively,
// we are producing two independent tables of size n, and for each element we
// independently choose which table to insert it into with equal probability.
// However the rather than wrapping around themselves on overflowing their
// indexes, the first table overflows into the first, and the first into the
// second. Visually, our new table will look something like:
//
// [yy_xxx_xxxx_xxx|xx_yyy_yyyy_yyy]
//
// Where x's are elements inserted into the first table, y's are elements
// inserted into the second, and _'s are empty sections. We now define a few
// key concepts that we will use later. Note that this is a very abstract
// perspective of the table. A real resized table would be at least half
// empty.
//
// Theorem: A linear robin hood reinsertion from the first ideal element
// produces identical results to a linear naive reinsertion from the same
// element.
//
// FIXME(Gankro, pczarn): review the proof and put it all in a separate README.md
//
// Adaptive early resizing
// ----------------------
// To protect against degenerate performance scenarios (including DOS attacks),
// the implementation includes an adaptive behavior that can resize the map
// early (before its capacity is exceeded) when suspiciously long probe sequences
// are encountered.
//
// With this algorithm in place it would be possible to turn a CPU attack into
// a memory attack due to the aggressive resizing. To prevent that the
// adaptive behavior only triggers when the map is at least half full.
// This reduces the effectiveness of the algorithm but also makes it completely safe.
//
// The previous safety measure also prevents degenerate interactions with
// really bad quality hash algorithms that can make normal inputs look like a
// DOS attack.
//
const DISPLACEMENT_THRESHOLD: usize = 128;
//
// The threshold of 128 is chosen to minimize the chance of exceeding it.
// In particular, we want that chance to be less than 10^-8 with a load of 90%.
// For displacement, the smallest constant that fits our needs is 90,
// so we round that up to 128.
//
// At a load factor of α, the odds of finding the target bucket after exactly n
// unsuccessful probes[1] are
//
// Pr_α{displacement = n} =
// (1 - α) / α * ∑_{k≥1} e^(-kα) * (kα)^(k+n) / (k + n)! * (1 - kα / (k + n + 1))
//
// We use this formula to find the probability of triggering the adaptive behavior
//
// Pr_0.909{displacement > 128} = 1.601 * 10^-11
//
// 1. Alfredo Viola (2005). Distributional analysis of Robin Hood linear probing
//    hashing with buckets.

/// A hash map implemented with linear probing and Robin Hood bucket stealing.
///
/// By default, `HashMap` uses a hashing algorithm selected to provide
/// resistance against HashDoS attacks. The algorithm is randomly seeded, and a
/// reasonable best-effort is made to generate this seed from a high quality,
/// secure source of randomness provided by the host without blocking the
/// program. Because of this, the randomness of the seed depends on the output
/// quality of the system's random number generator when the seed is created.
/// In particular, seeds generated when the system's entropy pool is abnormally
/// low such as during system boot may be of a lower quality.
///
/// The default hashing algorithm is currently SipHash 1-3, though this is
/// subject to change at any point in the future. While its performance is very
/// competitive for medium sized keys, other hashing algorithms will outperform
/// it for small keys such as integers as well as large keys such as long
/// strings, though those algorithms will typically *not* protect against
/// attacks such as HashDoS.
///
/// The hashing algorithm can be replaced on a per-`HashMap` basis using the
/// [`default`], [`with_hasher`], and [`with_capacity_and_hasher`] methods. Many
/// alternative algorithms are available on crates.io, such as the [`fnv`] crate.
///
/// It is required that the keys implement the [`Eq`] and [`Hash`] traits, although
/// this can frequently be achieved by using `#[derive(PartialEq, Eq, Hash)]`.
/// If you implement these yourself, it is important that the following
/// property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes must be equal.
///
/// It is a logic error for a key to be modified in such a way that the key's
/// hash, as determined by the [`Hash`] trait, or its equality, as determined by
/// the [`Eq`] trait, changes while it is in the map. This is normally only
/// possible through [`Cell`], [`RefCell`], global state, I/O, or unsafe code.
///
/// Relevant papers/articles:
///
/// 1. Pedro Celis. ["Robin Hood Hashing"](https://cs.uwaterloo.ca/research/tr/1986/CS-86-14.pdf)
/// 2. Emmanuel Goossaert. ["Robin Hood
///    hashing"](http://codecapsule.com/2013/11/11/robin-hood-hashing/)
/// 3. Emmanuel Goossaert. ["Robin Hood hashing: backward shift
///    deletion"](http://codecapsule.com/2013/11/17/robin-hood-hashing-backward-shift-deletion/)
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `HashMap<&str, &str>` in this example).
/// let mut book_reviews = HashMap::new();
///
/// // review some books.
/// book_reviews.insert("Adventures of Huckleberry Finn",    "My favorite book.");
/// book_reviews.insert("Grimms' Fairy Tales",               "Masterpiece.");
/// book_reviews.insert("Pride and Prejudice",               "Very enjoyable.");
/// book_reviews.insert("The Adventures of Sherlock Holmes", "Eye lyked it alot.");
///
/// // check for a specific one.
/// if !book_reviews.contains_key("Les Misérables") {
///     println!("We've got {} reviews, but Les Misérables ain't one.",
///              book_reviews.len());
/// }
///
/// // oops, this review has a lot of spelling mistakes, let's delete it.
/// book_reviews.remove("The Adventures of Sherlock Holmes");
///
/// // look up the values associated with some keys.
/// let to_find = ["Pride and Prejudice", "Alice's Adventure in Wonderland"];
/// for book in &to_find {
///     match book_reviews.get(book) {
///         Some(review) => println!("{}: {}", book, review),
///         None => println!("{} is unreviewed.", book)
///     }
/// }
///
/// // iterate over everything.
/// for (book, review) in &book_reviews {
///     println!("{}: \"{}\"", book, review);
/// }
/// ```
///
/// `HashMap` also implements an [`Entry API`](#method.entry), which allows
/// for more complex methods of getting, setting, updating and removing keys and
/// their values:
///
/// ```
/// use std::collections::HashMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `HashMap<&str, u8>` in this example).
/// let mut player_stats = HashMap::new();
///
/// fn random_stat_buff() -> u8 {
///     // could actually return some random value here - let's just return
///     // some fixed value for now
///     42
/// }
///
/// // insert a key only if it doesn't already exist
/// player_stats.entry("health").or_insert(100);
///
/// // insert a key using a function that provides a new value only if it
/// // doesn't already exist
/// player_stats.entry("defence").or_insert_with(random_stat_buff);
///
/// // update a key, guarding against the key possibly not being set
/// let stat = player_stats.entry("attack").or_insert(100);
/// *stat += random_stat_buff();
/// ```
///
/// The easiest way to use `HashMap` with a custom type as key is to derive [`Eq`] and [`Hash`].
/// We must also derive [`PartialEq`].
///
/// [`Eq`]: ../../std/cmp/trait.Eq.html
/// [`Hash`]: ../../std/hash/trait.Hash.html
/// [`PartialEq`]: ../../std/cmp/trait.PartialEq.html
/// [`RefCell`]: ../../std/cell/struct.RefCell.html
/// [`Cell`]: ../../std/cell/struct.Cell.html
/// [`default`]: #method.default
/// [`with_hasher`]: #method.with_hasher
/// [`with_capacity_and_hasher`]: #method.with_capacity_and_hasher
/// [`fnv`]: https://crates.io/crates/fnv
///
/// ```
/// use std::collections::HashMap;
///
/// #[derive(Hash, Eq, PartialEq, Debug)]
/// struct Viking {
///     name: String,
///     country: String,
/// }
///
/// impl Viking {
///     /// Create a new Viking.
///     fn new(name: &str, country: &str) -> Viking {
///         Viking { name: name.to_string(), country: country.to_string() }
///     }
/// }
///
/// // Use a HashMap to store the vikings' health points.
/// let mut vikings = HashMap::new();
///
/// vikings.insert(Viking::new("Einar", "Norway"), 25);
/// vikings.insert(Viking::new("Olaf", "Denmark"), 24);
/// vikings.insert(Viking::new("Harald", "Iceland"), 12);
///
/// // Use derived implementation to print the status of the vikings.
/// for (viking, health) in &vikings {
///     println!("{:?} has {} hp", viking, health);
/// }
/// ```
///
/// A `HashMap` with fixed list of elements can be initialized from an array:
///
/// ```
/// use std::collections::HashMap;
///
/// fn main() {
///     let timber_resources: HashMap<&str, i32> =
///     [("Norway", 100),
///      ("Denmark", 50),
///      ("Iceland", 10)]
///      .iter().cloned().collect();
///     // use the values stored in map
/// }
/// ```

#[derive(Clone)]
pub struct HashMap<V, S = FnvYoshiBuildHasher> {
    // All hashes are keyed on these values, to prevent hash collision attacks.
    hash_builder: S,

    table: RawTable<V>,

    resize_policy: DefaultResizePolicy,
}

/// Search for a pre-hashed key.
/// If you don't already know the hash, use search or search_mut instead
#[inline]
fn search_hashed<V, M, F>(table: M, hash: SafeHash, is_match: F) -> InternalEntry<V, M>
    where M: Deref<Target = RawTable<V>>,
          F: FnMut(&str) -> bool
{
    // This is the only function where capacity can be zero. To avoid
    // undefined behavior when Bucket::new gets the raw bucket in this
    // case, immediately return the appropriate search result.
    if table.capacity() == 0 {
        return InternalEntry::TableIsEmpty;
    }

    search_hashed_nonempty(table, hash, is_match)
}

/// Search for a pre-hashed key when the hash map is known to be non-empty.
#[inline]
fn search_hashed_nonempty<V, M, F>(table: M, hash: SafeHash, mut is_match: F)
    -> InternalEntry<V, M>
    where M: Deref<Target = RawTable<V>>,
          F: FnMut(&str) -> bool
{
    // Do not check the capacity as an extra branch could slow the lookup.

    let size = table.size();
    let mut probe = Bucket::new(table, hash);
    let mut displacement = 0;

    loop {
        let full = match probe.peek() {
            Empty(bucket) => {
                // Found a hole!
                return InternalEntry::Vacant {
                    hash,
                    elem: NoElem(bucket, displacement),
                };
            }
            Full(bucket) => bucket,
        };

        let probe_displacement = full.displacement();

        if probe_displacement < displacement {
            // Found a luckier bucket than me.
            // We can finish the search early if we hit any bucket
            // with a lower distance to initial bucket than we've probed.
            return InternalEntry::Vacant {
                hash,
                elem: NeqElem(full, probe_displacement),
            };
        }

        // If the hash doesn't match, it can't be this one..
        if hash == full.hash() {
            // If the key doesn't match, it can't be this one..
            if is_match(full.read().0) { // TODO compare strings here ... or a second hash :)
                return InternalEntry::Occupied { elem: full };
            }
        }
        displacement += 1;
        probe = full.next();
        debug_assert!(displacement <= size);
    }
}


fn add_and_get_text_position(key: &str, bytes: &mut Vec<u8>) -> (u32, u32) {
    let text_position = (bytes.len() as u32, key.as_bytes().len() as u32);
    bytes.extend(key.as_bytes());
    text_position
}

/// Perform robin hood bucket stealing at the given `bucket`. You must
/// also pass that bucket's displacement so we don't have to recalculate it.
///
/// `hash`, `key`, and `val` are the elements to "robin hood" into the hashtable.
fn robin_hood<'a, V: 'a>(mut bucket: FullBucketMut<'a, V>,
                                mut displacement: usize,
                                mut hash: SafeHash,
                                key: &str,
                                mut val: V)
                                -> FullBucketMut<'a, V> {
    let size = bucket.table().size();
    let raw_capacity = bucket.table().capacity();
    // There can be at most `size - dib` buckets to displace, because
    // in the worst case, there are `size` elements and we already are
    // `displacement` buckets away from the initial one.
    let idx_end = (bucket.index() + size - bucket.displacement()) % raw_capacity;
    // Save the *starting point*.
    
    let mut text_position = add_and_get_text_position(&key, &mut bucket.table_mut().raw_text_data );
    let mut bucket = bucket.stash();
    loop {
        let (old_hash, old_key, old_val) = bucket.replace(hash, text_position, val);
        hash = old_hash;
        text_position = old_key;
        val = old_val;

        loop {
            displacement += 1;
            let probe = bucket.next();
            debug_assert!(probe.index() != idx_end);

            let full_bucket = match probe.peek() {
                Empty(bucket) => {
                    // Found a hole!
                    let bucket = bucket.put(hash, &&text_position, val);
                    // Now that it's stolen, just read the value's pointer
                    // right out of the table! Go back to the *starting point*.
                    //
                    // This use of `into_table` is misleading. It turns the
                    // bucket, which is a FullBucket on top of a
                    // FullBucketMut, into just one FullBucketMut. The "table"
                    // refers to the inner FullBucketMut in this context.
                    return bucket.into_table();
                }
                Full(bucket) => bucket,
            };

            let probe_displacement = full_bucket.displacement();

            bucket = full_bucket;

            // Robin hood! Steal the spot.
            if probe_displacement < displacement {
                displacement = probe_displacement;
                break;
            }
        }
    }
}

impl<V, S> HashMap<V, S>
    where S: BuildHasher
{
    #[inline]
    pub fn make_hash(&self, q: &str) -> SafeHash
    {
        SafeHash::new_u32(hasher::fnv32a_yoshimitsu_hasher(q.as_bytes()))
        // table::make_hash(&self.hash_builder, x)
    }

    /// Search for a key, yielding the index if it's found in the hashtable.
    /// If you already have the hash for the key lying around, or if you need an
    /// InternalEntry, use search_hashed or search_hashed_nonempty.
    #[inline]
    fn search<'a>(&'a self, q: &str)
        -> Option<FullBucket<V, &'a RawTable<V>>>
    {
        if self.is_empty() {
            return None;
        }
        let hash = self.make_hash(q);
        search_hashed_nonempty(&self.table, hash, |k| q.eq(k))
            .into_occupied_bucket()
    }

    #[inline]
    fn search_mut<'a>(&'a mut self, q: &str)
        -> Option<FullBucket<V, &'a mut RawTable<V>>>
    {
        if self.is_empty() {
            return None;
        }

        let hash = self.make_hash(q);
        search_hashed_nonempty(&mut self.table, hash, |k| q.eq(k))
            .into_occupied_bucket()
    }

    // The caller should ensure that invariants by Robin Hood Hashing hold
    // and that there's space in the underlying table.
    fn insert_hashed_ordered_text_pos(&mut self, hash: SafeHash, text_position: (u32, u32), v: V) {
        let mut buckets = Bucket::new(&mut self.table, hash);
        let start_index = buckets.index();

        loop {
            // We don't need to compare hashes for value swap.
            // Not even DIBs for Robin Hood.
            buckets = match buckets.peek() {
                Empty(empty) => {
                    empty.put(hash, &text_position, v);
                    return;
                }
                Full(b) => b.into_bucket(),
            };
            buckets.next();
            debug_assert!(buckets.index() != start_index);
        }
    }
}

impl<V> HashMap<V, FnvYoshiBuildHasher> {
    /// Creates an empty `HashMap`.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, i32> = HashMap::new();
    /// ```
    #[inline]
    pub fn new() -> HashMap<V, FnvYoshiBuildHasher> {
        Default::default()
    }

    /// Creates an empty `HashMap` with the specified capacity.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, i32> = HashMap::with_capacity(10);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> HashMap<V, FnvYoshiBuildHasher> {
        HashMap::with_capacity_and_hasher(capacity, Default::default())
    }
}

impl<V, S> HashMap<V, S>
    where S: BuildHasher
{
    /// Creates an empty `HashMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// The created map has the default initial capacity.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    #[inline]
    pub fn with_hasher(hash_builder: S) -> HashMap<V, S> {
        HashMap {
            hash_builder,
            resize_policy: DefaultResizePolicy::new(),
            table: RawTable::new(0),
        }
    }

    /// Creates an empty `HashMap` with the specified capacity, using `hash_builder`
    /// to hash the keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> HashMap<V, S> {
        let resize_policy = DefaultResizePolicy::new();
        let raw_cap = resize_policy.raw_capacity(capacity);
        HashMap {
            hash_builder,
            resize_policy,
            table: RawTable::new(raw_cap),
        }
    }

    /// Returns a reference to the map's [`BuildHasher`].
    ///
    /// [`BuildHasher`]: ../../std/hash/trait.BuildHasher.html
    ///
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// This number is a lower bound; the `HashMap<V>` might be able to hold
    /// more, but is guaranteed to be able to hold at least this many.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let map: HashMap<i32, i32> = HashMap::with_capacity(100);
    /// assert!(map.capacity() >= 100);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.resize_policy.capacity(self.raw_capacity())
    }

    /// Returns the hash map's raw capacity.
    #[inline]
    fn raw_capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `HashMap`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows [`usize`].
    ///
    /// [`usize`]: ../../std/primitive.usize.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, i32> = HashMap::new();
    /// map.reserve(10);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        match self.reserve_internal(additional, Infallible) {
            Err(CollectionAllocErr::CapacityOverflow) => panic!("capacity overflow"),
            Err(CollectionAllocErr::AllocErr) => unreachable!(),
            Ok(()) => { /* yay */ }
        }
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted
    /// in the given `HashMap<V>`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_reserve)]
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, isize> = HashMap::new();
    /// map.try_reserve(10).expect("why is the test harness OOMing on 10 bytes?");
    /// ```
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), CollectionAllocErr> {
        self.reserve_internal(additional, Fallible)
    }

    fn reserve_internal(&mut self, additional: usize, fallibility: Fallibility)
        -> Result<(), CollectionAllocErr> {

        let remaining = self.capacity() - self.len(); // this can't overflow
        if remaining < additional {
            let min_cap = self.len()
                .checked_add(additional)
                .ok_or(CollectionAllocErr::CapacityOverflow)?;
            let raw_cap = self.resize_policy.try_raw_capacity(min_cap)?;
            self.try_resize(raw_cap, fallibility)?;
        } else if self.table.tag() && remaining <= self.len() {
            // Probe sequence is too long and table is half full,
            // resize early to reduce probing length.
            let new_capacity = self.table.capacity() * 2;
            self.try_resize(new_capacity, fallibility)?;
        }
        Ok(())
    }

    /// Resizes the internal vectors to a new capacity. It's your
    /// responsibility to:
    ///   1) Ensure `new_raw_cap` is enough for all the elements, accounting
    ///      for the load factor.
    ///   2) Ensure `new_raw_cap` is a power of two or zero.
    #[inline(never)]
    #[cold]
    fn try_resize(&mut self, new_raw_cap: usize, fallibility: Fallibility,) -> Result<(), CollectionAllocErr> {
        assert!(self.table.size() <= new_raw_cap);
        assert!(new_raw_cap.is_power_of_two() || new_raw_cap == 0);

        let mut old_table = replace(
            &mut self.table,
            match fallibility {
                Infallible => RawTable::new(new_raw_cap),
                Fallible => RawTable::try_new(new_raw_cap)?,
            }
        );
        let old_size = old_table.size();

        mem::swap(&mut self.table.raw_text_data, &mut old_table.raw_text_data);

        if old_table.size() == 0 {
            return Ok(());
        }

        let mut bucket = Bucket::head_bucket(&mut old_table);

        // This is how the buckets might be laid out in memory:
        // ($ marks an initialized bucket)
        //  ________________
        // |$$$_$$$$$$_$$$$$|
        //
        // But we've skipped the entire initial cluster of buckets
        // and will continue iteration in this order:
        //  ________________
        //     |$$$$$$_$$$$$
        //                  ^ wrap around once end is reached
        //  ________________
        //  $$$_____________|
        //    ^ exit once table.size == 0
        loop {
            bucket = match bucket.peek() {
                Full(bucket) => {
                    let h = bucket.hash();
                    let (b, k, v) = bucket.take();
                    self.insert_hashed_ordered_text_pos(h, k, v);
                    if b.table().size() == 0 {
                        break;
                    }
                    b.into_bucket()
                }
                Empty(b) => b.into_bucket(),
            };
            bucket.next();
        }

        assert_eq!(self.table.size(), old_size);
        Ok(())
    }

    /// Insert a pre-hashed key-value pair, without first checking
    /// that there's enough room in the buckets. Returns a reference to the
    /// newly insert value.
    ///
    /// If the key already exists, the hashtable will be returned untouched
    /// and a reference to the existing element will be returned.
    fn insert_hashed_nocheck(&mut self, hash: SafeHash, k: &str, v: V) {
        let entry = search_hashed(&mut self.table, hash, |key| key == k);
        insert_if_empty_and_get_mut(entry, k, || v);
    }

    /// Insert a pre-hashed key-value pair, without first checking
    /// that there's enough room in the buckets. Returns a reference to the
    /// newly insert value.
    ///
    /// If the key already exists, the hashtable will be returned untouched
    /// and a reference to the existing element will be returned.
    pub fn get_or_insert<F>(&mut self, k: &str, constructor: F) -> &mut V
    where
        F: FnOnce() -> V
    {
        self.reserve(1);
        let hash = self.make_hash(&k);
        let entry = search_hashed(&mut self.table, hash, |key| key == k);
        insert_if_empty_and_get_mut(entry, k, constructor)
    }
//TODO: -- enable iterators
    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    // pub fn keys(&self) -> Keys<V> {
    //     Keys { inner: self.iter() }
    // }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    // pub fn values(&self) -> Values<V> {
    //     Values { inner: self.iter() }
    // }

    // pub fn values_mut(&mut self) -> ValuesMut<V> {
    //     ValuesMut { inner: self.iter_mut() }
    // }

    // pub fn iter(&self) -> Iter<V> {
    //     Iter { inner: self.table.iter() }
    // }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// The iterator element type is `(&'a String, &'a mut V)`.
    ///
    // pub fn iter_mut(&mut self) -> IterMut<V> {
    //     IterMut { inner: self.table.iter_mut() }
    // }

//TODO -- enable Iterators

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    // pub fn entry(&mut self, key: &str) -> Entry<V> {
    //     // Gotta resize now.
    //     self.reserve(1);
    //     let hash = self.make_hash(key);
    //     search_hashed(&mut self.table, hash, |q| q.eq(key))
    //         .into_entry(&key).expect("unreachable")
    // }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.table.size()
    }

    /// Returns true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[inline]
    pub fn get(&self, k: &str) -> Option<&V>
    {
        self.search(k).map(|bucket| bucket.into_refs().1)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(map_get_key_value)]
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get_key_value(&1), Some((&1, &"a")));
    /// assert_eq!(map.get_key_value(&2), None);
    /// ```
    // TODO ENABLE
    // pub fn get_key_value(&self, k: &str) -> Option<(&String, &V)>
    // {
    //     self.search(k).map(|bucket| bucket.into_refs())
    // }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    pub fn contains_key(&self, k: &str) -> bool
    {
        self.search(k).is_some()
    }

    pub fn contains_hashed_key(&self, q: &str, hash:SafeHash) -> bool
    {
        if self.is_empty() {
            return false;
        }

        search_hashed_nonempty(&self.table, hash, |k| q.eq(k))
            .into_occupied_bucket().is_some()

        // self.search(k).is_some()
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    // pub fn contains_key_hashed(&self, hash: SafeHash, q: &str) -> bool
    //     where Q: Hash + Eq
    // {

    //     if self.is_empty() {
    //         return false;
    //     }

    //     match search_hashed_nonempty(&self.table, hash, |k| q.eq(k.borrow())) {
    //         InternalEntry::Occupied { elem: _ } => false,
    //         InternalEntry::Vacant { hash: _, elem: _ } => true,
    //         InternalEntry::TableIsEmpty => false,
    //     }

    //     // == InternalEntry::Vacant
    //     // self.search(k).is_some()
    // }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: ../../std/cmp/trait.Eq.html
    /// [`Hash`]: ../../std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    pub fn get_mut(&mut self, k: &str) -> Option<&mut V>
    {
        self.search_mut(k).map(|bucket| bucket.into_mut_refs().1)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [module-level documentation]: index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    pub fn insert(&mut self, k: String, v: V) {
        let hash = self.make_hash(&k);
        self.reserve(1);
        self.insert_hashed_nocheck(hash, &k, v);
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [module-level documentation]: index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    pub fn insert_hashed(&mut self, hash: SafeHash, k: String, v: V) {
        self.reserve(1);
        self.insert_hashed_nocheck(hash, &k, v);
    }

}

//TODO Enable
// impl<V, S> PartialEq for HashMap<V, S>
//     where V: PartialEq,
//           S: BuildHasher
// {
//     fn eq(&self, other: &HashMap<V, S>) -> bool {
//         if self.len() != other.len() {
//             return false;
//         }

//         self.iter().all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
//     }
// }

//TODO Enable
// impl<V, S> Eq for HashMap<V, S>
//     where V: Eq,
//           S: BuildHasher
// {
// }

//TODO Enable
// impl<V, S> Debug for HashMap<V, S>
//     where V: Debug,
//           S: BuildHasher
// {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_map().entries(self.iter()).finish()
//     }
// }

impl<V, S> Default for HashMap<V, S>
    where S: BuildHasher + Default
{
    /// Creates an empty `HashMap<V, S>`, with the `Default` value for the hasher.
    fn default() -> HashMap<V, S> {
        HashMap::with_hasher(Default::default())
    }
}

impl<'a, V, S> Index<&'a String> for HashMap<V, S>
    where S: BuildHasher
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the `HashMap`.
    #[inline]
    fn index(&self, key: &String) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

// ------ TODO: Enable Iterators

/// An iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`iter`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`iter`]: struct.HashMap.html#method.iter
/// [`HashMap`]: struct.HashMap.html
// pub struct Iter<'a, V: 'a> {
//     inner: table::Iter<'a, V>,
// }

// // FIXME(#26925) Remove in favor of `#[derive(Clone)]`
// impl<'a, V> Clone for Iter<'a, V> {
//     fn clone(&self) -> Iter<'a, V> {
//         Iter { inner: self.inner.clone() }
//     }
// }

// impl<'a, V: Debug> fmt::Debug for Iter<'a, V> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_list()
//             .entries(self.clone())
//             .finish()
//     }
// }

// /// A mutable iterator over the entries of a `HashMap`.
// ///
// /// This `struct` is created by the [`iter_mut`] method on [`HashMap`]. See its
// /// documentation for more.
// ///
// /// [`iter_mut`]: struct.HashMap.html#method.iter_mut
// /// [`HashMap`]: struct.HashMap.html
// pub struct IterMut<'a, V: 'a> {
//     inner: table::IterMut<'a, V>,
// }

// /// An owning iterator over the entries of a `HashMap`.
// ///
// /// This `struct` is created by the [`into_iter`] method on [`HashMap`][`HashMap`]
// /// (provided by the `IntoIterator` trait). See its documentation for more.
// ///
// /// [`into_iter`]: struct.HashMap.html#method.into_iter
// /// [`HashMap`]: struct.HashMap.html
// pub struct IntoIter<V> {
//     pub inner: table::IntoIter<V>,
// }

// /// An iterator over the keys of a `HashMap`.
// ///
// /// This `struct` is created by the [`keys`] method on [`HashMap`]. See its
// /// documentation for more.
// ///
// /// [`keys`]: struct.HashMap.html#method.keys
// /// [`HashMap`]: struct.HashMap.html
// pub struct Keys<'a, V: 'a> {
//     inner: Iter<'a, V>,
// }

// // FIXME(#26925) Remove in favor of `#[derive(Clone)]`
// impl<'a, V> Clone for Keys<'a, V> {
//     fn clone(&self) -> Keys<'a, V> {
//         Keys { inner: self.inner.clone() }
//     }
// }

// impl<'a, V> fmt::Debug for Keys<'a, V> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_list()
//             .entries(self.clone())
//             .finish()
//     }
// }

// /// An iterator over the values of a `HashMap`.
// ///
// /// This `struct` is created by the [`values`] method on [`HashMap`]. See its
// /// documentation for more.
// ///
// /// [`values`]: struct.HashMap.html#method.values
// /// [`HashMap`]: struct.HashMap.html
// // pub struct Values<'a, V: 'a> {
// //     inner: Iter<'a, V>,
// // }

// // FIXME(#26925) Remove in favor of `#[derive(Clone)]`
// impl<'a, V> Clone for Values<'a, V> {
//     fn clone(&self) -> Values<'a, V> {
//         Values { inner: self.inner.clone() }
//     }
// }

// impl<'a, V: Debug> fmt::Debug for Values<'a, V> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_list()
//             .entries(self.clone())
//             .finish()
//     }
// }

/// A mutable iterator over the values of a `HashMap`.
///
/// This `struct` is created by the [`values_mut`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`values_mut`]: struct.HashMap.html#method.values_mut
/// [`HashMap`]: struct.HashMap.html
// pub struct ValuesMut<'a, V: 'a> {
//     inner: IterMut<'a, V>,
// }

// ---- TODO: Enable Iterators

enum InternalEntry<V, M> {
    Occupied { elem: FullBucket<V, M> },
    Vacant {
        hash: SafeHash,
        elem: VacantEntryState<V, M>,
    },
    TableIsEmpty,
}

impl<V, M> InternalEntry<V, M> {
    #[inline]
    fn into_occupied_bucket(self) -> Option<FullBucket<V, M>> {
        match self {
            InternalEntry::Occupied { elem } => Some(elem),
            _ => None,
        }
    }
}

fn insert_if_empty_and_get_mut<'a, V, F>(entry: InternalEntry<V, &'a mut RawTable<V>>, key: &str, constructor: F) -> &'a mut V 
where
    F: FnOnce() -> V
{

    match entry {
        InternalEntry::Occupied { elem } => {
            elem.into_mut_refs().1
        }
        InternalEntry::Vacant { hash, elem } => {
            let value = constructor();
            let b = match elem {
                NeqElem(mut bucket, disp) => {
                    if disp >= DISPLACEMENT_THRESHOLD {
                        bucket.table_mut().set_tag(true);
                    }
                    robin_hood(bucket, disp, hash, key, value)
                },
                NoElem(mut bucket, disp) => {
                    if disp >= DISPLACEMENT_THRESHOLD {
                        bucket.table_mut().set_tag(true);
                    }
                    let mut text_position = add_and_get_text_position(key, &mut bucket.table_mut().raw_text_data );
                    bucket.put(hash, &text_position, value)
                },
            };
            b.into_mut_refs().1
        }
        InternalEntry::TableIsEmpty => unreachable!(),
    }


}


// impl<'a, V> InternalEntry<V, &'a mut RawTable<V>> {
//     #[inline]
//     fn into_entry(self, key: &str) -> Option<Entry<'a, V>> {
//         match self {
//             InternalEntry::Occupied { elem } => {
//                 Some(Occupied(OccupiedEntry {
//                     //key: Some(key),
//                     elem,
//                 }))
//             }
//             InternalEntry::Vacant { hash, elem } => {
//                 Some(Vacant(VacantEntry {
//                     hash,
//                     key: key.to_string(),
//                     elem,
//                 }))
//             }
//             InternalEntry::TableIsEmpty => None,
//         }
//     }
// }

// /// A view into a single entry in a map, which may either be vacant or occupied.
// ///
// /// This `enum` is constructed from the [`entry`] method on [`HashMap`].
// ///
// /// [`HashMap`]: struct.HashMap.html
// /// [`entry`]: struct.HashMap.html#method.entry
// pub enum Entry<'a, V: 'a> {
//     /// An occupied entry.
//     Occupied(OccupiedEntry<'a, V>),

//     /// A vacant entry.
//     Vacant(VacantEntry<'a, V>),
// }

// impl<'a, V: 'a + Debug> Debug for Entry<'a, V> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         match *self {
//             Vacant(ref v) => {
//                 f.debug_tuple("Entry")
//                     .field(v)
//                     .finish()
//             }
//             Occupied(ref o) => {
//                 f.debug_tuple("Entry")
//                     .field(o)
//                     .finish()
//             }
//         }
//     }
// }

// /// A view into an occupied entry in a `HashMap`.
// /// It is part of the [`Entry`] enum.
// ///
// /// [`Entry`]: enum.Entry.html
// pub struct OccupiedEntry<'a, V: 'a> {
//     // key: Option<String>,
//     elem: FullBucket<V, &'a mut RawTable<V>>,
// }

// impl<'a, V: 'a + Debug> Debug for OccupiedEntry<'a, V> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_struct("OccupiedEntry")
//             .field("key", self.key())
//             .field("value", self.get())
//             .finish()
//     }
// }

// /// A view into a vacant entry in a `HashMap`.
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: enum.Entry.html
pub struct VacantEntry<'a, V: 'a> {
    hash: SafeHash,
    key: String,
    elem: VacantEntryState<V, &'a mut RawTable<V>>,
}

impl<'a, V: 'a> Debug for VacantEntry<'a, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("VacantEntry")
            .field(self.key())
            .finish()
    }
}

/// Possible states of a VacantEntry.
enum VacantEntryState<V, M> {
    /// The index is occupied, but the key to insert has precedence,
    /// and will kick the current one out on insertion.
    NeqElem(FullBucket<V, M>, usize),
    /// The index is genuinely vacant.
    NoElem(EmptyBucket<V, M>, usize),
}



// TODO: ------ Enable Iterators


// impl<'a, V, S> IntoIterator for &'a HashMap<V, S>
//     where S: BuildHasher
// {
//     type Item = (&'a String, &'a V);
//     type IntoIter = Iter<'a, V>;

//     fn into_iter(self) -> Iter<'a, V> {
//         self.iter()
//     }
// }

// impl<'a, V, S> IntoIterator for &'a mut HashMap<V, S>
//     where S: BuildHasher
// {
//     type Item = (&'a String, &'a mut V);
//     type IntoIter = IterMut<'a, V>;

//     fn into_iter(self) -> IterMut<'a, V> {
//         self.iter_mut()
//     }
// }

// impl<V, S> IntoIterator for HashMap<V, S>
//     where S: BuildHasher
// {
//     type Item = (String, V);
//     type IntoIter = IntoIter<V>;

//     /// Creates a consuming iterator, that is, one that moves each key-value
//     /// pair out of the map in arbitrary order. The map cannot be used after
//     /// calling this.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use std::collections::HashMap;
//     ///
//     /// let mut map = HashMap::new();
//     /// map.insert("a", 1);
//     /// map.insert("b", 2);
//     /// map.insert("c", 3);
//     ///
//     /// // Not possible with .iter()
//     /// let vec: Vec<(&str, i32)> = map.into_iter().collect();
//     /// ```
//     fn into_iter(self) -> IntoIter<V> {
//         IntoIter { inner: self.table.into_iter() }
//     }
// }

// impl<'a, V> Iterator for Iter<'a, V> {
//     type Item = (&'a String, &'a V);

//     #[inline]
//     fn next(&mut self) -> Option<(&'a String, &'a V)> {
//         self.inner.next()
//     }
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.inner.size_hint()
//     }
// }
// impl<'a, V> ExactSizeIterator for Iter<'a, V> {
//     #[inline]
//     fn len(&self) -> usize {
//         self.inner.len()
//     }
// }

// impl<'a, V> FusedIterator for Iter<'a, V> {}

// impl<'a, V> Iterator for IterMut<'a, V> {
//     type Item = (&'a String, &'a mut V);

//     #[inline]
//     fn next(&mut self) -> Option<(&'a String, &'a mut V)> {
//         self.inner.next()
//     }
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.inner.size_hint()
//     }
// }
// impl<'a, V> ExactSizeIterator for IterMut<'a, V> {
//     #[inline]
//     fn len(&self) -> usize {
//         self.inner.len()
//     }
// }
// impl<'a, V> FusedIterator for IterMut<'a, V> {}

// impl<'a, V> fmt::Debug for IterMut<'a, V>
//     where V: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_list()
//             .entries(self.inner.iter())
//             .finish()
//     }
// }

// impl<V> Iterator for IntoIter<V> {
//     type Item = (String, V);

//     #[inline]
//     fn next(&mut self) -> Option<(String, V)> {
//         self.inner.next().map(|(_, k, v)| (k, v))
//     }
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.inner.size_hint()
//     }
// }
// impl<V> ExactSizeIterator for IntoIter<V> {
//     #[inline]
//     fn len(&self) -> usize {
//         self.inner.len()
//     }
// }
// impl<V> FusedIterator for IntoIter<V> {}

// impl<V: Debug> fmt::Debug for IntoIter<V> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_list()
//             .entries(self.inner.iter())
//             .finish()
//     }
// }

// impl<'a, V> Iterator for Keys<'a, V> {
//     type Item = &'a String;

//     #[inline]
//     fn next(&mut self) -> Option<(&'a String)> {
//         self.inner.next().map(|(k, _)| k)
//     }
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.inner.size_hint()
//     }
// }
// impl<'a, V> ExactSizeIterator for Keys<'a, V> {
//     #[inline]
//     fn len(&self) -> usize {
//         self.inner.len()
//     }
// }
// impl<'a, V> FusedIterator for Keys<'a, V> {}

// impl<'a, V> Iterator for Values<'a, V> {
//     type Item = &'a V;

//     #[inline]
//     fn next(&mut self) -> Option<(&'a V)> {
//         self.inner.next().map(|(_, v)| v)
//     }
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.inner.size_hint()
//     }
// }
// impl<'a, V> ExactSizeIterator for Values<'a, V> {
//     #[inline]
//     fn len(&self) -> usize {
//         self.inner.len()
//     }
// }
// impl<'a, V> FusedIterator for Values<'a, V> {}

// impl<'a, V> Iterator for ValuesMut<'a, V> {
//     type Item = &'a mut V;

//     #[inline]
//     fn next(&mut self) -> Option<(&'a mut V)> {
//         self.inner.next().map(|(_, v)| v)
//     }
//     #[inline]
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         self.inner.size_hint()
//     }
// }
// impl<'a, V> ExactSizeIterator for ValuesMut<'a, V> {
//     #[inline]
//     fn len(&self) -> usize {
//         self.inner.len()
//     }
// }
// impl<'a, V> FusedIterator for ValuesMut<'a, V> {}

// impl<'a, V> fmt::Debug for ValuesMut<'a, V>
//     where V: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_list()
//             .entries(self.inner.inner.iter())
//             .finish()
//     }
// }

// TODO: ------ Enable Iterators

// impl<'a, V> Entry<'a, V> {
//     pub fn or_insert(self, default: V) -> &'a mut V {
//         match self {
//             Occupied(entry) => entry.into_mut(),
//             Vacant(entry) => entry.insert(default),
//         }
//     }

//     pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
//         match self {
//             Occupied(entry) => entry.into_mut(),
//             Vacant(entry) => entry.insert(default()),
//         }
//     }

//     pub fn key(&self) -> &str {
//         match *self {
//             Occupied(ref entry) => entry.key(),
//             Vacant(ref entry) => entry.key(),
//         }
//     }

//     pub fn and_modify<F>(self, f: F) -> Self
//         where F: FnOnce(&mut V)
//     {
//         match self {
//             Occupied(mut entry) => {
//                 f(entry.get_mut());
//                 Occupied(entry)
//             },
//             Vacant(entry) => Vacant(entry),
//         }
//     }

// }

// impl<'a, V: Default> Entry<'a, V> {
//     pub fn or_default(self) -> &'a mut V {
//         match self {
//             Occupied(entry) => entry.into_mut(),
//             Vacant(entry) => entry.insert(Default::default()),
//         }
//     }
// }

// impl<'a, V> OccupiedEntry<'a, V> {
//     pub fn key(&self) -> &String {
//         self.elem.read().0
//     }
//     pub fn get(&self) -> &V {
//         self.elem.read().1
//     }
//     pub fn get_mut(&mut self) -> &mut V {
//         self.elem.read_mut().1
//     }
//     pub fn into_mut(self) -> &'a mut V {
//         self.elem.into_mut_refs().1
//     }

//     /// Sets the value of the entry, and returns the entry's old value.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use std::collections::HashMap;
//     /// use std::collections::hash_map::Entry;
//     ///
//     /// let mut map: HashMap<&str, u32> = HashMap::new();
//     /// map.entry("poneyland").or_insert(12);
//     ///
//     /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
//     ///     assert_eq!(o.insert(15), 12);
//     /// }
//     ///
//     /// assert_eq!(map["poneyland"], 15);
//     /// ```
//     pub fn insert(&mut self, mut value: V) -> V {
//         let old_value = self.get_mut();
//         mem::swap(&mut value, old_value);
//         value
//     }


// }

//TODO key shouldn't be store here as String
impl<'a, V: 'a> VacantEntry<'a, V> {
    pub fn key(&self) -> &String {
        &self.key
    }
    pub fn into_key(self) -> String {
        self.key
    }
    pub fn insert(self, value: V) -> &'a mut V {
        let b = match self.elem {
            NeqElem(mut bucket, disp) => {
                if disp >= DISPLACEMENT_THRESHOLD {
                    bucket.table_mut().set_tag(true);
                }
                robin_hood(bucket, disp, self.hash, &self.key, value)
            },
            NoElem(mut bucket, disp) => {
                if disp >= DISPLACEMENT_THRESHOLD {
                    bucket.table_mut().set_tag(true);
                }
                let mut text_position = add_and_get_text_position(&self.key, &mut bucket.table_mut().raw_text_data );
                bucket.put(self.hash, &text_position, value)
            },
        };
        b.into_mut_refs().1
    }
}

impl<V, S> FromIterator<(String, V)> for HashMap<V, S>
    where S: BuildHasher + Default
{
    fn from_iter<T: IntoIterator<Item = (String, V)>>(iter: T) -> HashMap<V, S> {
        let mut map = HashMap::with_hasher(Default::default());
        map.extend(iter);
        map
    }
}

impl<V, S> Extend<(String, V)> for HashMap<V, S>
    where S: BuildHasher
{
    fn extend<T: IntoIterator<Item = (String, V)>>(&mut self, iter: T) {
        // Keys may be already present or show multiple times in the iterator.
        // Reserve the entire hint lower bound if the map is empty.
        // Otherwise reserve half the hint (rounded up), so the map
        // will only resize twice in the worst case.
        let iter = iter.into_iter();
        let reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) / 2
        };
        self.reserve(reserve);
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

// impl<'a, V, S> Extend<(&'a String, &'a V)> for HashMap<V, S>
//     where V: Copy,
//           S: BuildHasher
// {
//     fn extend<T: IntoIterator<Item = (&'a String, &'a V)>>(&mut self, iter: T) {
//         self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
//     }
// }


#[cfg(test)]
mod test_map {
    use super::HashMap;
    use std::cell::RefCell;
    use std::usize;

    #[test]
    fn insert_and_get_get() {
        let mut map: HashMap<u32> = HashMap::new();

        map.insert("3".to_string(), 4);
        map.insert("1".to_string(), 2);

        assert_eq!(map.get("3"), Some(&4));
        assert_eq!(map.get("2"), None);
    }

    // #[test]
    // fn test_own_hasher() {
    //     let mut map = HashMap::new();
    //     let empty: HashMap<i32> = HashMap::new();

    //     let text = "lucky number".to_string();


    //     let hash = hasher::fnv32a_yoshimitsu_hasher(text.as_bytes());

    //     // map.insert_hashed(text, hash, 0);
    //     map.insert("3".to_string(), 4);
    //     map.insert("1".to_string(), 2);

    //     let map_str = format!("{:?}", map);

    //     assert!(map_str == "{1: 2, 3: 4}" ||
    //             map_str == "{3: 4, 1: 2}");
    //     assert_eq!(format!("{:?}", empty), "{}");
    // }

    thread_local! { static DROP_VECTOR: RefCell<Vec<i32>> = RefCell::new(Vec::new()) }

    #[derive(Hash, PartialEq, Eq)]
    struct Droppable {
        k: usize,
    }

    impl Droppable {
        fn new(k: usize) -> Droppable {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[k] += 1;
            });

            Droppable { k: k }
        }
    }

    impl Drop for Droppable {
        fn drop(&mut self) {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[self.k] -= 1;
            });
        }
    }

    impl Clone for Droppable {
        fn clone(&self) -> Droppable {
            Droppable::new(self.k)
        }
    }

    // #[test]
    // fn test_eq() {
    //     let mut m1 = HashMap::new();
    //     m1.insert(1, 2);
    //     m1.insert(2, 3);
    //     m1.insert(3, 4);

    //     let mut m2 = HashMap::new();
    //     m2.insert(1, 2);
    //     m2.insert(2, 3);

    //     assert!(m1 != m2);

    //     m2.insert(3, 4);

    //     assert_eq!(m1, m2);
    // }

    // #[test]
    // fn test_show() {
    //     let mut map = HashMap::new();
    //     let empty: HashMap<i32, i32> = HashMap::new();

    //     map.insert(1, 2);
    //     map.insert(3, 4);

    //     let map_str = format!("{:?}", map);

    //     assert!(map_str == "{1: 2, 3: 4}" ||
    //             map_str == "{3: 4, 1: 2}");
    //     assert_eq!(format!("{:?}", empty), "{}");
    // }

    // #[test]
    // fn test_expand() {
    //     let mut m = HashMap::new();

    //     assert_eq!(m.len(), 0);
    //     assert!(m.is_empty());

    //     let mut i = 0;
    //     let old_raw_cap = m.raw_capacity();
    //     while old_raw_cap == m.raw_capacity() {
    //         m.insert(i, i);
    //         i += 1;
    //     }

    //     assert_eq!(m.len(), i);
    //     assert!(!m.is_empty());
    // }

    // #[test]
    // fn test_behavior_resize_policy() {
    //     let mut m = HashMap::new();

    //     assert_eq!(m.len(), 0);
    //     assert_eq!(m.raw_capacity(), 0);
    //     assert!(m.is_empty());

    //     m.insert(0, 0);
    //     m.remove(&0);
    //     assert!(m.is_empty());
    //     let initial_raw_cap = m.raw_capacity();
    //     m.reserve(initial_raw_cap);
    //     let raw_cap = m.raw_capacity();

    //     assert_eq!(raw_cap, initial_raw_cap * 2);

    //     let mut i = 0;
    //     for _ in 0..raw_cap * 3 / 4 {
    //         m.insert(i, i);
    //         i += 1;
    //     }
    //     // three quarters full

    //     assert_eq!(m.len(), i);
    //     assert_eq!(m.raw_capacity(), raw_cap);

    //     for _ in 0..raw_cap / 4 {
    //         m.insert(i, i);
    //         i += 1;
    //     }
    //     // half full

    //     let new_raw_cap = m.raw_capacity();
    //     assert_eq!(new_raw_cap, raw_cap * 2);

    //     for _ in 0..raw_cap / 2 - 1 {
    //         i -= 1;
    //         m.remove(&i);
    //         assert_eq!(m.raw_capacity(), new_raw_cap);
    //     }
    // }

    // #[test]
    // fn test_from_iter() {
    //     let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    //     let map: HashMap<_, _> = xs.iter().cloned().collect();

    //     for &(k, v) in &xs {
    //         assert_eq!(map.get(&k), Some(&v));
    //     }
    // }

    // #[test]
    // fn test_size_hint() {
    //     let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    //     let map: HashMap<_, _> = xs.iter().cloned().collect();

    //     let mut iter = map.iter();

    //     for _ in iter.by_ref().take(3) {}

    //     assert_eq!(iter.size_hint(), (3, Some(3)));
    // }

    // #[test]
    // fn test_iter_len() {
    //     let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    //     let map: HashMap<_, _> = xs.iter().cloned().collect();

    //     let mut iter = map.iter();

    //     for _ in iter.by_ref().take(3) {}

    //     assert_eq!(iter.len(), 3);
    // }

    // #[test]
    // fn test_mut_size_hint() {
    //     let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    //     let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    //     let mut iter = map.iter_mut();

    //     for _ in iter.by_ref().take(3) {}

    //     assert_eq!(iter.size_hint(), (3, Some(3)));
    // }

    // #[test]
    // fn test_iter_mut_len() {
    //     let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

    //     let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    //     let mut iter = map.iter_mut();

    //     for _ in iter.by_ref().take(3) {}

    //     assert_eq!(iter.len(), 3);
    // }

    // #[test]
    // fn test_index() {
    //     let mut map = HashMap::new();

    //     map.insert(1, 2);
    //     map.insert(2, 1);
    //     map.insert(3, 4);

    //     assert_eq!(map[&2], 1);
    // }

    // #[test]
    // #[should_panic]
    // fn test_index_nonexistent() {
    //     let mut map = HashMap::new();

    //     map.insert(1, 2);
    //     map.insert(2, 1);
    //     map.insert(3, 4);

    //     map[&4];
    // }

    // #[test]
    // fn test_entry() {
    //     let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    //     let mut map: HashMap<_, _> = xs.iter().cloned().collect();

    //     // Existing key (insert)
    //     match map.entry(1) {
    //         Vacant(_) => unreachable!(),
    //         Occupied(mut view) => {
    //             assert_eq!(view.get(), &10);
    //             assert_eq!(view.insert(100), 10);
    //         }
    //     }
    //     assert_eq!(map.get(&1).unwrap(), &100);
    //     assert_eq!(map.len(), 6);


    //     // Existing key (update)
    //     match map.entry(2) {
    //         Vacant(_) => unreachable!(),
    //         Occupied(mut view) => {
    //             let v = view.get_mut();
    //             let new_v = (*v) * 10;
    //             *v = new_v;
    //         }
    //     }
    //     assert_eq!(map.get(&2).unwrap(), &200);
    //     assert_eq!(map.len(), 6);

    //     // Existing key (take)
    //     match map.entry(3) {
    //         Vacant(_) => unreachable!(),
    //         Occupied(view) => {
    //             assert_eq!(view.remove(), 30);
    //         }
    //     }
    //     assert_eq!(map.get(&3), None);
    //     assert_eq!(map.len(), 5);


    //     // Inexistent key (insert)
    //     match map.entry(10) {
    //         Occupied(_) => unreachable!(),
    //         Vacant(view) => {
    //             assert_eq!(*view.insert(1000), 1000);
    //         }
    //     }
    //     assert_eq!(map.get(&10).unwrap(), &1000);
    //     assert_eq!(map.len(), 6);
    // }

    // #[test]
    // fn test_extend_ref() {
    //     let mut a = HashMap::new();
    //     a.insert(1, "one");
    //     let mut b = HashMap::new();
    //     b.insert(2, "two");
    //     b.insert(3, "three");

    //     a.extend(&b);

    //     assert_eq!(a.len(), 3);
    //     assert_eq!(a[&1], "one");
    //     assert_eq!(a[&2], "two");
    //     assert_eq!(a[&3], "three");
    // }

    // #[test]
    // fn test_capacity_not_less_than_len() {
    //     let mut a = HashMap::new();
    //     let mut item = 0;

    //     for _ in 0..116 {
    //         a.insert(item, 0);
    //         item += 1;
    //     }

    //     assert!(a.capacity() > a.len());

    //     let free = a.capacity() - a.len();
    //     for _ in 0..free {
    //         a.insert(item, 0);
    //         item += 1;
    //     }

    //     assert_eq!(a.len(), a.capacity());

    //     // Insert at capacity should cause allocation.
    //     a.insert(item, 0);
    //     assert!(a.capacity() > a.len());
    // }

    // #[test]
    // fn test_occupied_entry_key() {
    //     let mut a = HashMap::new();
    //     let key = "hello there";
    //     let value = "value goes here";
    //     assert!(a.is_empty());
    //     a.insert(key.clone(), value.clone());
    //     assert_eq!(a.len(), 1);
    //     assert_eq!(a[key], value);

    //     match a.entry(key.clone()) {
    //         Vacant(_) => panic!(),
    //         Occupied(e) => assert_eq!(key, *e.key()),
    //     }
    //     assert_eq!(a.len(), 1);
    //     assert_eq!(a[key], value);
    // }

    // #[test]
    // fn test_vacant_entry_key() {
    //     let mut a = HashMap::new();
    //     let key = "hello there";
    //     let value = "value goes here";

    //     assert!(a.is_empty());
    //     match a.entry(key.clone()) {
    //         Occupied(_) => panic!(),
    //         Vacant(e) => {
    //             assert_eq!(key, *e.key());
    //             e.insert(value.clone());
    //         }
    //     }
    //     assert_eq!(a.len(), 1);
    //     assert_eq!(a[key], value);
    // }

    // #[test]
    // fn test_retain() {
    //     let mut map: HashMap<i32, i32> = (0..100).map(|x|(x, x*10)).collect();

    //     map.retain(|&k, _| k % 2 == 0);
    //     assert_eq!(map.len(), 50);
    //     assert_eq!(map[&2], 20);
    //     assert_eq!(map[&4], 40);
    //     assert_eq!(map[&6], 60);
    // }

    // #[test]
    // fn test_adaptive() {
    //     const TEST_LEN: usize = 5000;
    //     // by cloning we get maps with the same hasher seed
    //     let mut first = HashMap::new();
    //     let mut second = first.clone();
    //     first.extend((0..TEST_LEN).map(|i| (i, i)));
    //     second.extend((TEST_LEN..TEST_LEN * 2).map(|i| (i, i)));

    //     for (&k, &v) in &second {
    //         let prev_cap = first.capacity();
    //         let expect_grow = first.len() == prev_cap;
    //         first.insert(k, v);
    //         if !expect_grow && first.capacity() != prev_cap {
    //             return;
    //         }
    //     }
    //     panic!("Adaptive early resize failed");
    // }

    // #[test]
    // fn test_try_reserve() {

    //     let mut empty_bytes: HashMap<u8,u8> = HashMap::new();

    //     const MAX_USIZE: usize = usize::MAX;

    //     // HashMap and RawTables use complicated size calculations
    //     // hashes_size is sizeof(HashUint) * capacity;
    //     // pairs_size is sizeof((K. V)) * capacity;
    //     // alignment_hashes_size is 8
    //     // alignment_pairs size is 4
    //     let size_of_multiplier = (size_of::<usize>() + size_of::<(u8, u8)>()).next_power_of_two();
    //     // The following formula is used to calculate the new capacity
    //     let max_no_ovf = ((MAX_USIZE / 11) * 10) / size_of_multiplier - 1;

    //     if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_USIZE) {
    //     } else { panic!("usize::MAX should trigger an overflow!"); }

    //     if size_of::<usize>() < 8 {
    //         if let Err(CapacityOverflow) = empty_bytes.try_reserve(max_no_ovf) {
    //         } else { panic!("isize::MAX + 1 should trigger a CapacityOverflow!") }
    //     } else {
    //         if let Err(AllocErr) = empty_bytes.try_reserve(max_no_ovf) {
    //         } else { panic!("isize::MAX + 1 should trigger an OOM!") }
    //     }
    // }

}
