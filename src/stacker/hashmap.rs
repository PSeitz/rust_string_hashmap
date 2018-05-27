use super::heap::{BytesRef, Heap, HeapAllocable};
// use postings::TermId;
use std::iter;
use std::mem;
use std::slice;

type TermId = u32;




mod murmurhash2 {

    const SEED: u32 = 3_242_157_231u32;
    const SEED2: u32 = 0xD8AFFD71u32;
    const M: u32 = 0x5bd1_e995;

    use std::ptr::copy_nonoverlapping;
    #[inline]
    fn rotl32(x: u32, r: u8) -> u32
    {
      (x << r) | (x >> (32 - r))
    }

    #[inline]
    fn read_u32_p(p: *const u8) -> u32{
        // let mut bytes: [u8; 4] = [0, 0, 0, 0];
        // bytes.copy_from_slice(&data[..4]);
        // unsafe { transmute(bytes) }

        let mut out: u32 = 0;
        unsafe {
            copy_nonoverlapping(p, &mut out as *mut u32 as *mut u8, 4);
        }
        out
    }

    #[inline]
    fn read_u16_p(p: *const u8) -> u16{
        // let mut bytes: [u8; 4] = [0, 0, 0, 0];
        // bytes.copy_from_slice(&data[..4]);
        // unsafe { transmute(bytes) }

        let mut out: u16 = 0;
        unsafe {
            copy_nonoverlapping(p, &mut out as *mut u16 as *mut u8, 2);
        }
        out
    }


    #[inline]
    pub fn fnv32a_yoshimitsu_triad(bytes: &[u8]) -> u32
    {
        let mut len: u32 = bytes.len() as u32;
        let mut p: *const u8 = bytes.as_ptr();

        const PRIME:u32 = 709607;
        let mut hash32_a: u32 = SEED2 ^ 2166136261;
        let mut hash32_b: u32 = 2166136261 + len;
        let mut hash32_c: u32 = 2166136261;

        while len >= 24{
            hash32_a = (hash32_a ^ (rotl32(read_u32_p(p), 5)  ^ read_u32_p(unsafe{p.offset(4)})))  .wrapping_mul(PRIME);
            hash32_b = (hash32_b ^ (rotl32(read_u32_p(unsafe{p.offset(8)}), 5)  ^ read_u32_p(unsafe{p.offset(12)}))) .wrapping_mul(PRIME);
            hash32_c = (hash32_c ^ (rotl32(read_u32_p(unsafe{p.offset(16)}), 5) ^ read_u32_p(unsafe{p.offset(20)}))) .wrapping_mul(PRIME);
            len-= 24;
            // pos+=24;
            p = unsafe{p.offset(24)};
        }

        if p == bytes.as_ptr() {
            hash32_a = (hash32_a ^ rotl32(hash32_c, 5)) .wrapping_mul(PRIME);
        }
        //Cases 0. .31
        if (len & 16) != 0{
            hash32_a = (hash32_a ^ (rotl32(read_u32_p(p), 5) ^ read_u32_p(unsafe{p.offset(4)}))) .wrapping_mul(PRIME);
            hash32_b = (hash32_b ^ (rotl32(read_u32_p(unsafe{p.offset(8)}), 5) ^ read_u32_p(unsafe{p.offset(12)}))) .wrapping_mul(PRIME);
            p = unsafe{p.offset(16)};
        }
        //Cases 0. .15
        if (len & 8) != 0 {
            hash32_a = (hash32_a ^ read_u32_p(p)) .wrapping_mul(PRIME);
            hash32_b = (hash32_b ^ read_u32_p(unsafe{p.offset(4)})) .wrapping_mul(PRIME);
            p = unsafe{p.offset(8)};
        }
        //Cases:0. .7
        if (len & 4) != 0 {
            hash32_a = (hash32_a ^ read_u16_p(p) as u32) .wrapping_mul(PRIME);
            hash32_b = (hash32_b ^ read_u16_p(unsafe{p.offset(2)}) as u32) .wrapping_mul(PRIME);
            p = unsafe{p.offset(4)};
        }
        //Cases:0. .3
        if (len & 2) != 0 {
            hash32_a = (hash32_a ^ read_u16_p(p) as u32) .wrapping_mul(PRIME);
            p = unsafe{p.offset(2)};
        }
        if (len & 1) != 0 {
            hash32_a = (hash32_a ^ unsafe{*p} as u32) .wrapping_mul(PRIME);
        }
        hash32_a = (hash32_a ^ rotl32(hash32_b, 5)) .wrapping_mul(PRIME);
        hash32_a ^ (hash32_a >> 16)

    }

    #[inline(always)]
    pub fn murmurhash2(key: &[u8]) -> u32 {
        let mut key_ptr: *const u32 = key.as_ptr() as *const u32;
        let len = key.len() as u32;
        let mut h: u32 = SEED ^ len;

        let num_blocks = len >> 2;
        for _ in 0..num_blocks {
            let mut k: u32 = unsafe { *key_ptr }; // ok because of num_blocks definition
            k = k.wrapping_mul(M);
            k ^= k >> 24;
            k = k.wrapping_mul(M);
            h = h.wrapping_mul(M);
            h ^= k;
            key_ptr = key_ptr.wrapping_offset(1);
        }

        // Handle the last few bytes of the input array
        let remaining: &[u8] = &key[key.len() & !3..];
        match remaining.len() {
            3 => {
                h ^= u32::from(remaining[2]) << 16;
                h ^= u32::from(remaining[1]) << 8;
                h ^= u32::from(remaining[0]);
                h = h.wrapping_mul(M);
            }
            2 => {
                h ^= u32::from(remaining[1]) << 8;
                h ^= u32::from(remaining[0]);
                h = h.wrapping_mul(M);
            }
            1 => {
                h ^= u32::from(remaining[0]);
                h = h.wrapping_mul(M);
            }
            _ => {}
        }
        h ^= h >> 13;
        h = h.wrapping_mul(M);
        h ^ (h >> 15)
    }
}

/// Split the thread memory budget into
/// - the heap size
/// - the hash table "table" itself.
///
/// Returns (the heap size in bytes, the hash table size in number of bits)
pub fn split_memory(per_thread_memory_budget: usize) -> (usize, usize) {
    let table_size_limit: usize = per_thread_memory_budget / 3;
    let compute_table_size = |num_bits: usize| (1 << num_bits) * mem::size_of::<KeyValue>();
    let table_num_bits: usize = (1..)
        .into_iter()
        .take_while(|num_bits: &usize| compute_table_size(*num_bits) < table_size_limit)
        .last()
        .expect(&format!(
            "Per thread memory is too small: {}",
            per_thread_memory_budget
        ));
    let table_size = compute_table_size(table_num_bits);
    let heap_size = per_thread_memory_budget - table_size;
    (heap_size, table_num_bits)
}

/// `KeyValue` is the item stored in the hash table.
/// The key is actually a `BytesRef` object stored in an external heap.
/// The `value_addr` also points to an address in the heap.
///
/// The key and the value are actually stored contiguously.
/// For this reason, the (start, stop) information is actually redundant
/// and can be simplified in the future
#[derive(Copy, Clone, Default)]
struct KeyValue {
    key_value_addr: BytesRef,
    hash: u32,
}

impl KeyValue {
    fn is_empty(&self) -> bool {
        self.key_value_addr.is_null()
    }
}

/// Customized `HashMap` with string keys
///
/// This `HashMap` takes String as keys. Keys are
/// stored in a user defined heap.
///
/// The quirky API has the benefit of avoiding
/// the computation of the hash of the key twice,
/// or copying the key as long as there is no insert.
///
pub struct TermHashMap<'a> {
    table: Box<[KeyValue]>,
    heap: &'a Heap,
    mask: usize,
    occupied: Vec<usize>,
}

struct QuadraticProbing {
    hash: usize,
    i: usize,
    mask: usize,
}

impl QuadraticProbing {
    fn compute(hash: usize, mask: usize) -> QuadraticProbing {
        QuadraticProbing { hash, i: 0, mask }
    }

    #[inline]
    fn next_probe(&mut self) -> usize {
        self.i += 1;
        (self.hash + self.i * self.i) & self.mask
    }
}

pub struct Iter<'a: 'b, 'b> {
    hashmap: &'b TermHashMap<'a>,
    inner: slice::Iter<'a, usize>,
}

impl<'a, 'b> Iterator for Iter<'a, 'b> {
    type Item = (&'b [u8], u32, TermId);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().cloned().map(move |bucket: usize| {
            let kv = self.hashmap.table[bucket];
            let (key, offset): (&'b [u8], u32) = self.hashmap.get_key_value(kv.key_value_addr);
            (key, offset, bucket as TermId)
        })
    }
}

impl<'a> TermHashMap<'a> {
    pub fn new(num_bucket_power_of_2: usize, heap: &'a Heap) -> TermHashMap<'a> {
        let table_size = 1 << num_bucket_power_of_2;
        let table: Vec<KeyValue> = iter::repeat(KeyValue::default()).take(table_size).collect();
        TermHashMap {
            table: table.into_boxed_slice(),
            heap,
            mask: table_size - 1,
            occupied: Vec::with_capacity(table_size / 2),
        }
    }

    fn probe(&self, hash: u32) -> QuadraticProbing {
        QuadraticProbing::compute(hash as usize, self.mask)
    }

    pub fn is_saturated(&self) -> bool {
        self.table.len() < self.occupied.len() * 3
    }

    #[inline(never)]
    fn get_key_value(&self, bytes_ref: BytesRef) -> (&[u8], u32) {
        let key_bytes: &[u8] = self.heap.get_slice(bytes_ref);
        let expull_addr: u32 = bytes_ref.addr() + 2 + key_bytes.len() as u32;
        (key_bytes, expull_addr)
    }

    pub fn set_bucket(&mut self, hash: u32, key_value_addr: BytesRef, bucket: usize) {
        self.occupied.push(bucket);
        self.table[bucket] = KeyValue {
            key_value_addr,
            hash,
        };
    }

    pub fn iter<'b: 'a>(&'b self) -> Iter<'a, 'b> {
        Iter {
            inner: self.occupied.iter(),
            hashmap: &self,
        }
    }

    pub fn get_or_create<S: AsRef<[u8]>, V: HeapAllocable>(
        &mut self,
        key: S,
    ) -> (TermId, &mut V) {
        let key_bytes: &[u8] = key.as_ref();
        let hash = murmurhash2::murmurhash2(key.as_ref());
        let mut probe = self.probe(hash);
        loop {
            let bucket = probe.next_probe();
            let kv: KeyValue = self.table[bucket];
            if kv.is_empty() {
                let key_bytes_ref = self.heap.allocate_and_set(key_bytes);
                let (addr, val): (u32, &mut V) = self.heap.allocate_object();
                assert_eq!(addr, key_bytes_ref.addr() + 2 + key_bytes.len() as u32);
                self.set_bucket(hash, key_bytes_ref, bucket);
                return (bucket as TermId, val);
            } else if kv.hash == hash {
                let (stored_key, expull_addr): (&[u8], u32) = self.get_key_value(kv.key_value_addr);
                if stored_key == key_bytes {
                    return (
                        bucket as TermId,
                        self.heap.get_mut_ref(expull_addr),
                    );
                }
            }
        }
    }
}

#[cfg(all(test, feature = "unstable"))]
mod bench {
    use super::murmurhash2::murmurhash2;
    use test::Bencher;

    #[bench]
    fn bench_murmurhash2(b: &mut Bencher) {
        let keys: [&'static str; 3] = ["wer qwe qwe qwe ", "werbq weqweqwe2 ", "weraq weqweqwe3 "];
        b.iter(|| {
            let mut s = 0;
            for &key in &keys {
                s ^= murmurhash2(key.as_bytes());
            }
            s
        });
    }

    #[bench]
    fn bench_yoshi(b: &mut Bencher) {
        let keys: [&'static str; 3] = ["wer qwe qwe qwe ", "werbq weqweqwe2 ", "weraq weqweqwe3 "];
        b.iter(|| {
            let mut s = 0;
            for &key in &keys {
                s ^= fnv32a_yoshimitsu_triad(key.as_bytes());
            }
            s
        });
    }
}

#[cfg(test)]
mod tests {

    use super::super::heap::{Heap, HeapAllocable};
    use super::murmurhash2::murmurhash2;
    use super::split_memory;
    use super::*;
    use std::collections::HashSet;

    struct TestValue {
        val: u32,
        _addr: u32,
    }

    impl HeapAllocable for TestValue {
        fn with_addr(addr: u32) -> TestValue {
            TestValue {
                val: 0u32,
                _addr: addr,
            }
        }
    }

    #[test]
    fn test_hashmap_size() {
        assert_eq!(split_memory(100_000), (67232, 12));
        assert_eq!(split_memory(1_000_000), (737856, 15));
        assert_eq!(split_memory(10_000_000), (7902848, 18));
    }

    #[test]
    fn test_hash_map() {
        let heap = Heap::with_capacity(2_000_000);
        let mut hash_map: TermHashMap = TermHashMap::new(18, &heap);
        {
            let v: &mut TestValue = hash_map.get_or_create("abc").1;
            assert_eq!(v.val, 0u32);
            v.val = 3u32;
        }
        {
            let v: &mut TestValue = hash_map.get_or_create("abcd").1;
            assert_eq!(v.val, 0u32);
            v.val = 4u32;
        }
        {
            let v: &mut TestValue = hash_map.get_or_create("abc").1;
            assert_eq!(v.val, 3u32);
        }
        {
            let v: &mut TestValue = hash_map.get_or_create("abcd").1;
            assert_eq!(v.val, 4u32);
        }
        let mut iter_values = hash_map.iter();
        {
            let (_, addr, _) = iter_values.next().unwrap();
            let val: &TestValue = heap.get_ref(addr);
            assert_eq!(val.val, 3u32);
        }
        {
            let (_, addr, _) = iter_values.next().unwrap();
            let val: &TestValue = heap.get_ref(addr);
            assert_eq!(val.val, 4u32);
        }
        assert!(iter_values.next().is_none());
    }

    #[test]
    fn test_murmur() {
        let s1 = "abcdef";
        let s2 = "abcdeg";
        for i in 0..5 {
            assert_eq!(
                murmurhash2(&s1[i..5].as_bytes()),
                murmurhash2(&s2[i..5].as_bytes())
            );
        }
    }

    #[test]
    fn test_murmur_against_reference_impl() {
        assert_eq!(murmurhash2("".as_bytes()), 3632506080);
        assert_eq!(murmurhash2("a".as_bytes()), 455683869);
        assert_eq!(murmurhash2("ab".as_bytes()), 2448092234);
        assert_eq!(murmurhash2("abc".as_bytes()), 2066295634);
        assert_eq!(murmurhash2("abcd".as_bytes()), 2588571162);
        assert_eq!(murmurhash2("abcde".as_bytes()), 2988696942);
        assert_eq!(murmurhash2("abcdefghijklmnop".as_bytes()), 2350868870);
    }

    #[test]
    fn test_murmur_collisions() {
        let mut set: HashSet<u32> = HashSet::default();
        for i in 0..10_000 {
            let s = format!("hash{}", i);
            let hash = murmurhash2(s.as_bytes());
            set.insert(hash);
        }
        assert_eq!(set.len(), 10_000);
    }

}
