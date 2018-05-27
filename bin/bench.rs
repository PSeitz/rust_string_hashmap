extern crate term_hashmap;
extern crate fnv;
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[macro_use]
extern crate measure_time;
fn main() {
    // let stream1 = ;
    let lines:Vec<String> = io::BufReader::new(File::open("words.txt").unwrap()).lines().map(|line|line.unwrap()).collect();
    println!("YOP");
    test_yoshi(&lines);
    test_yoshi(&lines);
    test_fnv(&lines);
    test_fnv(&lines);
    test_set(&lines);
    test_set(&lines);
    test_stacker(&lines);
    test_stacker(&lines);
    test_custom(&lines);
    test_custom(&lines);
    test_yoshi(&lines);
    test_yoshi(&lines);
    test_fnv(&lines);
    test_fnv(&lines);
    test_set(&lines);
    test_set(&lines);
    test_stacker(&lines);
    test_stacker(&lines);
    test_custom(&lines);
    test_custom(&lines);
}

fn test_yoshi(lines: &Vec<String>) {
    let mut mappo = term_hashmap::hasher::FnvYoshiHashMap::default();
    mappo.reserve(100_000);
    print_time!("yoshi");
    for line in lines {
        if !mappo.contains_key(line) {
            mappo.insert(line.to_owned(), 0);
        }
    }
    println!("{:?}", mappo.len());
}

fn test_fnv(lines: &Vec<String>) {
    let mut mappo = fnv::FnvHashMap::default();
    mappo.reserve(100_000);
    print_time!("FnvHashMap");
    for line in lines {
        if !mappo.contains_key(line) {
            mappo.insert(line.to_owned(), 0);
        }
    }
    println!("{:?}", mappo.len());
}

fn test_custom(lines: &Vec<String>) {
    let mut mappo:term_hashmap::HashMap<u32> = term_hashmap::HashMap::default();
    // let mut all_the_bytes:Vec<u8> = Vec::with_capacity(5_000_000);
    mappo.reserve(100_000);
    print_time!("term_hashmap");
    for line in lines {
        let hash = mappo.make_hash(line);
        if !mappo.contains_hashed_key(line, hash) {
            mappo.insert_hashed(hash, line.to_owned(), 10);
        }
        // all_the_bytes.extend(line.as_bytes());
    }
    println!("{:?}", mappo.len());
}

fn test_set(lines: &Vec<String>) {
    let mut mappo = fnv::FnvHashSet::default();
    mappo.reserve(100_000);
    print_time!("FnvHashSet");
    for line in lines {
        mappo.insert(line.to_owned());
    }
    println!("{:?}", mappo.len());
}

use term_hashmap::stacker::split_memory;
use term_hashmap::stacker::Heap;
use term_hashmap::stacker::TermHashMap;

fn test_stacker(lines: &Vec<String>) {
    let heap_size_in_bytes_per_thread  = 8_000_000;
    let (heap_size, table_size) = split_memory(heap_size_in_bytes_per_thread);
    let heap = Heap::with_capacity(heap_size);
    let mut mappo: TermHashMap = TermHashMap::new(table_size, &heap);
    // let mut mappo = fnv2::FnvHashMap::default();
    print_time!("test_stacker");
    for line in lines {
        let el = mappo.get_or_create::<_, u32>(line);
        *el.1 = 10;
        // if !mappo.contains_key(line) {
        //     mappo.insert(line.to_owned(), 0);
        // }
    }
    // println!("{:?}", mappo.len());
    println!("oke");
}



// #[test]
// fn test_map() {
//     use std::collections;
//     let heap = Heap::with_capacity(30_000_000);
//     let mut hashmap: TermHashMap = TermHashMap::new(10, &heap);
//     {
//         let el = hashmap.get_or_create::<_, u32>("nice");
//         println!("WAAAAA");
//         println!("TERMID {:?}", el.0);
//         println!("WAS DAS {:?}", el.1);
//         *el.1 = 10;

//     }
//     let el = hashmap.get_or_create::<_, u32>("heiß");

// }
