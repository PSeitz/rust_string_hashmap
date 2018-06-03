extern crate term_hashmap;
extern crate fnv;
extern crate cpuprofiler;
use std::fs::File;
use std::io::prelude::*;

#[macro_use]
extern crate measure_time;
fn main() {
    let mut file = File::open("1342-0.txt").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    test_yoshi(&contents);
    test_fnv(&contents);
    test_stacker(&contents);
    test_custom(&contents);
    test_custom_entry(&contents);
    test_yoshi(&contents);
    test_fnv(&contents);
    // test_normal(&contents);
    test_stacker(&contents);
    test_custom(&contents);
    test_custom_entry(&contents);
    test_yoshi(&contents);
    test_fnv(&contents);
    // test_normal(&contents);
    test_stacker(&contents);
    test_custom(&contents);
    test_custom_entry(&contents);
    test_yoshi(&contents);
    test_fnv(&contents);
    // test_normal(&contents);
    test_stacker(&contents);
    test_custom(&contents);
    test_custom_entry(&contents);

    // PROFILER.lock().unwrap().start("./my-prof.profile2").unwrap();
    // for _ in 0..1000 {
    //     test_fnv(&contents);
    // }
    // PROFILER.lock().unwrap().stop().unwrap();
    // PROFILER.lock().unwrap().start("./my-prof.profile").unwrap();
    // for _ in 0..1000 {
    //     test_custom_entry(&contents);
    // }
    // PROFILER.lock().unwrap().stop().unwrap();
}

fn test_yoshi(text: &str)-> bool {
    let mut mappo = term_hashmap::hasher::FnvYoshiHashMap::default();
    print_time!("yoshi");
    // mappo.reserve(100_000);
    for line in text.split_whitespace() {
        if !mappo.contains_key(line) {
            mappo.insert(line.to_owned(), 0);
        }
    }
    mappo.contains_key("test")
    // println!("{:?}", mappo.len());
}

fn test_fnv(text: &str)-> bool {
    let mut mappo = fnv::FnvHashMap::default();
    print_time!("FnvHashMap");
    // mappo.reserve(100_000);
    for line in text.split_whitespace() {
        if !mappo.contains_key(line) {
            mappo.insert(line.to_string(), 1);
        }else{
            *mappo.get_mut(line).unwrap() += 1;
        }
    }
    mappo.contains_key("test")
    // println!("{:?}", mappo.len());
}

fn test_normal(text: &str)-> bool {
    let mut mappo:std::collections::HashMap<String, u32> = std::collections::HashMap::default();
    print_time!("HashMap");
    // mappo.reserve(100_000);
    for line in text.split_whitespace() {
        if !mappo.contains_key(line) {
            mappo.insert(line.to_string(), 1);
        }else{
            *mappo.get_mut(line).unwrap() += 1;
        }
    }
    mappo.contains_key("test")
    // println!("{:?}", mappo.len());
}

fn test_custom(text: &str) -> bool {
    let mut mappo:term_hashmap::HashMap<u32> = term_hashmap::HashMap::default();
    // let mut all_the_bytes:Vec<u8> = Vec::with_capacity(5_000_000);
    print_time!("term_hashmap");
    // mappo.reserve(100_000);
    for line in text.split_whitespace() {
        let hash = mappo.make_hash(line);
        if !mappo.contains_hashed_key(line, hash) {
            mappo.insert_hashed(hash, line.to_owned(), 10);
        }
        // all_the_bytes.extend(line.as_bytes());
    }
    mappo.contains_key("test")
    // println!("{:?}", mappo.len());
}


use cpuprofiler::PROFILER;


fn test_custom_entry(text: &str) -> bool {
    let mut mappo:term_hashmap::HashMap<u32> = term_hashmap::HashMap::default();
    // let mut all_the_bytes:Vec<u8> = Vec::with_capacity(5_000_000);
    print_time!("term_hashmap entry api");
    // mappo.reserve(100_000);
    for line in text.split_whitespace() {

        
        let stat = mappo.get_or_insert(line, ||0);
        *stat += 1;
        // let stat = mappo.entry(line).or_insert(0);
        // *stat += 1;

        // mappo.entry(line)
        // let hash = mappo.make_hash(line);
        // if !mappo.contains_hashed_key(line, hash) {
        //     mappo.insert_hashed(hash, line.to_owned(), 10);
        // }
        // all_the_bytes.extend(line.as_bytes());
    }
    mappo.contains_key("test")
    // println!("{:?}", mappo.len());
}

fn test_set(text: &str) {
    let mut mappo = fnv::FnvHashSet::default();
    print_time!("FnvHashSet");
    // mappo.reserve(100_000);
    for line in text.split_whitespace() {
        mappo.insert(line.to_owned());
    }
    println!("{:?}", mappo.len());
}

use term_hashmap::stacker::split_memory;
use term_hashmap::stacker::Heap;
use term_hashmap::stacker::TermHashMap;

fn test_stacker(text: &str) -> u32  {
    print_time!("test_stacker");
    let heap_size_in_bytes_per_thread  = 2_550_000;
    let (heap_size, table_size) = split_memory(heap_size_in_bytes_per_thread);
    let heap = Heap::with_capacity(heap_size);
    let mut mappo: TermHashMap = TermHashMap::new(table_size, &heap);
    // let mut mappo = fnv2::FnvHashMap::default();
    for line in text.split_whitespace() {
        let el = mappo.get_or_create::<_, u32>(line);
        *el.1 += 1;
        // if !mappo.contains_key(line) {
        //     mappo.insert(line.to_owned(), 0);
        // }
    }
    mappo.get_or_create::<_, u32>("test").0
    // println!("{:?}", mappo.len());
    // println!("oke");
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
