use std::mem::transmute;

const ONLY_HIGH_BIT_U8: u8 = 0b_1000_0000;

#[inline]
pub(crate) fn set_high_bit_u8(input: u8) -> u8 {
    input | ONLY_HIGH_BIT_U8
}

#[inline]
pub(crate) fn unset_high_bit_u8(input: u8) -> u8 {
    input << 1 >> 1
}

#[inline]
pub(crate) fn is_high_bit_set(input: u8) -> bool {
    input & ONLY_HIGH_BIT_U8 != 0
}

/// Only split for performance reasons
#[inline]
pub(crate) fn push_n_set_big(val: u32, data: &mut [u8]) {
    let mut el_u64: u64 = val as u64;
    el_u64 <<= 4;
    let bytes: [u8; 8] = unsafe { transmute(el_u64) };
    data[4] = bytes[4];
}

#[inline]
pub(crate) fn push_n_set(last_block: bool, el: &mut u32, pos: &mut u8, data: &mut [u8]) {
    if *pos > 0 {
        *el <<= 1;
    }
    if last_block {
        let bytes: [u8; 4] = unsafe { transmute(*el) };
        data[*pos as usize] = bytes[*pos as usize];
    } else {
        let bytes: [u8; 4] = unsafe { transmute(*el) };
        data[*pos as usize] = set_high_bit_u8(bytes[*pos as usize]);
    }
    *pos += 1;
}

#[inline]
pub(crate) fn encode_num(val: u32) -> ([u8; 8], u8) {
    let mut el = val;

    let mut data = [0, 0, 0, 0, 0, 0, 0, 0];
    let mut pos: u8 = 0;

    if val < 1 << 7 {
        //128
        push_n_set(true, &mut el, &mut pos, &mut data);
    } else if val < 1 << 14 {
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(true, &mut el, &mut pos, &mut data);
    } else if val < 1 << 21 {
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(true, &mut el, &mut pos, &mut data);
    } else if val < 1 << 28 {
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(true, &mut el, &mut pos, &mut data);
    } else {
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set(false, &mut el, &mut pos, &mut data);
        push_n_set_big(val, &mut data);
        pos += 1;
    }
    (data, pos)
}


#[derive(Debug, Clone)]
pub(crate) struct VintArrayIterator<'a> {
    pub(crate) data: &'a [u8],
    /// the current offset in the slice
    pub(crate) pos: usize,
}

impl<'a> VintArrayIterator<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        VintArrayIterator { data: data, pos: 0 }
    }

    #[inline]
    fn decode_u8(&self, pos: usize) -> (u8, bool) {
        unsafe {
            let el = *self.data.get_unchecked(pos);
            if is_high_bit_set(el) {
                (unset_high_bit_u8(el), true)
            } else {
                (el, false)
            }
        }
    }

    #[inline]
    fn get_apply_bits(&self, pos: usize, offset: usize, val: &mut u32) -> bool {
        let (val_u8, has_more) = self.decode_u8(pos);

        let mut bytes: [u8; 4] = [0, 0, 0, 0];
        bytes[offset] = val_u8;
        let mut add_val: u32 = unsafe { transmute(bytes) };
        add_val >>= offset;
        *val |= add_val;

        has_more
    }
}
impl<'a> Iterator for VintArrayIterator<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.pos == self.data.len() {
            None
        } else {
            let (val_u8, has_more) = self.decode_u8(self.pos);
            self.pos += 1;
            let mut val = val_u8 as u32;
            if has_more {
                let has_more = self.get_apply_bits(self.pos, 1, &mut val);
                self.pos += 1;
                if has_more {
                    let has_more = self.get_apply_bits(self.pos, 2, &mut val);
                    self.pos += 1;
                    if has_more {
                        let has_more = self.get_apply_bits(self.pos, 3, &mut val);
                        self.pos += 1;
                        if has_more {
                            let el = unsafe { *self.data.get_unchecked(self.pos) };
                            let bytes: [u8; 4] = [0, 0, 0, el];
                            let mut add_val: u32 = unsafe { transmute(bytes) };
                            add_val <<= 4;
                            val |= add_val as u32;
                            self.pos += 1;
                        }
                    }
                }
            }
            Some(val)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.data.len() - self.pos / 2,
            Some(self.data.len() - self.pos),
        )
    }
}

