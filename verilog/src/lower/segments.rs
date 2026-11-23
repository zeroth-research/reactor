//! The value of a signal during symbolic execution: a list of wire segments.

use crate::Wire;
use std::collections::BTreeSet;

/// Bits `width` bits of a wire starting at bit `lo`, or `None` for bits
/// of an `always_comb` signal that are not assigned yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Seg {
    pub(super) width: usize,
    pub(super) src: Option<(Wire, usize)>,
}

/// The value of a signal of the block so far: segments from the high bits
/// down, with adjacent segments of the same wire (or both unassigned)
/// joined, so that equal values have equal segments.
pub(super) type Value = Vec<Seg>;

/// The value of all bits of `w`.
pub(super) fn whole(w: Wire) -> Value {
    vec![Seg {
        width: w.dtype().width(),
        src: Some((w, 0)),
    }]
}

/// The wire of a value that is all bits of one wire.
pub(super) fn as_wire(v: &Value) -> Option<Wire> {
    match v.as_slice() {
        [
            Seg {
                width,
                src: Some((w, 0)),
            },
        ] if *width == w.dtype().width() => Some(*w),
        _ => None,
    }
}

/// Joins adjacent segments of the same wire (or both unassigned).
pub(super) fn coalesce(v: Value) -> Value {
    let mut out: Value = Vec::new();
    for seg in v {
        if let Some(last) = out.last_mut() {
            match (last.src, seg.src) {
                (None, None) => {
                    last.width += seg.width;
                    continue;
                }
                (Some((a, alo)), Some((b, blo))) if a == b && alo == blo + seg.width => {
                    *last = Seg {
                        width: last.width + seg.width,
                        src: Some((b, blo)),
                    };
                    continue;
                }
                _ => {}
            }
        }
        out.push(seg);
    }
    out
}

/// `v` split so that a segment boundary lies at each of the bit positions
/// `cuts` (counted from bit 0).
pub(super) fn split(v: &Value, cuts: &BTreeSet<usize>) -> Value {
    let mut out = Vec::new();
    let mut top: usize = v.iter().map(|s| s.width).sum();
    for seg in v {
        let lo = top - seg.width;
        // the cuts strictly inside the segment, from the top down
        let mut hi = top;
        for &c in cuts.range(lo + 1..top).rev() {
            out.push(Seg {
                width: hi - c,
                src: seg.src.map(|(w, wlo)| (w, wlo + c - lo)),
            });
            hi = c;
        }
        out.push(Seg {
            width: hi - lo,
            src: seg.src,
        });
        top = lo;
    }
    out
}

/// The segment boundaries of `v`, counted from bit 0.
pub(super) fn cuts(v: &Value) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    let mut lo: usize = v.iter().map(|s| s.width).sum();
    for seg in v {
        lo -= seg.width;
        out.insert(lo);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sort;

    fn seg(width: usize, src: Option<(Wire, usize)>) -> Seg {
        Seg { width, src }
    }

    #[test]
    fn whole_values_are_wires() {
        let w = Wire::new(Sort(8));
        assert_eq!(as_wire(&whole(w)), Some(w));
        // the high half, or the whole wire at an offset, is no wire
        assert_eq!(as_wire(&vec![seg(4, Some((w, 4)))]), None);
        assert_eq!(as_wire(&vec![seg(4, Some((w, 4))), seg(4, None)]), None);
    }

    #[test]
    fn coalesce_joins_adjacent_bits_of_one_wire() {
        let (a, b) = (Wire::new(Sort(8)), Wire::new(Sort(4)));
        // a[7:4] a[3:0] is all of a
        let v = coalesce(vec![seg(4, Some((a, 4))), seg(4, Some((a, 0)))]);
        assert_eq!(as_wire(&v), Some(a));
        // unassigned bits join too
        assert_eq!(
            coalesce(vec![seg(2, None), seg(3, None)]),
            vec![seg(5, None)]
        );
        // a[3:0] a[7:4] (swapped halves) and bits of different wires stay apart
        let swapped = vec![seg(4, Some((a, 0))), seg(4, Some((a, 4)))];
        assert_eq!(coalesce(swapped.clone()), swapped);
        let mixed = vec![seg(4, Some((b, 0))), seg(4, Some((a, 0)))];
        assert_eq!(coalesce(mixed.clone()), mixed);
    }

    #[test]
    fn split_cuts_at_bit_positions() {
        let (a, b) = (Wire::new(Sort(8)), Wire::new(Sort(4)));
        // {b, a}: bits 11:8 from b, 7:0 from a
        let v = vec![seg(4, Some((b, 0))), seg(8, Some((a, 0)))];
        assert_eq!(cuts(&v), BTreeSet::from([0, 8]));
        let parts = split(&v, &BTreeSet::from([2, 8, 10]));
        assert_eq!(
            parts,
            vec![
                seg(2, Some((b, 2))),
                seg(2, Some((b, 0))),
                seg(6, Some((a, 2))),
                seg(2, Some((a, 0))),
            ]
        );
        assert_eq!(cuts(&parts), BTreeSet::from([0, 2, 8, 10]));
        // cuts outside the value or on existing boundaries change nothing
        assert_eq!(split(&v, &BTreeSet::from([0, 8, 12])), v);
        // splitting and joining again gives the value back
        assert_eq!(coalesce(parts), v);
    }
}
