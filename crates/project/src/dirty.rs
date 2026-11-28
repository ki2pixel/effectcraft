//! Which frames of a comp an edit can change: frame caches keep the others.

use super::comp::Comp;
use super::item::{Heard, ItemId, LayerId};
use super::layer::{Layer, LayerSource, reads_other_frames};
use super::{Arc, GroupKind, Node, Project, PropGroup, Tick};

/// Precomps nested deeper than this count as changed wherever they show.
const MAX_DEPTH: usize = 32;

type Seen = std::collections::HashMap<ItemId, Option<Vec<(Tick, Tick)>>>;

/// The frames a trim of a comp keeps: [`Project::trim_carry`].
pub struct TrimCarry {
    /// Frames of the comp before the edit sit `shift` frames after the edited comp's (positive
    /// when the comp was trimmed, negative when the trim was undone).
    pub shift: i64,
    /// Comp-time spans of the edited comp where its frames may draw differently: the trim
    /// shifted the comp time expressions and effects read, or cut a layer that samples across
    /// the new edge.
    pub dirty: Vec<(Tick, Tick)>,
}

impl Project {
    /// The frames a trim of `comp` (`Composition ▸ Trim Comp to Work Area`) keeps from `old`:
    /// the trim moves every layer by the work area's start, so its frames show the same layer
    /// times at shifted comp times. `None` when the edit is not such a trim (or its undo), or
    /// when it changed something the shift can't account for.
    pub fn trim_carry(&self, old: &Project, comp: ItemId, heard: &Heard) -> Option<TrimCarry> {
        let (new_c, old_c) = (self.comp(comp)?, old.comp(comp)?);
        if self.settings != old.settings || self.items.len() != old.items.len() {
            return None;
        }
        // Only this comp changed: every other item (and so every nested comp) is identical.
        for (id, item) in &self.items {
            if *id != comp && old.items.get(id) != Some(item) {
                return None;
            }
        }
        if new_c.layers.len() != old_c.layers.len()
            || new_c.layers.iter().zip(old_c.layers.iter()).any(|(a, b)| a.id != b.id)
            || !new_c.same_trim_settings(old_c)
        {
            return None;
        }
        let new_is_trimmed = new_c.duration < old_c.duration;
        let (trimmed, untrimmed) = if new_is_trimmed {
            (new_c, old_c)
        } else if old_c.duration < new_c.duration {
            (old_c, new_c)
        } else {
            return None;
        };
        // The work area's start, from the layers it moved (a work area dragged since says
        // nothing of the shift).
        let a = match (untrimmed.layers.first(), trimmed.layers.first()) {
            (Some(u), Some(t)) => u.start_time - t.start_time,
            // (an empty comp draws its background at every frame: the work areas say which
            // range was kept)
            _ if untrimmed.layers.is_empty() => {
                if untrimmed.work_area.1 - untrimmed.work_area.0 != trimmed.duration {
                    return None;
                }
                untrimmed.work_area.0
            }
            _ => return None,
        };
        if a < Tick::ZERO || a + trimmed.duration > untrimmed.duration {
            return None;
        }
        // The shift must be a whole number of frames at the same tick: a work area between
        // frames has no frame to move to.
        let fr = new_c.frame_rate;
        let a_frames = fr.frame_at(a);
        if a_frames < 0 || fr.tick_of(a_frames) != a {
            return None;
        }
        for (t, u) in trimmed.layers.iter().zip(untrimmed.layers.iter()) {
            if t.start_time != u.start_time - a || t.in_point != (u.in_point - a).max(Tick::ZERO) || t.out_point != (u.out_point - a).min(trimmed.duration) {
                return None;
            }
            // Everything but the span draws the same (expressions and time-dependent effects
            // are handled below).
            if !(Layer { start_time: u.start_time, in_point: u.in_point, out_point: u.out_point, ..t.clone() }).same_pixels_with(
                u,
                true,
                heard.contains(comp, t.id),
            ) {
                return None;
            }
        }
        // A trim only shifts comp time: expressions (`time`, `thisComp.duration`) and effects
        // that read it can change everywhere they draw, in this comp and in the comps it nests.
        let mut dirty = expression_spans_at(self, comp, &mut std::collections::HashMap::new(), 0)?;
        // Effects that read comp time directly (the timecode text, plug-ins) are not expressions.
        let mut time_dependent: Vec<LayerId> = new_c.layers.iter().filter(|l| reads_comp_time(l)).map(|l| l.id).collect();
        loop {
            let more: Vec<LayerId> = new_c
                .layers
                .iter()
                .filter(|l| !time_dependent.contains(&l.id) && l.referenced_layers().iter().any(|r| time_dependent.contains(r)))
                .map(|l| l.id)
                .collect();
            if more.is_empty() {
                break;
            }
            time_dependent.extend(more);
        }
        for l in new_c.layers.iter().filter(|l| time_dependent.contains(&l.id)) {
            if l.environment || l.environment_background {
                return None;
            }
            dirty.push((l.in_point, l.out_point));
        }
        // A cut layer's shutter samples across the new edge where the other comp had none (a
        // layer reading its own other frames can reach further: then all of it is dirty).
        let silent = self.silent_layers(comp, untrimmed, heard);
        let (before, after) = new_c.shutter_margin();
        let trim_start = if new_is_trimmed { Tick::ZERO } else { a };
        let trim_end = trim_start + trimmed.duration;
        for l in untrimmed.layers.iter().filter(|l| !silent.contains(&l.id)) {
            let cut_in = l.in_point < a && l.out_point > a;
            let cut_out = l.out_point > a + trimmed.duration && l.in_point < a + trimmed.duration;
            if !cut_in && !cut_out {
                continue;
            }
            if reads_itself_elsewhen(l) {
                if let Some(new_l) = new_c.layer(l.id) {
                    dirty.push((new_l.in_point, new_l.out_point));
                }
            } else if before > Tick::ZERO || after > Tick::ZERO {
                if cut_in {
                    dirty.push((trim_start - after, trim_start + before));
                }
                if cut_out {
                    dirty.push((trim_end - after, trim_end + before));
                }
            }
        }
        Some(TrimCarry { shift: if new_is_trimmed { a_frames } else { -a_frames }, dirty })
    }

    /// The comp-time spans of `comp` where `self` may draw other pixels than `old` (the project
    /// before an edit): [`Comp::changed_spans`] of its layers, and for a precomp whose comp
    /// changed, the spans that comp changed in, where the precomp shows them (#1641). Layers that
    /// can't change the picture ([`Project::silent_layers`]) don't count, nor does the audio of
    /// layers no effect draws it from (`heard`, best of both states). Expressions can read
    /// anything, so any change (`outside` these comps too: another comp, an item's name) reaches
    /// the layers that have one. `None` when the edit can reach any frame.
    pub fn dirty_spans(&self, old: &Project, comp: ItemId, heard: &Heard, outside: bool) -> Option<Vec<(Tick, Tick)>> {
        self.dirty_spans_at(old, comp, heard, outside, 0, &mut std::collections::HashMap::new())
    }

    /// `seen`: the spans of the comps already walked in this call, so a precomp shared by many
    /// layers is walked once.
    fn dirty_spans_at(&self, old: &Project, comp: ItemId, heard: &Heard, outside: bool, depth: usize, seen: &mut Seen) -> Option<Vec<(Tick, Tick)>> {
        if depth > MAX_DEPTH {
            return None;
        }
        if let Some(spans) = seen.get(&comp) {
            return spans.clone();
        }
        let spans = self.comp_dirty_spans(old, comp, heard, outside, depth, seen);
        seen.insert(comp, spans.clone());
        spans
    }

    fn comp_dirty_spans(&self, old: &Project, comp: ItemId, heard: &Heard, outside: bool, depth: usize, seen: &mut Seen) -> Option<Vec<(Tick, Tick)>> {
        let (new_c, old_c) = (self.comp_arc(comp)?, old.comp_arc(comp)?);
        // Expressions may read what changed elsewhere, even from comps that didn't change.
        let reads_elsewhere = heard.expressions && outside;
        let quiet = self.silent_layers(comp, &new_c, heard);
        // An edit of this comp is outside the comps it nests: their expressions may read it
        // (`comp("Main").layer("Controls")`, at their own time).
        let nested_outside = outside || !Arc::ptr_eq(&new_c, &old_c);
        // Precomps (drawing the same comp before and after) whose comp changed: where they show it.
        let mut touched = Vec::new();
        let mut nested = Vec::new();
        for l in new_c.layers.iter().filter(|l| !quiet.contains(&l.id)) {
            let LayerSource::Comp { item } = l.source else { continue };
            if old_c.layer(l.id).is_none_or(|before| before.source != l.source) {
                continue;
            }
            // (an unchanged `Arc` says nothing of the comps nested in it: walk it anyway)
            let (Some(n), Some(_)) = (self.comp_arc(item), old.comp_arc(item)) else { continue };
            let inner = self.dirty_spans_at(old, item, heard, nested_outside, depth + 1, seen);
            if inner.as_ref().is_some_and(Vec::is_empty) {
                continue;
            }
            touched.push(l.id);
            nested.extend(shown(l, &new_c, &n, inner));
        }
        if Arc::ptr_eq(&new_c, &old_c) && touched.is_empty() && !reads_elsewhere {
            return Some(vec![]);
        }
        let picture = |p: &Project, c: &Comp| {
            let mut c = c.clone();
            let quiet = p.silent_layers(comp, &c, heard);
            c.layers.retain(|l| !quiet.contains(&l.id));
            c
        };
        let (a, b) = (picture(old, &old_c), picture(self, &new_c));
        let elsewhere = heard.expressions && (outside || !new_c.same_pixels_with(&old_c, true, &|l| heard.contains(comp, l)) || !touched.is_empty());
        let mut spans = b.changed_spans(&a, heard.expressions, &|l| heard.contains(comp, l), elsewhere, &touched)?;
        spans.extend(nested);
        Some(spans)
    }
}

/// The comp-time spans of `outer` where its precomp layer `l` shows the spans `inner` of its comp
/// `n` (`None`: all of it), widened by `outer`'s shutter. A time remap, Responsive Design
/// timing or an effect reading the layer at other times (Echo, Posterize Time…: [`reads_other_frames`],
/// or an effect picking the layer itself) can show any nested time anywhere: then the layer's
/// whole span.
fn shown(l: &Layer, outer: &Comp, n: &Comp, inner: Option<Vec<(Tick, Tick)>>) -> Vec<(Tick, Tick)> {
    let (before, after) = outer.shutter_margin();
    let whole = vec![(l.in_point - before, l.out_point + after)];
    let responsive = (l.stretch - 100.0).abs() > 1e-9 && n.markers.iter().any(|m| m.protected && m.duration > Tick::ZERO);
    let Some(inner) = inner else { return whole };
    if l.props.get("timeRemap").is_some() || responsive || reads_itself_elsewhen(l) {
        return whole;
    }
    // (a nested frame either side: Preserve Frame Rate holds nested frames, Frame Blending mixes
    // neighbours)
    let fd = n.frame_duration();
    inner
        .into_iter()
        .filter_map(|(s, e)| {
            let (x, y) = (l.comp_time(s - fd), l.comp_time(e + fd));
            let (lo, hi) = (x.min(y).max(l.in_point), x.max(y).min(l.out_point));
            (lo <= hi).then(|| (lo - before, hi + after))
        })
        .collect()
}

/// Some effect of `l` reads the layer at other times ([`reads_other_frames`]) or picks the layer
/// itself (a layer parameter set to its own layer).
fn reads_itself_elsewhen(l: &Layer) -> bool {
    let time_fx = |c: &Node| matches!(c, Node::Group(f) if matches!(&f.kind, GroupKind::Effect { effect } if reads_other_frames(effect)));
    l.picked_layers().contains(&l.id) || l.props.group("effects").is_some_and(|e| e.children.iter().any(time_fx))
}

/// An enabled effect whose output reads comp time directly: the timecode text showing the comp's
/// time, and plug-ins, which the host gives the comp time.
fn reads_comp_time(l: &Layer) -> bool {
    l.effects().is_some_and(|fx| {
        fx.groups().any(|g| g.enabled && matches!(&g.kind, GroupKind::Effect { effect } if effect == "ec.text.timecode" || !effect.starts_with("ec.")))
    })
}

/// The comp-time spans of `comp` where a layer with an expression, or one that reads it, draws: a
/// trim shifts the comp time expressions read (`time`, `thisComp.duration`). `None` when an
/// environment layer is reached (its expression lights every 3D frame) or the comps nest too deep.
fn expression_spans_at(
    p: &Project,
    comp: ItemId,
    seen: &mut std::collections::HashMap<ItemId, Option<Vec<(Tick, Tick)>>>,
    depth: usize,
) -> Option<Vec<(Tick, Tick)>> {
    if depth > MAX_DEPTH {
        return None;
    }
    if let Some(spans) = seen.get(&comp) {
        return spans.clone();
    }
    let Some(c) = p.comp(comp) else { return Some(vec![]) };
    let mut reached: Vec<LayerId> = c.layers.iter().filter(|l| any_expression(&l.props)).map(|l| l.id).collect();
    loop {
        let more: Vec<LayerId> =
            c.layers.iter().filter(|l| !reached.contains(&l.id) && l.referenced_layers().iter().any(|r| reached.contains(r))).map(|l| l.id).collect();
        if more.is_empty() {
            break;
        }
        reached.extend(more);
    }
    let (before, after) = c.shutter_margin();
    let mut spans = Vec::new();
    for l in &c.layers {
        if !reached.contains(&l.id) {
            continue;
        }
        if l.environment || l.environment_background {
            return None;
        }
        if l.in_point < l.out_point {
            spans.push((l.in_point - before, l.out_point + after));
        }
    }
    for l in &c.layers {
        let LayerSource::Comp { item } = l.source else { continue };
        let Some(n) = p.comp(item) else { continue };
        let inner = expression_spans_at(p, item, seen, depth + 1)?;
        if inner.is_empty() {
            continue;
        }
        spans.extend(shown(l, c, n, Some(inner)));
    }
    seen.insert(comp, Some(spans.clone()));
    Some(spans)
}

/// Some property of the group carries an expression.
fn any_expression(g: &PropGroup) -> bool {
    g.children.iter().any(|c| match c {
        Node::Prop(p) => p.expr.is_some(),
        Node::Group(sub) => any_expression(sub),
    })
}
