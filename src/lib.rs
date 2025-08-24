// src/lib.rs
// gmcl_speech: clientside Garry's Mod module
// - Captures default (or chosen) Windows mic via cpal (WASAPI)
// - Runs whisper-rs locally
// - Matches monograms/bigrams against internal trigger map with cooldowns
// - Queues matches and dispatches them on the Lua main thread via hook.Run
// Lua API (single table `speech`):
//   speech.AddTrigger(phrase[, cooldown], callback) -- register trigger with optional cooldown and a callback
//   speech.RemoveTrigger(phrase)                    -- remove a trigger
//   speech.GetTriggers() -> { {phrase=..., cooldown=...}, ... }
// Auto Start/Stop via hooks: PlayerStartVoice/PlayerEndVoice (local player only)
// Dispatch happens automatically every frame via a Think hook.
//
// Build: crate-type = ["cdylib"]
// Requires dependencies: gmod, anyhow, cpal, whisper-rs, crossbeam-channel, parking_lot, once_cell

//#![allow(clippy::needless_return)]
//#![allow(unsafe_op_in_unsafe_fn)]

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[macro_use]
extern crate gmod;


// --------------------------- Globals ---------------------------

static RUNNING: AtomicBool = AtomicBool::new(false);
static WORKER: Lazy<Mutex<Option<JoinHandle<()>>>> = Lazy::new(|| Mutex::new(None));

#[derive(Clone)]
struct Event {
    phrase: String, // normalized phrase that matched (unigram or bigram)
}

static QTX: Lazy<Mutex<Option<Sender<Event>>>> = Lazy::new(|| Mutex::new(None));
static QRX: Lazy<Mutex<Option<Receiver<Event>>>> = Lazy::new(|| Mutex::new(None));


// trigger map: normalized phrase -> Trigger
struct Trigger {
    cooldown: f32,
    last_fire: Instant,
}
static TRIGGERS: Lazy<Mutex<HashMap<String, Trigger>>> = Lazy::new(|| Mutex::new(HashMap::new()));

// configuration set by Start()
static MODEL_PATH: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::from("ggml-tiny.en.bin")));
static DEVICE_SUBSTR: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

// --------------------------- Utils ----------------------------

unsafe fn norm(s: &str) -> String {
    let s = s.to_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut last_space = false;
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            out.push(ch);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_string()
}

unsafe fn resample_linear_mono(input: &[f32], in_hz: u32, out_hz: u32) -> Vec<f32> {
    if input.is_empty() || in_hz == out_hz {
        return input.to_vec();
    }
    let out_len = ((input.len() as u64) * (out_hz as u64) / (in_hz as u64)) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let t = (i as f32) * (in_hz as f32) / (out_hz as f32);
        let i0 = t.floor() as usize;
        let i1 = (i0 + 1).min(input.len().saturating_sub(1));
        let frac = t - (i0 as f32);
        out.push(input[i0] * (1.0 - frac) + input[i1] * frac);
    }
    out
}

// -------------------------- Worker ----------------------------

unsafe fn spawn_worker() -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

    // prepare queue
    let (tx, rx) = unbounded::<Event>();
    *QTX.lock() = Some(tx);
    *QRX.lock() = Some(rx);

    // clone config
    let model = { MODEL_PATH.lock().clone() };
    let device_sel = { DEVICE_SUBSTR.lock().clone() };

    RUNNING.store(true, Ordering::SeqCst);

    let handle = thread::spawn(move || {
        let _ = || -> Result<()> {
            // init whisper
            let ctx = WhisperContext::new(&model)?;
            let mut state = ctx.create_state()?;

            // init cpal input
            let host = cpal::default_host();
            let device = if let Some(sel) = device_sel {
                let sel_l = sel.to_lowercase();
                let mut found = None;
                if let Ok(iter) = host.input_devices() {
                    for dev in iter {
                        let name = dev.name().unwrap_or_default();
                        if name.to_lowercase().contains(&sel_l) {
                            found = Some(dev);
                            break;
                        }
                    }
                }
                found.unwrap_or_else(|| host.default_input_device().expect("no default input device"))
            } else {
                host.default_input_device().expect("no default input device")
            };

            let cfg = device.default_input_config()?.config();
            let rate = cfg.sample_rate.0;
            let chans = cfg.channels as usize;

            let (audio_tx, audio_rx) = unbounded::<Vec<f32>>();

            let stream = device.build_input_stream(
                &cfg,
                move |data: &[f32], _| {
                    // interleaved -> mono
                    let mono: Vec<f32> = if chans <= 1 {
                        data.to_vec()
                    } else {
                        data.chunks_exact(chans).map(|f| {
                            let mut s = 0.0f32;
                            for c in 0..chans {
                                s += f[c];
                            }
                            s / (chans as f32)
                        }).collect()
                    };
                    // ~100ms chunks
                    let block = (rate / 10) as usize;
                    for ch in mono.chunks(block) {
                        if !ch.is_empty() {
                            let _ = audio_tx.send(ch.to_vec());
                        }
                    }
                },
                move |e| eprintln!("[gmcl_speech] stream error: {e}"),
                None,
            )?;
            stream.play()?;

            // sliding ring + decode
            let mut ring: VecDeque<f32> =
                VecDeque::with_capacity((rate as usize).saturating_mul(4));
            let mut last_words: VecDeque<String> = VecDeque::new();

            let tick = Duration::from_millis(400);
            let mut next = Instant::now();

            while RUNNING.load(Ordering::SeqCst) {
                // drain audio
                while let Ok(chunk) = audio_rx.try_recv() {
                    for s in chunk {
                        ring.push_back(s);
                    }
                    let cap = (rate as usize) * 4;
                    while ring.len() > cap {
                        ring.pop_front();
                    }
                }

                if Instant::now() >= next {
                    next += tick;

                    // 3s window @ 16kHz
                    let audio: Vec<f32> = {
                        let mut tmp: Vec<f32> = ring.iter().copied().collect();
                        tmp = resample_linear_mono(&tmp, rate, 16_000);
                        let want = 16_000 * 3;
                        if tmp.len() > want {
                            tmp[tmp.len() - want..].to_vec()
                        } else {
                            tmp
                        }
                    };

                    if audio.len() >= 8_000 {
                        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                        params.set_language(Some("en"));
                        params.set_n_threads(8);
                        params.set_no_timestamps(false);
                        params.set_single_segment(false);
                        params.set_print_realtime(false);
                        params.set_print_progress(false);
                        if state.full(params, &audio).is_ok() {
                            if let Ok(nseg) = state.full_n_segments() {
                                if nseg > 0 {
                                    if let Ok(text) = state.full_get_segment_text((nseg - 1) as i32) {
                                        let mut tokens: Vec<String> = text
                                            .split_whitespace()
                                            .map(|w| norm(w))
                                            .filter(|w| !w.is_empty())
                                            .collect();

                                        for w in tokens.drain(..) {
                                            last_words.push_back(w);
                                            if last_words.len() > 2 {
                                                last_words.pop_front();
                                            }

                                            // unigram
                                            if let Some(u) = last_words.back() {
                                                try_fire(u);
                                            }
                                            // bigram
                                            if last_words.len() == 2 {
                                                let bi = format!("{} {}", last_words[0], last_words[1]);
                                                try_fire(&bi);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                thread::sleep(Duration::from_millis(2));
            }

            Ok(())
        }();

        // ensure stopped flag if exited
        RUNNING.store(false, Ordering::SeqCst);
    });

    *WORKER.lock() = Some(handle);
    Ok(())
}

unsafe fn try_fire(key: &str) {
    let keyn = norm(key);
    if keyn.is_empty() {
        return;
    }
    let mut map = TRIGGERS.lock();
    if let Some(tr) = map.get_mut(&keyn) {
        if tr.last_fire.elapsed().as_secs_f32() >= tr.cooldown {
            tr.last_fire = Instant::now();
            if let Some(tx) = &*QTX.lock() {
                let _ = tx.send(Event { phrase: keyn.clone() });
            }
        }
    }
}

// -------------------------- Lua Glue --------------------------

unsafe fn lua_bool(lua: &gmod::lua::State, b: bool) -> i32 { lua.push_boolean(b); 1 }

// -------------------------- Lua Fns --------------------------

#[lua_function]
unsafe fn l_speech_add_trigger(lua: gmod::lua::State) -> i32 {
    let phrase = norm(&lua.check_string(1));
    let cooldown = if lua.get_top() >= 2 { lua.check_number(2) as f32 } else { 0.5 };
    if phrase.is_empty() || lua.get_top() < 3 || !lua.is_function(3) {
        return lua_bool(&lua, false);
    }
    TRIGGERS.lock().insert(
        phrase.clone(),
        Trigger { cooldown, last_fire: Instant::now() - Duration::from_secs(1) },
    );
    // speech.__callbacks[phrase] = arg3
    lua.get_global(lua_string!("speech"));
    lua.get_field(-1, lua_string!("__callbacks"));
    lua.push_string(&phrase);
    lua.push_value(3);
    lua.set_table(-3);
    lua.pop(); // __callbacks
    lua.pop(); // speech
    lua_bool(&lua, true)
}

#[lua_function]
unsafe fn l_speech_remove_trigger(lua: gmod::lua::State) -> i32 {
    let phrase = norm(&lua.check_string(1));
    TRIGGERS.lock().remove(&phrase);
    // speech.__callbacks[phrase] = nil
    lua.get_global(lua_string!("speech"));
    lua.get_field(-1, lua_string!("__callbacks"));
    lua.push_string(&phrase);
    lua.push_nil();
    lua.set_table(-3);
    lua.pop(); // __callbacks
    lua.pop(); // speech
    0
}

#[lua_function]
unsafe fn l_speech_get_triggers(lua: gmod::lua::State) -> i32 {
    lua.new_table();
    let mut i = 1isize;
    for (phrase, tr) in TRIGGERS.lock().iter() {
        lua.push_integer(i);
        lua.new_table();
        lua.push_string("phrase");
        lua.push_string(phrase);
        lua.set_table(-3);
        lua.push_string("cooldown");
        lua.push_number(tr.cooldown as f64);
        lua.set_table(-3);
        lua.set_table(-3);
        i += 1;
    }
    1
}

// manual start/stop are internal now; voice hooks control them

// Auto-start/stop only for the local player's voice activity
#[lua_function]
unsafe fn l_on_player_start_voice(lua: gmod::lua::State) -> i32 {
    // arg1: Player who started voice
    lua.get_global(lua_string!("LocalPlayer"));
    if lua.pcall(0, 1, 0) != 0 {
        // failed to resolve LocalPlayer
        lua.pop();
        return 0;
    }
    let is_local = lua.raw_equal(-1, 1);
    lua.pop();
    if is_local && !RUNNING.load(Ordering::SeqCst) {
        if let Err(e) = spawn_worker() {
            eprintln!("[gmcl_speech] failed to start: {e}");
        }
    }
    0
}

#[lua_function]
unsafe fn l_on_player_end_voice(lua: gmod::lua::State) -> i32 {
    lua.get_global(lua_string!("LocalPlayer"));
    if lua.pcall(0, 1, 0) != 0 {
        lua.pop();
        return 0;
    }
    let is_local = lua.raw_equal(-1, 1);
    lua.pop();
    if is_local {
        RUNNING.store(false, Ordering::SeqCst);
        if let Some(h) = WORKER.lock().take() {
            let _ = h.thread().id();
        }
    }
    0
}

#[lua_function]
unsafe fn l_speech_dispatch(lua: gmod::lua::State) -> i32 {
    let rx_opt = QRX.lock();
    if rx_opt.is_none() {
        return 0;
    }
    let rx = rx_opt.as_ref().unwrap();
    for _ in 0..64 {
        match rx.try_recv() {
            Ok(ev) => {
                // Call registered callback: speech.__callbacks[phrase](phrase)
                lua.get_global(lua_string!("speech"));
                lua.get_field(-1, lua_string!("__callbacks"));
                lua.push_string(&ev.phrase);
                lua.get_table(-2);
                if lua.is_function(-1) {
                    lua.push_string(&ev.phrase);
                    if lua.pcall(1, 0, 0) != 0 {
                        if let Some(err) = lua.get_string(-1) {
                            eprintln!("[gmcl_speech] callback error: {}", err);
                        }
                        lua.pop();
                    }
                } else {
                    lua.pop();
                }
                lua.pop(); // __callbacks
                lua.pop(); // speech
            }
            Err(_) => break,
        }
    }
    0
}

unsafe fn setup_hooks(lua: gmod::lua::State) {
    // Auto-dispatch every frame using Think hook
    lua.get_global(lua_string!("hook"));

    lua.get_field(-1, lua_string!("Add"));
    lua.push_string("Think");
    lua.push_string("gmcl_speech_Dispatch");
    lua.push_function(l_speech_dispatch);
    if lua.pcall(3, 0, 0) != 0 {
        if let Some(err) = lua.get_string(-1) {
            eprintln!("[gmcl_speech] hook.Add error: {}", err);
        }
        lua.pop();
    }

    lua.get_field(-1, lua_string!("Add"));
    lua.push_string("PlayerStartVoice");
    lua.push_string("gmcl_speech_PlayerStartVoice");
    lua.push_function(l_on_player_start_voice);
    if lua.pcall(3, 0, 0) != 0 {
        if let Some(err) = lua.get_string(-1) {
            eprintln!("[gmcl_speech] hook.Add error: {}", err);
        }
        lua.pop();
    }

    lua.get_field(-1, lua_string!("Add"));
    lua.push_string("PlayerEndVoice");
    lua.push_string("gmcl_speech_PlayerEndVoice");
    lua.push_function(l_on_player_end_voice);
    if lua.pcall(3, 0, 0) != 0 {
        if let Some(err) = lua.get_string(-1) {
            eprintln!("[gmcl_speech] hook.Add error: {}", err);
        }
        lua.pop();
    }

    lua.pop(); // pop hook table
}

#[gmod13_open]
unsafe fn gmod13_open(lua: gmod::lua::State) -> i32 {
    // Create `speech` table with minimal API
    lua.new_table();
    // internal callbacks table: speech.__callbacks = {}
    lua.push_string("__callbacks");
    lua.new_table();
    lua.set_table(-3);
    // methods
    lua.push_function(l_speech_add_trigger);
    lua.set_field(-2, lua_string!("AddTrigger"));
    lua.push_function(l_speech_remove_trigger);
    lua.set_field(-2, lua_string!("RemoveTrigger"));
    lua.push_function(l_speech_get_triggers);
    lua.set_field(-2, lua_string!("GetTriggers"));
    // expose global
    lua.set_global(lua_string!("speech"));

    // Setup hooks
    setup_hooks(lua);

    0
}

#[gmod13_close]
unsafe fn gmod13_close(_lua: gmod::lua::State) -> i32 {
    RUNNING.store(false, Ordering::SeqCst);
    if let Some(h) = WORKER.lock().take() {
        let _ = h.thread().id();
    }
    // Remove Think hook we added
    // Note: Lua state may already be shutting down; ignore errors
    let lua = _lua;
    lua.get_global(lua_string!("hook"));
    lua.get_field(-1, lua_string!("Remove"));
    lua.push_string("Think");
    lua.push_string("gmcl_speech_Dispatch");
    if lua.pcall(2, 0, 0) != 0 {
        lua.pop();
    }
    lua.pop();
    // Remove voice hooks
    lua.get_global(lua_string!("hook"));
    lua.get_field(-1, lua_string!("Remove"));
    lua.push_string("PlayerStartVoice");
    lua.push_string("gmcl_speech_PlayerStartVoice");
    if lua.pcall(2, 0, 0) != 0 { lua.pop(); }
    lua.pop();
    lua.get_global(lua_string!("hook"));
    lua.get_field(-1, lua_string!("Remove"));
    lua.push_string("PlayerEndVoice");
    lua.push_string("gmcl_speech_PlayerEndVoice");
    if lua.pcall(2, 0, 0) != 0 { lua.pop(); }
    lua.pop();
    0
}
