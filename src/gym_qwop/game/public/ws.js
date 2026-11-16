// Glue for the Selenium-driven QWOP environments (qwop_env.py's lock-step QWOPEnv,
// qwop_realtime_env.py's QWOPRealtimeEnv) - Python drives ticks/actions directly via
// execute_script calls into the functions defined here (__qwopAdvance,
// __qwopApplyAction, __qwopWaitForTicks), and reads state/terminal straight from
// `window.state` / `window.terminal`, which QWOP.min.js already sets on every physics
// tick.

// Timing mode: lock-step (default) only advances physics when explicitly driven from
// Python (see __qwopAdvance below). realtime=1 instead drives it from an autonomous,
// Python-independent timer (see the __qwopRealtime block below __qwopAdvance) - used by
// QWOPRealtimeEnv so the agent can't pause or influence how much sim time passes between
// its actions. speed lets that autonomous timer run faster than true wall-clock
// ("overclocking"), for training throughput; speed=1 is genuine real-time parity.
function getQueryParam(name) {
    var m = new RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search)
    return m ? decodeURIComponent(m[1]) : null
}
var __qwopRealtime = getQueryParam('realtime') === '1'
var __qwopSpeed = parseFloat(getQueryParam('speed')) || 1

// Headless/backgrounded browser tabs are typically reported as hidden even when they're
// the only tab open. The game engine listens for visibilitychange and auto-pauses physics
// when the page looks hidden, which would silently freeze training. Neutralize that before
// QWOP.min.js (injected later, on window.onload) gets a chance to register its listener.
Object.defineProperty(document, 'hidden', { get: function () { return false }, configurable: true })
Object.defineProperty(document, 'visibilityState', { get: function () { return 'visible' }, configurable: true })
var _origAddEventListener = document.addEventListener.bind(document)
document.addEventListener = function (type, listener, options) {
    if (/visibilitychange/i.test(type)) return
    return _origAddEventListener(type, listener, options)
}

// Physics tick counter, incremented once per simulated 0.04s world.step() (see the patched
// call site in QWOP.min.js: `window.__qwopStepWorld(this.m_world)` in place of a raw
// `this.m_world.step(0.04, 5, 5)`). This is the source of truth for how much *simulated*
// time has actually elapsed - callers should compute velocity/reward from
// (posx delta) / (ticks delta * 0.04) instead of measuring wall-clock time.
window.physicsTick = 0
window.__qwopStepWorld = function (world) {
    world.step(0.04, 5, 5)
    window.physicsTick += 1
}

// Foot-ground contact state, set/cleared by the beginContact/endContact patches in
// QWOP.min.js - lets Python require an actual footstep (not just a thigh angle
// crossing the swap threshold) before crediting a stride. See _stride_bonus.
window.leftFootDown = false
window.rightFootDown = false

// Lock-step physics: the game schedules its loop via a plain window.requestAnimationFrame
// call, looked up fresh each time rather than cached, so replacing it here (before
// QWOP.min.js's loop ever starts) is enough to take over scheduling entirely. Instead of
// auto-firing on a timer, the frame callback is captured and only actually runs when
// explicitly pumped from Python via execute_script - normally once per env.step()/reset().
// This makes "how much simulated time elapses per RL step" a deliberate choice instead of
// depending on wall-clock/system timing (Selenium latency, GC pauses, CPU contention from
// other parallel workers).
window.__qwopFrameCallback = null
window.requestAnimationFrame = function (callback) {
    window.__qwopFrameCallback = callback
}

// The frame callback's own timestamp argument isn't what actually gates physics stepping -
// the engine's internal frame accumulator (core.timestamp()) reads window.performance.now()
// directly, bypassing whatever we pass into the callback. So a fake, callback-argument-only
// clock doesn't work: calling the callback back-to-back with real wall-clock time between
// calls made the engine's own accumulator batch a variable, unpredictable number of physics
// steps per call (sometimes 0, sometimes several), which is exactly the wall-clock coupling
// we're trying to eliminate. Overriding performance.now() itself - so the engine's own
// accumulator is driven by a fake clock we advance by a fixed amount per callback - fixes
// this: empirically, incrementing by FRAME_INTERVAL_MS per call yields (very close to)
// exactly one physics tick per call, regardless of how fast we actually call it.
window.__qwopFakeNow = performance.now()
Object.defineProperty(window.performance, 'now', {
    value: function () { return window.__qwopFakeNow },
    writable: true,
    configurable: true,
})
var FRAME_INTERVAL_MS = 33.34 // ~1000/30; empirically matches the engine's internal frame_time

// A pending window.__qwopWaitForTicks() request, resolved synchronously from inside
// __qwopTick the moment it's satisfied - see __qwopWaitForTicks below. Checking here
// (rather than on a separate setTimeout polling loop) means resolution happens exactly
// when a tick actually lands, with no extra polling-granularity slop on top.
var __qwopPendingWait = null

// Runs the currently-queued frame callback exactly once, advancing the fake clock by one
// frame interval first - which (see above) reliably yields one physics tick per call, with
// no guarantee it results in a physics tick when called pre-click (e.g. before the game's
// click-to-start has happened - see __qwopPumpFrame below). Shared by both timing modes:
// __qwopAdvance/__qwopPumpFrame call it explicitly (lock-step); the autonomous setInterval
// below calls it on its own schedule (realtime).
function __qwopTick() {
    var cb = window.__qwopFrameCallback
    window.__qwopFrameCallback = null
    window.__qwopFakeNow += FRAME_INTERVAL_MS
    if (cb) cb(window.__qwopFakeNow)

    // window.terminal is checked alongside the tick count, not just the tick count
    // alone: QWOP.min.js pauses physics the instant the game ends in a win (jumpLanded)
    // - window.__qwopStepWorld stops incrementing physicsTick entirely once paused - and
    // window.terminal is set true in that exact same synchronous tick. Without this
    // check, a pending wait that hasn't yet reached minTicks when a win lands would
    // never resolve, since no further ticks are coming until the next reset - Python
    // would block until Selenium's script timeout, not a fast step.
    if (__qwopPendingWait && (window.terminal || window.physicsTick - __qwopPendingWait.startTick >= __qwopPendingWait.minTicks)) {
        var pending = __qwopPendingWait
        __qwopPendingWait = null
        pending.callback({ state: window.state, terminal: !!window.terminal, tick: window.physicsTick })
    }
}

// Used only to pump the loop during initial connection, to get window.state populated -
// state capture happens whenever the player object exists, independent of whether physics
// is stepping yet.
window.__qwopPumpFrame = function () {
    __qwopTick()
}

// Runs n ticks back to back - explicit, Python-driven advancement (lock-step mode).
window.__qwopAdvance = function (n) {
    for (var i = 0; i < n; i++) __qwopTick()
}

if (__qwopRealtime) {
    // Autonomous, Python-independent ticking: the agent cannot pause or influence this by
    // taking longer to decide - ticks accrue on this timer regardless of when/whether
    // Python calls step(). __qwopSpeed > 1 runs it faster than true wall-clock, purely for
    // training throughput, while still preserving that pacing independence.
    setInterval(__qwopTick, FRAME_INTERVAL_MS / __qwopSpeed)
}

// Lets Python block (via execute_async_script) until at least minTicks NEW ticks have
// landed since this call started, instead of reading immediately - avoids a near-zero-
// progress read if step() happens to be called faster than physics is currently ticking.
// Only meaningful in realtime mode (lock-step's __qwopAdvance already drives an exact tick
// count directly, so there's nothing to wait for). `action` may be null to just wait
// without pressing any keys (e.g. during reset()). Resolution is push-based (see
// __qwopTick's __qwopPendingWait check), not polled - a setTimeout-based poll would add
// its own scheduling jitter on top of the tick timer's, which is exactly what made
// min_ticks_per_step=1 land closer to 1.5 ticks on average than a true 1.
window.__qwopWaitForTicks = function (action, minTicks, callback) {
    if (action) window.__qwopApplyAction(action)
    var startTick = window.physicsTick
    if (window.physicsTick - startTick >= minTicks) {
        // Already satisfied (e.g. minTicks <= 0) - resolve immediately, no need to wait
        // for the next tick.
        callback({ state: window.state, terminal: !!window.terminal, tick: window.physicsTick })
        return
    }
    __qwopPendingWait = { startTick: startTick, minTicks: minTicks, callback: callback }
}

var keyStates = {}

// QWOP.min.js unconditionally calls window.qwopLoaded() once its own asset loading
// finishes - without SOME definition, this throws a ReferenceError inside that call
// chain, silently leaving physicsTick stuck at 0 forever (see the identical stub and
// explanation in ws_authentic.js). The Selenium-driven envs (qwop_env.py,
// qwop_realtime_env.py) click the canvas and send keys directly via WebDriver instead
// of anything defined here, so this needs to do nothing else.
window.qwopLoaded = function () {}

function actionToKeys(action) {
    var checkKeys = ["q", "w", "o", "p"]
    var keys = []
    for (var i = 0; i < checkKeys.length; i++) {
        if (action[i]) keys.push(checkKeys[i])
    }
    return keys
}

// Exposed for the Selenium-driven environment (qwop_env.py) to apply an action via
// execute_script - dispatching synthetic key events directly is one script call, vs.
// Selenium's ActionChains which is a separate WebDriver round trip. Folding this into the
// same execute_script call as window.__qwopAdvance (see _advance_and_get_state in
// qwop_env.py) collapses step() to a single round trip instead of two.
window.__qwopApplyAction = function (action) {
    pressKeys(actionToKeys(action))
}

function pressKeys(keys) {
    var checkKeys = ["q", "w", "o", "p"]
    for (var i = 0; i < checkKeys.length; i++) {
        var key = checkKeys[i]
        var code = getCode(key)
        var shouldBeDown = keys.indexOf(key) > -1
        if (shouldBeDown && !keyStates[code]) {
            simulateKeydown(code)
            keyStates[code] = true
        } else if (!shouldBeDown && keyStates[code]) {
            simulateKeyup(code)
            keyStates[code] = false
        }
    }
}

function getCode(keyChar) {
    if (keyChar === 'q') return 81
    if (keyChar === 'w') return 87
    if (keyChar === 'o') return 79
    if (keyChar === 'p') return 80
    if (keyChar === 'r') return 82
    return 32
}

function simulateKeydown(keycode, isCtrl, isAlt, isShift) {
    var e = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, char: String.fromCharCode(keycode), key: String.fromCharCode(keycode), shiftKey: isShift, ctrlKey: isCtrl, altKey: isAlt })
    Object.defineProperty(e, 'keyCode', { get: function () { return this.keyCodeVal } })
    e.keyCodeVal = keycode
    document.dispatchEvent(e)
}

function simulateKeyup(keycode, isCtrl, isAlt, isShift) {
    var e = new KeyboardEvent("keyup", { bubbles: true, cancelable: true, char: String.fromCharCode(keycode), key: String.fromCharCode(keycode), shiftKey: isShift, ctrlKey: isCtrl, altKey: isAlt })
    Object.defineProperty(e, 'keyCode', { get: function () { return this.keyCodeVal } })
    e.keyCodeVal = keycode
    document.dispatchEvent(e)
}
