// Minimal hook for "authentic" evaluation runs - the game's own native
// requestAnimationFrame and performance.now are left completely untouched, unlike
// ws.js (which hijacks both to give Python explicit, deterministic control over
// physics timing for training and approximate real-time eval). This file exists only
// to satisfy QWOP.min.js's one hard external dependency and give Python a way to
// observe tick progress and foot-contact state - everything else (state/terminal,
// reset(), key handling) is the game's own unmodified behavior, genuinely paced by
// the browser exactly as a human's session would be.
//
// Intended for QWOPAuthenticEnv (src/gym_qwop/envs/qwop_authentic_env.py) - final,
// trustworthy world-record-comparison runs, not training or iterative eval (see that
// class's docstring). Actions are driven by genuine Selenium ActionChains key events,
// not any JS-side action-application helper - QWOP.min.js's own native
// on_keydown/on_keyup handlers pick those up directly, the same as a human's keypresses
// would, so no window.__qwopApplyAction-equivalent is defined here at all.

// Defensive: keeps physics from auto-pausing if the automated window ever loses
// visibility/focus mid-run. Harmless in the normal case (a real, focused window).
Object.defineProperty(document, 'hidden', { get: function () { return false }, configurable: true })
Object.defineProperty(document, 'visibilityState', { get: function () { return 'visible' }, configurable: true })
var _origAddEventListener = document.addEventListener.bind(document)
document.addEventListener = function (type, listener, options) {
    if (/visibilitychange/i.test(type)) return
    return _origAddEventListener(type, listener, options)
}

// QWOP.min.js calls window.__qwopStepWorld(this.m_world) in place of a raw
// this.m_world.step(0.04, 5, 5) call (see QWOP.min.js's update()) - this hook must
// exist or the game throws. Its body is identical to ws.js's version (same physics
// timestep, same tick counter), but - critically - requestAnimationFrame itself is
// never touched here, so the browser's own native scheduler keeps calling QWOP.min.js's
// update loop at its natural, genuine pace. window.physicsTick then faithfully counts
// real physics ticks driven by real elapsed time, with no setInterval approximation or
// FRAME_INTERVAL_MS tuning involved anywhere.
var __qwopPendingWait = null
window.physicsTick = 0

// Per-tick log of the game's own native key-down flags (window.game.QDown/WDown/
// ODown/PDown - set directly by QWOP.min.js's real on_keydown/on_keyup handlers, the
// same ones a human's keystrokes land in). Lets Python measure pipeline latency: how
// many ticks after the observation it decided an action from does that action's key
// change actually take effect, given the real Selenium round trip (ActionChains
// dispatch, then this wait call) in between - see QWOPAuthenticEnv._apply_action_and_read.
// A bounded ring buffer, not a running log: only recent history is ever needed to find
// when a just-sent key state first appears, and this runs every tick for the life of
// the page.
var __qwopKeyLog = []
var __QWOP_KEY_LOG_MAX = 64

function __qwopLogKeyState() {
    var g = window.game
    if (!g) return
    __qwopKeyLog.push([window.physicsTick, !!g.QDown, !!g.WDown, !!g.ODown, !!g.PDown])
    if (__qwopKeyLog.length > __QWOP_KEY_LOG_MAX) __qwopKeyLog.shift()
}

function __qwopRespond(callback) {
    callback({ state: window.state, terminal: !!window.terminal, tick: window.physicsTick, keyLog: __qwopKeyLog.slice() })
}

function __qwopResolvePendingWait() {
    if (!__qwopPendingWait) return
    if (!(window.terminal || window.physicsTick - __qwopPendingWait.startTick >= __qwopPendingWait.minTicks)) return
    var pending = __qwopPendingWait
    __qwopPendingWait = null
    __qwopRespond(pending.callback)
}

window.__qwopStepWorld = function (world) {
    world.step(0.04, 5, 5)
    window.physicsTick += 1
    __qwopLogKeyState()
    // Deferred (not called synchronously here): on the exact tick a win lands,
    // world.step() above synchronously triggers beginContact -> jumpLanded = true,
    // but QWOP.min.js doesn't rebuild window.state (won: this.jumpLanded) or call
    // endGame() (which sets window.terminal = true) until *after* this function
    // returns, later in the same update() call. Resolving synchronously here would
    // read window.state/window.terminal before that happens, handing Python a
    // stale won=false/terminal=false for the winning tick - and since endGame()
    // also sets this.pause = true, no further tick ever arrives to let a
    // subsequent wait resolve normally, hanging until the Selenium script
    // timeout. A microtask runs after the rest of this synchronous update() call
    // (state rebuild + endGame()) completes, so it always sees this tick's final,
    // correct state - the window.terminal setter trap below still resolves things
    // immediately in the interim if a pending wait is still open when endGame()
    // sets it.
    Promise.resolve().then(__qwopResolvePendingWait)
}

// window.terminal is a plain property, not a function call this file controls, so a
// win can't be observed by hooking a call site the way __qwopStepWorld hooks physics
// ticks. It matters because QWOP.min.js's endGame() sets `this.pause = true` on a win
// (QWOP.min.js:1086) *before* setting window.terminal = true (:906) - and once paused,
// QWOP.min.js never calls window.__qwopStepWorld again until reset(). A wait pending
// at that moment would otherwise only be resolved by a tick that's never coming,
// hanging until Selenium's execute_async_script timeout. Defining terminal as an
// accessor lets the assignment itself trigger resolution, independent of whether any
// further physics tick ever lands - ws.js's __qwopTick achieves the same effect by
// checking window.terminal right after running update() each interval (a way this
// mode, driven by the game's own native rAF rather than a wrapper we control, has no
// equivalent hook for).
var __qwopTerminal = false
Object.defineProperty(window, 'terminal', {
    get: function () { return __qwopTerminal },
    set: function (value) {
        __qwopTerminal = value
        __qwopResolvePendingWait()
    },
    configurable: true,
})

// Waits for at least minTicks NEW physics ticks to land before resolving, then reads
// state - without this, QWOPAuthenticEnv.step() (which applies an action via genuine
// ActionChains, then reads whatever's current) can easily land *within* the same
// physics tick as the previous read if Python's own round trip happens to be faster
// than the ~33ms native tick interval - confirmed empirically (ticks_this_step was 0
// a majority of the time without this), which forces sim_dt=0 and therefore velx/joint
// velocities to be reported as exactly zero even while the character is actively
// moving - a far more severe distortion than a scale mismatch. No action-application
// coupling here (unlike ws.js's version) - Python applies the action via ActionChains
// in a separate call before this one, so this is purely "wait, then read".
window.__qwopWaitForTicks = function (minTicks, callback) {
    var startTick = window.physicsTick
    if (window.physicsTick - startTick >= minTicks) {
        __qwopRespond(callback)
        return
    }
    __qwopPendingWait = { startTick: startTick, minTicks: minTicks, callback: callback }
}

// Foot-ground contact state - the actual set/clear logic lives in QWOP.min.js's own
// beginContact/endContact patches (added directly to that file, not ws.js), so these
// are just the same initial values ws.js also declares.
window.leftFootDown = false
window.rightFootDown = false

// So timer.js's existing HUD (already Date.now()-based for this flag - see its
// file-level comment) applies here too, without needing its own new mode.
window.__qwopRealtime = true

// QWOP.min.js unconditionally calls window.qwopLoaded() once its own asset loading
// finishes (QWOP.min.js:532, inside its ready() callback chain) - see ws.js's own stub
// for the lock-step/realtime envs. Without SOME definition, this throws a
// ReferenceError inside ready()'s own call chain, which aborts the rest of that
// sequence before it gets to registering the game's own requestAnimationFrame loop at
// all - silently leaving
// physicsTick stuck at 0 forever, with no other symptom. A no-op is all this mode needs.
window.qwopLoaded = function () {}
