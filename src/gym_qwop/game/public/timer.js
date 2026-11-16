// Speedrun-style HUD: a run timer plus 10m splits compared against a personal best,
// LiveSplit-style. Purely client-side and read-only against globals ws.js/QWOP.min.js
// already expose (window.physicsTick, window.terminal, window.state,
// window.__qwopRealtime), so it needs zero changes to either of those files and can't
// affect game/training behavior.
//
// Clock source depends on mode:
//  - Lock-step (default): physicsTick * PHYSICS_DT (simulated seconds). Wall-clock time
//    is meaningless here - Python drives ticks explicitly via execute_script, at
//    whatever pace the training machine/round-trip latency happens to allow, completely
//    decoupled from real elapsed time.
//  - Real-time (?realtime=1): genuine wall-clock time (Date.now()), NOT
//    physicsTick * PHYSICS_DT. Verified both by reading the code and against a real
//    stopwatch: ws.js's autonomous timer fires one physics tick every
//    FRAME_INTERVAL_MS=33.34ms of real time, but world.step(0.04, 5, 5) always advances
//    the simulation by a *fixed* 0.04s of simulated time per tick regardless of that -
//    0.04/0.03334 ~= 1.2x, i.e. physicsTick-based timing runs fast relative to real
//    time by design under this mode (confirmed empirically: ~5s/min drift against a
//    real stopwatch, in the same direction and rough magnitude predicted by that ratio).
//    Since real-time mode's whole purpose is comparing against a human-timed world
//    record, wall-clock time is the correct thing to display here, not the inflated
//    simulated-time count. window.performance.now() is *not* usable for this - ws.js
//    fakes it (see window.__qwopFakeNow) - but Date.now() is untouched.
//
// PB splits persist in localStorage (per browser/machine, not per training run) - see
// PB_KEY below.
(function () {
    var PHYSICS_DT = 0.04;
    var SPLIT_INTERVAL_M = 10;
    var NUM_SPLITS = 10; // 10m, 20m, ..., 100m (the finish line)
    var PB_KEY = 'qwop_speedrun_pb_splits';
    var isRealtime = !!window.__qwopRealtime;

    function formatTime(totalSeconds) {
        var minutes = Math.floor(totalSeconds / 60);
        var seconds = totalSeconds - minutes * 60;
        var secStr = seconds.toFixed(3);
        if (seconds < 10) secStr = '0' + secStr;
        return minutes + ':' + secStr;
    }

    function formatDelta(delta) {
        return (delta <= 0 ? '-' : '+') + Math.abs(delta).toFixed(2);
    }

    function loadPB() {
        try {
            var raw = localStorage.getItem(PB_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            return null;
        }
    }

    function savePB(splits) {
        try {
            localStorage.setItem(PB_KEY, JSON.stringify(splits));
        } catch (e) {
            // Ignore (e.g. private-browsing storage denial) - PB tracking is a nice-to-have,
            // never worth breaking the timer itself over.
        }
    }

    var prevTerminal = null;
    var prevPosx = 0;
    var RESET_POSX_THRESHOLD = 0.5; // meters - reset() snaps posx back near 0
    var runStartTick = 0;
    var runStartWallClock = 0; // Date.now() ms - only used/meaningful when isRealtime
    var frozen = false;
    var timerEl, splitsEl, pbEl;
    var pb = loadPB();
    var currentSplits = []; // this run's recorded times, filled in order as crossed
    var nextSplitIdx = 0; // 0-based index into 0..NUM_SPLITS-1

    function renderPB() {
        if (!pbEl) return;
        var haveFinal = pb && pb[NUM_SPLITS - 1] !== undefined;
        pbEl.textContent = 'PB: ' + (haveFinal ? formatTime(pb[NUM_SPLITS - 1]) : '--:--.---');
    }

    function renderSplits() {
        if (!splitsEl) return;
        var rows = [];
        for (var i = 0; i < NUM_SPLITS; i++) {
            var meters = (i + 1) * SPLIT_INTERVAL_M;
            var have = currentSplits[i] !== undefined;
            var timeStr = have ? formatTime(currentSplits[i]) : '--:--.---';
            var deltaStr = '';
            var deltaClass = '';
            if (have && pb && pb[i] !== undefined) {
                var delta = currentSplits[i] - pb[i];
                deltaStr = formatDelta(delta);
                deltaClass = delta <= 0 ? 'delta-ahead' : 'delta-behind';
            }
            rows.push(
                '<div class="split-row' + (have ? '' : ' pending') + '">' +
                    '<span>' + meters + 'm</span>' +
                    '<span>' + timeStr + '</span>' +
                    '<span class="' + deltaClass + '">' + deltaStr + '</span>' +
                '</div>'
            );
        }
        splitsEl.innerHTML = rows.join('');
    }

    function resetRun(curTick) {
        runStartTick = curTick;
        runStartWallClock = Date.now();
        frozen = false;
        currentSplits = [];
        nextSplitIdx = 0;
        if (timerEl) timerEl.classList.remove('won', 'fallen');
        renderSplits();
    }

    function tick() {
        // Never let one bad frame kill the interval - an uncaught exception inside a
        // setInterval callback doesn't stop future firings (unlike a self-rescheduling
        // requestAnimationFrame loop would), but it's still better to log than silently
        // skip a frame's update.
        try {
            var curTick = window.physicsTick || 0;
            var terminal = !!window.terminal;
            var posx = (window.state && window.state.posx) || 0;

            // Two independent reset signals, since neither alone covers every case:
            //  - terminal going true -> false catches a fall/win followed by reset()
            //    (see QWOP.min.js's reset(), which sets window.terminal = false).
            //  - posx snapping back near 0 catches a Python-side TimeLimit truncation
            //    (max_ep_len reached with no fall/win) - the browser has no concept of
            //    that at all, so window.terminal never goes true in that case, but
            //    reset() still runs (Python always calls env.reset() next) and always
            //    resets the runner's position, regardless of why the episode ended.
            var terminalReset = !terminal && (prevTerminal === null || prevTerminal === true);
            var posxReset = posx < RESET_POSX_THRESHOLD && prevPosx >= RESET_POSX_THRESHOLD;
            if (terminalReset || posxReset) {
                resetRun(curTick);
            }
            prevTerminal = terminal;
            prevPosx = posx;

            if (!frozen) {
                // See the file-level comment for why these differ: real-time mode uses
                // genuine wall-clock time (physicsTick-based timing runs fast there,
                // ~1.2x, since each tick always advances the simulation by a fixed
                // 0.04s regardless of how much real time actually elapsed since the
                // last one); lock-step mode has no meaningful wall-clock relationship
                // at all, so physicsTick * PHYSICS_DT is the only sensible clock there.
                var elapsed = isRealtime
                    ? (Date.now() - runStartWallClock) / 1000
                    : (curTick - runStartTick) * PHYSICS_DT;
                if (timerEl) timerEl.textContent = formatTime(elapsed);

                // A single tick can in principle cross more than one 10m boundary
                // (coarse ticks_per_step), so loop rather than a single if - each
                // newly-crossed split gets this tick's elapsed time, which is the
                // best resolution available at that ticks_per_step.
                while (nextSplitIdx < NUM_SPLITS && posx >= (nextSplitIdx + 1) * SPLIT_INTERVAL_M) {
                    currentSplits[nextSplitIdx] = elapsed;
                    nextSplitIdx += 1;
                    renderSplits();
                }

                if (terminal) {
                    frozen = true;
                    var won = window.state && window.state.won;
                    if (timerEl) timerEl.classList.add(won ? 'won' : 'fallen');
                    // Only a completed run (all 10 splits crossed) is eligible to set a
                    // new PB - a fast-but-fallen attempt never overwrites a slower-but-
                    // finished one.
                    if (won && currentSplits.length === NUM_SPLITS) {
                        // currentSplits[NUM_SPLITS - 1] was stamped when posx crossed
                        // 100m, which happens before the runner necessarily clears the
                        // sand pit and lands (the actual win). `elapsed` here is this
                        // tick's time - the moment `won`/`terminal` actually fired - so
                        // overwrite the last split with it for both the PB comparison
                        // and the on-screen 100m row.
                        currentSplits[NUM_SPLITS - 1] = elapsed;
                        renderSplits();
                        var finalTime = elapsed;
                        var pbFinal = pb ? pb[NUM_SPLITS - 1] : null;
                        if (pbFinal == null || finalTime < pbFinal) {
                            pb = currentSplits.slice();
                            savePB(pb);
                            renderPB();
                        }
                    }
                }
            }
        } catch (e) {
            console.error('[speedrun-timer] tick() error:', e);
        }
    }

    function init() {
        try {
            timerEl = document.getElementById('speedrun-timer');
            splitsEl = document.getElementById('speedrun-splits');
            pbEl = document.getElementById('speedrun-pb');
            if (!timerEl) console.error('[speedrun-timer] #speedrun-timer not found in DOM - is index.html stale (cached)?');
            if (!splitsEl) console.error('[speedrun-timer] #speedrun-splits not found in DOM - is index.html stale (cached)?');
            if (!pbEl) console.error('[speedrun-timer] #speedrun-pb not found in DOM - is index.html stale (cached)?');
            renderSplits();
            renderPB();
        } catch (e) {
            console.error('[speedrun-timer] init() error:', e);
        }
        // Always start the loop, even if the above failed - a broken splits panel
        // shouldn't also prevent the timer (or vice versa) from ever updating.
        //
        // setInterval, NOT requestAnimationFrame: ws.js completely hijacks
        // window.requestAnimationFrame (see its comment above __qwopFrameCallback) into
        // a single-slot callback store for its own lock-step/real-time physics driving -
        // it does not schedule anything with the browser at all. Calling the real
        // rAF-shaped API here doesn't register an independent listener - it *overwrites*
        // whatever ws.js/QWOP.min.js had stored in that one slot, which was silently
        // replacing the actual physics-stepping callback with this timer and freezing
        // the game (physicsTick stuck at 0 forever) the moment this script ran, in any
        // mode. setInterval is a completely separate, un-hijacked browser API.
        setInterval(tick, 33);
    }

    // DOMContentLoaded may have already fired by the time this script runs (e.g. if
    // something delays script execution past that point) - readyState covers that
    // case, since the event would never fire again for a listener added after it.
    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
