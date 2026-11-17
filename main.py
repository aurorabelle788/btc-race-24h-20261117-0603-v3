"""btc-race-24h-20261117-0603 v3: mean-reversion dip buyer.

Strategy, tuned on the frozen 1,440-bar dev window (1,788 + 400 + 200 variants
tested across three rounds of backtests against the kit harness's exact fill
model: 4 bps fee, 2 bps slippage, constant-product impact vs open depth):
  * Entry: flat and the latest close is at least 0.8% below the 90-bar simple
    moving average of closes. A slow mean lags the price, so the trigger fires
    only once a dip has deepened below the average -- a better entry than
    buying the first dip off a rolling high. Buy 99% of cash.
  * Exit: take profit at +2.0% from the fill price, stop loss at -1.5%,
    or a 480-bar max hold, whichever comes first.
  * Safety: if equity drawdown from its peak reaches 20%, sell everything
    and stop trading for the rest of the window.
  * One position at a time, long only, at most one order per bar, every order
    capped at 1,000 bps of slippage from the latest close.

Dev-window result (exact kit fill model, run twice, byte-identical action
logs): net PnL +$38.65, 5 trades, 3 wins, max drawdown 0.72%. Dev gate PASS.

Deterministic: no clock, no randomness, no environment reads, no files.
Only the standard library plus the vendored protocol shim (vendor/).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# Strategy parameters (tuned on the dev window; see RESULTS-v3.md).
MEAN_WINDOW = 90        # bars in the moving average
MEAN_DROP = 0.008       # buy when close is this far below the mean
SIZE_BPS = 9900         # buy this share of cash (99%: margin under the 100% weight cap)
TAKE_PROFIT = 0.02      # sell at +2.0% from the fill price
STOP_LOSS = 0.015       # sell at -1.5% from the fill price
MAX_HOLD = 480          # bars; sell at the max hold even if neither TP nor SL hit
DD_HALT = 0.20          # sell everything and stop if equity drawdown hits 20%
MAX_SLIPPAGE_BPS = 1000
MIN_NOTIONAL = 1_000_000  # 1 USDC, the harness minimum for a buy

# Mutable run state (reset per execution; the harness runs the bot fresh).
closes = []             # closes seen so far, in bar order
entry_px = None         # fill price of the open position, else None
entry_bar = -1          # batch seq at which the open position filled
peak_equity = 0         # highest equity seen, for the drawdown halt
halted = False          # True once the drawdown halt has fired
order_no = 0            # running counter for unique client ids
sent_buys = {}          # client_id -> True for buy orders awaiting a fill
orders_sent = 0


def _next_id(prefix):
    global order_no
    order_no += 1
    return "%s%d" % (prefix, order_no)


def on_batch(state, batch):
    global entry_px, entry_bar, peak_equity, halted, orders_sent
    # Learn our fills from this batch before deciding anything. A buy fill
    # sets the entry price and entry bar; a rejected buy leaves us flat.
    for fill in batch.get("fills", []):
        cid = fill.get("client_id")
        if cid in sent_buys and fill.get("status") == "filled":
            entry_px = int(fill["price"])
            entry_bar = state.seq
            del sent_buys[cid]
        elif cid in sent_buys:
            del sent_buys[cid]

    # The bar for this batch (a minute with no sample carries no CBBTC bar:
    # no fresh close, so no decision on that batch).
    bar = None
    for b in batch.get("bars", []):
        if b.get("symbol") == SYMBOL:
            bar = b
            break
    if bar is None:
        return []
    close = int(bar["c"])
    closes.append(close)

    # Drawdown halt: equity comes from the harness account (cash plus the
    # position at its latest close), the same mark the grader uses.
    if state.equity > peak_equity:
        peak_equity = state.equity
    holding = state.positions.get(SYMBOL)
    qty = holding.qty if holding is not None else 0
    if qty and entry_px is None:
        # Safety net: a position exists but no buy fill was seen for it
        # (e.g. a fill reported under an id we did not track). Fall back to
        # the position's cost basis as the reference price so the exit
        # rules below still have something to measure against.
        entry_px = holding.cost * 10 ** 7 // qty
        entry_bar = state.seq
    if qty and peak_equity > 0:
        dd = (peak_equity - state.equity) / peak_equity
        if dd >= DD_HALT:
            halted = True
            cid = _next_id("x")
            orders_sent += 1
            return [{"side": "sell", "symbol": SYMBOL, "qty": qty,
                     "max_slippage_bps": MAX_SLIPPAGE_BPS, "client_id": cid}]
    if halted or state.warming_up:
        return []

    if qty == 0:
        # Entry: flat, at least a full mean window of history, and the close
        # at least MEAN_DROP below the 90-bar mean of closes.
        if len(closes) < MEAN_WINDOW:
            return []
        mean = sum(closes[-MEAN_WINDOW:]) / MEAN_WINDOW
        if close > mean * (1 - MEAN_DROP):
            return []
        notional = state.cash * SIZE_BPS // 10000
        if notional < MIN_NOTIONAL:
            return []
        cid = _next_id("e")
        sent_buys[cid] = True
        orders_sent += 1
        return [{"side": "buy", "symbol": SYMBOL, "notional_quote_units": notional,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS, "client_id": cid}]

    # Exit: take profit, stop loss, or max hold, measured from the fill price.
    held = state.seq - entry_bar
    if (close >= entry_px * (1 + TAKE_PROFIT)
            or close <= entry_px * (1 - STOP_LOSS)
            or held >= MAX_HOLD):
        cid = _next_id("x")
        orders_sent += 1
        return [{"side": "sell", "symbol": SYMBOL, "qty": qty,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS, "client_id": cid}]
    return []


def on_done(state, done):
    sys.stderr.write(
        "done: final cash %d micro-USDC, equity %d, orders sent %d\n"
        % (state.cash, state.equity, orders_sent))


makefun_bars.run(on_batch, on_done=on_done)
