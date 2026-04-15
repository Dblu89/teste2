"""
SMC OPTIMIZER v5 - PROFESSIONAL
WDO Dolar Mini B3 - 32 vCPUs
6 Estrategias SMC + Grid 15.000 combos
"""

import pandas as pd
import numpy as np
import json, sys, os, time, warnings, itertools
from datetime import datetime
from joblib import Parallel, delayed
from smartmoneyconcepts import smc as SMC
warnings.filterwarnings("ignore")

CSV_PATH   = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output"
N_CORES    = 32
CAPITAL    = 50_000.0
MULT_WDO   = 10.0
COMISSAO   = 5.0
SLIPPAGE   = 2.0

GRID = {
    "swing_length": [3, 5, 7, 10],
    "rr_min":       [1.5, 2.0, 2.5, 3.0],
    "atr_mult_sl":  [0.3, 0.5, 0.7, 1.0],
    "poi_janela":   [10, 20, 40, 80],
    "choch_janela": [10, 20, 40, 80],
    "estrategia":   [1, 2, 3, 4, 5, 6],
    "close_break":  [True, False],
}

# 4x4x4x4x4x6x2 = 12.288 combos

MIN_TRADES = 20
MIN_PF     = 0.0
MAX_DD     = -99.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

def carregar():
    print(f"[DATA] Carregando {CSV_PATH}…")
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["open","high","low","close","volume"]].copy()
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna()
    df = df[df["close"] > 0]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] OK {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}")
    return df

def preparar_smc(df, swing_length=5, close_break=True):
    swings    = SMC.swing_highs_lows(df, swing_length=swing_length)
    estrutura = SMC.bos_choch(df, swings, close_break=close_break)
    fvg       = SMC.fvg(df)
    ob        = SMC.ob(df, swings)

    try:
        liq = SMC.liquidity(df, swings)
        liq_v   = liq["Liquidity"].values if "Liquidity" in liq.columns else np.zeros(len(df))
        liq_sw  = liq["Swept"].values if "Swept" in liq.columns else np.full(len(df), np.nan)
    except Exception:
        liq_v  = np.zeros(len(df))
        liq_sw = np.full(len(df), np.nan)

    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    tr  = pd.concat([h-l, (h-cp).abs(), (l-cp).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    atr_ma      = atr.rolling(20).mean()
    atr_exp     = (atr > atr_ma).fillna(False)

    swing_high  = df["high"].rolling(50).max()
    swing_low   = df["low"].rolling(50).min()
    mid         = swing_low + (swing_high - swing_low) * 0.5
    premium     = (df["close"] > mid).fillna(False)
    discount    = (df["close"] < mid).fillna(False)

    london      = (df.index.hour >= 9)  & (df.index.hour < 12)
    ny          = (df.index.hour >= 13) & (df.index.hour < 17)

    r = df.copy()
    r["choch"]     = pd.to_numeric(pd.Series(estrutura["CHOCH"].values, index=df.index), errors="coerce").fillna(0)
    r["bos"]       = pd.to_numeric(pd.Series(estrutura["BOS"].values, index=df.index), errors="coerce").fillna(0)
    r["fvg"]       = pd.to_numeric(pd.Series(fvg["FVG"].values, index=df.index), errors="coerce").fillna(0)
    r["fvg_top"]   = pd.Series(fvg["Top"].values, index=df.index)
    r["fvg_bot"]   = pd.Series(fvg["Bottom"].values, index=df.index)
    r["ob"]        = pd.to_numeric(pd.Series(ob["OB"].values, index=df.index), errors="coerce").fillna(0)
    r["ob_top"]    = pd.Series(ob["Top"].values, index=df.index)
    r["ob_bot"]    = pd.Series(ob["Bottom"].values, index=df.index)
    r["liq"]       = pd.Series(liq_v, index=df.index)
    r["liq_swept"] = pd.Series(liq_sw, index=df.index)
    r["atr"]       = atr
    r["atr_exp"]   = atr_exp.astype(float)
    r["premium"]   = premium.astype(float)
    r["discount"]  = discount.astype(float)
    r["london"]    = london.astype(float)
    r["ny"]        = ny.astype(float)
    return r

def backtest(df_ind, rr_min=2.0, atr_mult_sl=0.5,
    poi_janela=20, choch_janela=20,
    estrategia=1, capital=CAPITAL):

    trades = []
    equity = [capital]
    cap    = capital
    em_pos = False
    trade  = None

    fvgs_bull, fvgs_bear = [], []
    obs_bull,  obs_bear  = [], []
    ult_choch_bull = ult_choch_bear = -9999
    ult_liq_bull   = ult_liq_bear   = -9999

    cols = {c: i for i, c in enumerate(df_ind.columns)}
    arr  = df_ind.values

    def v(row, col):
        return row[cols[col]]

    for i in range(20, len(df_ind)):
        row   = arr[i]
        close = float(v(row, "close"))
        atr   = float(v(row, "atr"))
        if np.isnan(atr) or atr <= 0:
            atr = 5.0

        # Gerenciar posicao
        if em_pos and trade:
            d, sl, tp, en = trade["d"], trade["sl"], trade["tp"], trade["entry"]
            lo = float(v(row, "low"))
            hi = float(v(row, "high"))
            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)
            if hit_sl or hit_tp:
                saida = sl if hit_sl else tp
                pts   = (saida - en) * d
                brl   = pts * MULT_WDO - COMISSAO - SLIPPAGE * MULT_WDO * 0.5
                cap  += brl
                equity.append(round(cap, 2))
                trade.update({
                    "saida": round(saida,2), "pnl_pts": round(pts,2),
                    "pnl_brl": round(brl,2),
                    "resultado": "WIN" if hit_tp else "LOSS",
                    "saida_dt": str(df_ind.index[i])[:16]
                })
                trades.append(trade)
                em_pos = False; trade = None
            continue

        # Coletar sinais
        choch = float(v(row, "choch"))
        if choch == 1:
            ult_choch_bull = i
            fvgs_bull.clear(); obs_bull.clear()
        elif choch == -1:
            ult_choch_bear = i
            fvgs_bear.clear(); obs_bear.clear()

        liq_v  = float(v(row, "liq"))
        liq_sw = v(row, "liq_swept")
        if not np.isnan(liq_sw):
            if liq_v == -1: ult_liq_bull = i
            if liq_v == 1:  ult_liq_bear = i

        fvg_v = float(v(row, "fvg"))
        if fvg_v == 1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bull.append({"top": float(v(row,"fvg_top")), "bot": float(v(row,"fvg_bot")), "tipo":"FVG"})
        elif fvg_v == -1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bear.append({"top": float(v(row,"fvg_top")), "bot": float(v(row,"fvg_bot")), "tipo":"FVG"})

        ob_v = float(v(row, "ob"))
        if ob_v == 1 and not np.isnan(v(row, "ob_top")):
            obs_bull.append({"top": float(v(row,"ob_top")), "bot": float(v(row,"ob_bot")), "tipo":"OB"})
        elif ob_v == -1 and not np.isnan(v(row, "ob_top")):
            obs_bear.append({"top": float(v(row,"ob_top")), "bot": float(v(row,"ob_bot")), "tipo":"OB"})

        fvgs_bull = fvgs_bull[-poi_janela:]
        fvgs_bear = fvgs_bear[-poi_janela:]
        obs_bull  = obs_bull[-poi_janela:]
        obs_bear  = obs_bear[-poi_janela:]

        # Filtros por estrategia
        premium = float(v(row, "premium")) == 1
        discount= float(v(row, "discount")) == 1
        london  = float(v(row, "london")) == 1
        ny      = float(v(row, "ny")) == 1
        atr_exp = float(v(row, "atr_exp")) == 1

        if estrategia == 2:
            pode_bull = discount
            pode_bear = premium
        elif estrategia == 3:
            pode_bull = pode_bear = (london or ny)
        elif estrategia == 5:
            pode_bull = (i - ult_liq_bull) <= choch_janela
            pode_bear = (i - ult_liq_bear) <= choch_janela
        elif estrategia == 6:
            pode_bull = pode_bear = atr_exp
        else:
            pode_bull = pode_bear = True

        sinal = poi = None

        # Estrategia 4: confluencia OB + FVG
        if estrategia == 4:
            if (i - ult_choch_bull) <= choch_janela and pode_bull:
                for ob in obs_bull:
                    for fv in fvgs_bull:
                        ot = min(ob["top"], fv["top"])
                        ob2 = max(ob["bot"], fv["bot"])
                        if ot > ob2 and ob2 <= close <= ot:
                            sinal = 1
                            poi   = {"top": ot, "bot": ob2, "tipo": "OB+FVG"}
                            break
                    if sinal: break
            if sinal is None and (i - ult_choch_bear) <= choch_janela and pode_bear:
                for ob in obs_bear:
                    for fv in fvgs_bear:
                        ot = min(ob["top"], fv["top"])
                        ob2 = max(ob["bot"], fv["bot"])
                        if ot > ob2 and ob2 <= close <= ot:
                            sinal = -1
                            poi   = {"top": ot, "bot": ob2, "tipo": "OB+FVG"}
                            break
                    if sinal: break
        else:
            if (i - ult_choch_bull) <= choch_janela and pode_bull:
                for p in reversed(fvgs_bull + obs_bull):
                    if close <= p["top"] and close >= p["bot"] * 0.998:
                        sinal = 1; poi = p; break
            if sinal is None and (i - ult_choch_bear) <= choch_janela and pode_bear:
                for p in reversed(fvgs_bear + obs_bear):
                    if close >= p["bot"] and close <= p["top"] * 1.002:
                        sinal = -1; poi = p; break

        if sinal is None or poi is None:
            continue

        slip = SLIPPAGE * 0.5
        if sinal == 1:
            entry = close + slip
            sl    = poi["bot"] - atr * atr_mult_sl
        else:
            entry = close - slip
            sl    = poi["top"] + atr * atr_mult_sl

        risk = abs(entry - sl)
        if risk <= 0: continue
        tp   = entry + sinal * risk * rr_min
        if abs(tp - entry) / risk < rr_min * 0.95: continue
        if risk * MULT_WDO / cap > 0.05: continue

        em_pos = True
        trade  = {
            "entry_dt":    str(df_ind.index[i])[:16],
            "d":           sinal,
            "entry":       round(entry, 2),
            "sl":          round(sl, 2),
            "tp":          round(tp, 2),
            "rr":          round(abs(tp-entry)/risk, 2),
            "poi_tipo":    poi.get("tipo","?"),
            "estrategia":  estrategia,
            "capital_pre": round(cap, 2),
        }

    if em_pos and trade:
        last = float(arr[-1][cols["close"]])
        pts  = (last - trade["entry"]) * trade["d"]
        brl  = pts * MULT_WDO - COMISSAO
        cap += brl
        trade.update({"saida": last, "pnl_pts": round(pts,2),
                      "pnl_brl": round(brl,2), "resultado": "ABERTO",
                      "saida_dt": str(df_ind.index[-1])[:16]})
        trades.append(trade)
        equity.append(round(cap, 2))

    return trades, equity

def metricas(trades, equity, capital=CAPITAL):
    fechados = [t for t in trades if t.get("resultado") in ("WIN","LOSS")]
    if len(fechados) < MIN_TRADES:
        return None
    df_t  = pd.DataFrame(fechados)
    wins  = df_t[df_t["resultado"] == "WIN"]
    loses = df_t[df_t["resultado"] == "LOSS"]
    n     = len(df_t)
    wr    = len(wins) / n * 100
    avg_w = float(wins["pnl_brl"].mean())  if len(wins)  else 0
    avg_l = float(loses["pnl_brl"].mean()) if len(loses) else -1
    pf    = abs(float(wins["pnl_brl"].sum()) / float(loses["pnl_brl"].sum())) if float(loses["pnl_brl"].sum()) != 0 else 9999
    pnl   = float(df_t["pnl_brl"].sum())

    eq   = pd.Series(equity)
    peak = eq.cummax()
    mdd  = float(((eq - peak) / peak * 100).min())
    if mdd < MAX_DD: return None

    rets    = eq.pct_change().dropna()
    sharpe  = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    neg     = rets[rets < 0]
    sortino = float(rets.mean()/neg.std()*np.sqrt(252)) if len(neg)>1 else 0

    return {
        "total_trades":   n,
        "wins":           int(len(wins)),
        "losses":         int(len(loses)),
        "win_rate":       round(wr, 2),
        "profit_factor":  round(pf, 3),
        "sharpe_ratio":   round(sharpe, 3),
        "sortino_ratio":  round(sortino, 3),
        "avg_win_brl":    round(avg_w, 2),
        "avg_loss_brl":   round(avg_l, 2),
        "expectancy_brl": round((wr/100*avg_w)+((1-wr/100)*avg_l), 2),
        "total_pnl_brl":  round(pnl, 2),
        "retorno_pct":    round(pnl/capital*100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "capital_final":  round(capital+pnl, 2),
        "poi_tipos":      df_t["poi_tipo"].value_counts().to_dict(),
    }

def worker(params):
    sw, rr, am, pj, cj, est, cb = params
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
        df.columns = [c.lower().strip() for c in df.columns]
        df = df[df.index.dayofweek < 5]
        df = df[(df.index.hour >= 9) & (df.index.hour < 18)].sort_index()
        n = len(df)
        df = df.iloc[:int(n*0.7)]

        df_ind = preparar_smc(df, swing_length=sw, close_break=cb)
        trades, equity = backtest(df_ind, rr_min=rr, atr_mult_sl=am,
                                  poi_janela=pj, choch_janela=cj,
                                  estrategia=est)
        m = metricas(trades, equity)
        if not m or m["profit_factor"] < MIN_PF:
            return None

        pf_s = min(m["profit_factor"], 10) / 10
        sh_s = min(max(m["sharpe_ratio"], 0), 8) / 8
        so_s = min(max(m["sortino_ratio"], 0), 10) / 10
        wr_s = m["win_rate"] / 100
        tr_s = min(m["total_trades"], 500) / 500
        score = pf_s*0.35 + sh_s*0.25 + so_s*0.15 + wr_s*0.15 + tr_s*0.10

        return {
            "swing_length": sw, "rr_min": rr, "atr_mult_sl": am,
            "poi_janela": pj, "choch_janela": cj,
            "estrategia": est, "close_break": cb,
            "score": round(float(score), 6), **m
        }
    except Exception:
        return None

def grid_search(mini=False):
    g = GRID
    combos = list(itertools.product(
        g["swing_length"], g["rr_min"], g["atr_mult_sl"],
        g["poi_janela"], g["choch_janela"],
        g["estrategia"], g["close_break"]
    ))

    if mini:
        combos = [(5,2.0,0.5,20,20,e,True) for e in range(1,7)]
        print(f"\n[GRID] Modo MINI - {len(combos)} combos (1 por estrategia)")
    else:
        print(f"\n[GRID] Grid Search v5 - {len(combos):,} combos | {N_CORES} cores")
        print(f"       6 estrategias SMC | parametros expandidos")

    t0 = time.time()

    if mini:
        resultados = [r for r in [worker(c) for c in combos] if r and "score" in r]
    else:
        print(f"[GRID] Iniciando joblib loky...")
        resultados = Parallel(n_jobs=N_CORES, backend="loky", verbose=5)(
            delayed(worker)(c) for c in combos
        )
        resultados = [r for r in resultados if r and "score" in r]

    elapsed = time.time() - t0
    resultados.sort(key=lambda x: -x["score"])
    print(f"\n[GRID] OK {elapsed:.1f}s | {len(resultados):,} validos de {len(combos):,}")

    if resultados:
        exibir_top(resultados)
        exibir_por_estrategia(resultados)

    return {
        "melhor":       resultados[0] if resultados else None,
        "top20":        resultados[:20],
        "total_combos": len(combos),
        "validos":      len(resultados),
        "elapsed_s":    round(elapsed, 1),
    }

def exibir_top(resultados, n=20):
    nomes = {1:"Base",2:"Disc/Prem",3:"London/NY",4:"OB+FVG",5:"Liq.Sweep",6:"ATR Filt"}
    print(f"\n{'='*82}")
    print(f"  TOP {min(n,len(resultados))} - SMC OPTIMIZER v5")
    print(f"{'='*82}")
    print(f"  {'#':>2} {'EST':>10} {'SW':>3} {'RR':>4} {'ATR':>4} {'POI':>4} {'CHoCH':>5} "
        f"{'PF':>6} {'Sharpe':>7} {'WR%':>6} {'Trades':>7} {'DD%':>6} {'Score':>7}")
    print(f"  {'-'*80}")
    for i, r in enumerate(resultados[:n], 1):
        star = "*" if i == 1 else " "
        print(f"  {star}{i:>2} {nomes.get(r['estrategia'],'?'):>10} "
            f"{r['swing_length']:>3} {r['rr_min']:>4} {r['atr_mult_sl']:>4} "
            f"{r['poi_janela']:>4} {r['choch_janela']:>5} "
            f"{r['profit_factor']:>6.3f} {r['sharpe_ratio']:>7.3f} "
            f"{r['win_rate']:>6.1f} {r['total_trades']:>7} "
            f"{r['max_drawdown_pct']:>6.1f} {r['score']:>7.4f}")
    print(f"{'='*82}")

def exibir_por_estrategia(resultados):
    nomes = {1:"Base",2:"Disc/Prem",3:"London/NY",4:"OB+FVG",5:"Liq.Sweep",6:"ATR Filt"}
    print(f"\n  MELHOR POR ESTRATEGIA:")
    for est in range(1, 7):
        subset = [r for r in resultados if r["estrategia"] == est]
        if subset:
            b = subset[0]
            print(f"  {nomes[est]:>12}: PF={b['profit_factor']:.3f} | "
                f"WR={b['win_rate']:.1f}% | Trades={b['total_trades']} | "
                f"Score={b['score']:.4f}")
        else:
            print(f"  {nomes[est]:>12}: sem resultados validos")

def walk_forward(df, config, n_splits=5):
    print(f"\n[WF] Walk-Forward: {n_splits} splits")
    resultados = []
    step = len(df) // n_splits
    for i in range(n_splits - 1):
        ini   = i * step
        fim   = (i + 2) * step
        split = ini + int((fim - ini) * 0.7)
        df_tr = df.iloc[ini:split]
        df_te = df.iloc[split:fim]
        if len(df_tr) < 500 or len(df_te) < 100: continue
        d0 = df_tr.index[0].strftime("%Y-%m-%d")
        d1 = df_tr.index[-1].strftime("%Y-%m-%d")
        d2 = df_te.index[0].strftime("%Y-%m-%d")
        d3 = df_te.index[-1].strftime("%Y-%m-%d")
        try:
            bt = {k: v for k, v in config.items()
                if k in ["rr_min","atr_mult_sl","poi_janela","choch_janela","estrategia"]}
            df_tr_ind = preparar_smc(df_tr, swing_length=config["swing_length"],
                close_break=config["close_break"])
            df_te_ind = preparar_smc(df_te, swing_length=config["swing_length"],
                close_break=config["close_break"])
            tr_t, tr_e = backtest(df_tr_ind, **bt)
            te_t, te_e = backtest(df_te_ind, **bt)
            m_tr = metricas(tr_t, tr_e) or {}
            m_te = metricas(te_t, te_e) or {}
            print(f"\n  Split {i+1}: Train [{d0}->{d1}] | Test [{d2}->{d3}]")
            if m_tr: print(f"    TRAIN -> PF:{m_tr.get('profit_factor',0)} | WR:{m_tr.get('win_rate',0)}% | PnL:R${m_tr.get('total_pnl_brl',0):,.0f}")
            if m_te: print(f"    TEST  -> PF:{m_te.get('profit_factor',0)} | WR:{m_te.get('win_rate',0)}% | PnL:R${m_te.get('total_pnl_brl',0):,.0f}")
            resultados.append({"split":i+1,"train":m_tr,"test":m_te,
                "train_start":d0,"train_end":d1,"test_start":d2,"test_end":d3})
        except Exception as e:
            print(f"    Split {i+1} erro: {e}")
    lucr = sum(1 for r in resultados if r["test"].get("total_pnl_brl",0) > 0)
    print(f"\n[WF] OK {lucr}/{len(resultados)} splits lucrativos")
    return resultados

def monte_carlo(trades, n_sim=2000, capital=CAPITAL):
    fechados = [t for t in trades if t.get("resultado") in ("WIN","LOSS")]
    if len(fechados) < 10: return {}
    print(f"\n[MC] Monte Carlo: {n_sim:,} simulacoes…")
    pnls = np.array([t["pnl_brl"] for t in fechados])
    np.random.seed(42)
    rets, dds, ruinas = [], [], 0
    for _ in range(n_sim):
        seq = np.random.choice(pnls, size=len(pnls), replace=True)
        eq  = np.insert(capital + np.cumsum(seq), 0, capital)
        pk  = np.maximum.accumulate(eq)
        dds.append(((eq-pk)/pk*100).min())
        rets.append((eq[-1]-capital)/capital*100)
        if eq[-1] < capital*0.5: ruinas+=1
    rf, md = np.array(rets), np.array(dds)
    res = {
        "n_simulacoes":    n_sim,
        "prob_lucro_pct":  round(float((rf>0).mean()*100),1),
        "retorno_mediana": round(float(np.median(rf)),2),
        "retorno_p10":     round(float(np.percentile(rf,10)),2),
        "retorno_p90":     round(float(np.percentile(rf,90)),2),
        "dd_mediano":      round(float(np.median(md)),2),
        "dd_pior":         round(float(md.min()),2),
        "prob_ruina_pct":  round(float(ruinas/n_sim*100),2),
        "prob_dd_10":      round(float((md<-10).mean()*100),1),
        "prob_dd_20":      round(float((md<-20).mean()*100),1),
    }
    print(f"[MC] OK Prob.lucro:{res['prob_lucro_pct']}% | DD mediano:{res['dd_mediano']}% | Ruina:{res['prob_ruina_pct']}%")
    return res

def relatorio(m, mc=None, titulo="RESULTADO", config=None):
    if not m: return
    nomes = {1:"Base",2:"Disc/Prem",3:"London/NY",4:"OB+FVG",5:"Liq.Sweep",6:"ATR Filt"}
    sep = "=" * 62
    def L(lb, vl): print(f"  {lb:<32} {str(vl):>26}")
    print(f"\n{sep}")
    print(f"  SMC OPTIMIZER v5 – {titulo}")
    print(sep)
    if config:
        print(f"  Estrategia: {nomes.get(config.get('estrategia',1),'?')} | "
            f"SW={config.get('swing_length')} RR={config.get('rr_min')} "
            f"ATR={config.get('atr_mult_sl')} POI={config.get('poi_janela')} "
            f"CHoCH={config.get('choch_janela')}")
        print(f"  {'-'*58}")
    L("Total Trades",      m["total_trades"])
    L("Wins / Losses",     f"{m['wins']} W  /  {m['losses']} L")
    L("Win Rate",          f"{m['win_rate']}%")
    L("Profit Factor",     m["profit_factor"])
    L("Sharpe Ratio",      m["sharpe_ratio"])
    L("Sortino Ratio",     m["sortino_ratio"])
    L("Expectancy",        f"R$ {m['expectancy_brl']:,.2f}")
    L("Total PnL",         f"R$ {m['total_pnl_brl']:,.2f}")
    L("Retorno %",         f"{m['retorno_pct']}%")
    L("Max Drawdown",      f"{m['max_drawdown_pct']}%")
    L("Capital Final",     f"R$ {m['capital_final']:,.2f}")
    if mc:
        print(f"  {'-'*58}")
        print(f"  MONTE CARLO ({mc['n_simulacoes']:,} simulacoes)")
        L("Prob. Lucro",       f"{mc['prob_lucro_pct']}%")
        L("Retorno Mediana",   f"{mc['retorno_mediana']}%")
        L("Retorno P10/P90",   f"{mc['retorno_p10']}% / {mc['retorno_p90']}%")
        L("DD Mediano",        f"{mc['dd_mediano']}%")
        L("DD Pior",           f"{mc['dd_pior']}%")
        L("Risco Ruina",       f"{mc['prob_ruina_pct']}%")
        L("Prob DD > 10%",     f"{mc['prob_dd_10']}%")
        L("Prob DD > 20%",     f"{mc['prob_dd_20']}%")
    print(sep)

def main():
    MINI = "–mini" in sys.argv

    print("=" * 70)
    print("  SMC OPTIMIZER v5 -- 6 ESTRATEGIAS SMC | 12.288 COMBOS")
    print("  WDO Dolar Mini B3 | Biblioteca oficial smartmoneyconcepts")
    print("=" * 70)

    df = carregar()
    split  = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]
    print(f"  In-sample : {len(df_ins):,} | {df_ins.index[0].date()} -> {df_ins.index[-1].date()}")
    print(f"  Out-sample: {len(df_oos):,} | {df_oos.index[0].date()} -> {df_oos.index[-1].date()}")

    if MINI:
        print("\n[MINI] Testando 6 combos (1 por estrategia)...")
        grid = grid_search(mini=True)
        if grid["validos"] > 0:
            print(f"\nOK {grid['validos']} estrategia(s) valida(s)!")
        else:
            print("AVISO: Nenhum resultado valido. Verifique MIN_TRADES.")
        return

    grid = grid_search(mini=False)

    if not grid["melhor"]:
        print("\n[ERRO] Nenhuma configuracao valida.")
        return

    melhor = grid["melhor"]
    CONFIG = {k: melhor[k] for k in
              ["swing_length","rr_min","atr_mult_sl",
               "poi_janela","choch_janela","estrategia","close_break"]}

    print("\n[OOS] Backtest Out-of-Sample...")
    bt = {k: v for k, v in CONFIG.items()
          if k in ["rr_min","atr_mult_sl","poi_janela","choch_janela","estrategia"]}
    df_oos_ind = preparar_smc(df_oos, swing_length=CONFIG["swing_length"],
                               close_break=CONFIG["close_break"])
    t_oos, e_oos = backtest(df_oos_ind, **bt)
    m_oos = metricas(t_oos, e_oos)
    relatorio(m_oos, titulo="OUT-OF-SAMPLE", config=CONFIG)

    print("\n[FULL] Backtest dataset completo...")
    df_full_ind = preparar_smc(df, swing_length=CONFIG["swing_length"],
                                close_break=CONFIG["close_break"])
    t_full, e_full = backtest(df_full_ind, **bt)
    m_full = metricas(t_full, e_full)

    wf = walk_forward(df, CONFIG, n_splits=5)
    mc = monte_carlo(t_full, n_sim=2000)
    relatorio(m_full, mc, titulo="COMPLETO + MONTE CARLO", config=CONFIG)

    out = {
        "versao": "v5",
        "gerado_em": datetime.now().isoformat(),
        "config_melhor": CONFIG,
        "metricas_full": m_full,
        "metricas_oos":  m_oos,
        "walk_forward":  wf,
        "monte_carlo":   mc,
        "grid_top20":    grid.get("top20",[]),
        "grid_stats": {
            "total_combos": grid.get("total_combos"),
            "validos":      grid.get("validos"),
            "elapsed_s":    grid.get("elapsed_s"),
        },
        "trades":      t_full,
        "equity_curve":e_full,
    }
    path = f"{OUTPUT_DIR}/resultado_v5.json"
    with open(path,"w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[OK] Salvo em {path}")
    nomes = {1:"Base",2:"Disc/Prem",3:"London/NY",4:"OB+FVG",5:"Liq.Sweep",6:"ATR Filt"}
    print(f"\nCONCLUIDO!")
    print(f"  Estrategia: {nomes.get(melhor['estrategia'],'?')}")
    print(f"  PF={melhor['profit_factor']} | WR={melhor['win_rate']}% | "
          f"Score={melhor['score']} | Trades={melhor['total_trades']}")

if __name__ == "__main__":
    main()
