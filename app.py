import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
from scipy import stats
from scipy.stats import zscore as scipy_zscore
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, linear_reset, breaks_cusumolsresid
from statsmodels.stats.stattools import jarque_bera
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(page_title="tahmin.ai | Regresyon Sihirbazı", layout="centered")

st.markdown("""
<style>
    .block-container {max-width: 760px; padding-top: 2rem;}
    .step-box {border-radius: 10px; padding: 14px 18px; margin: 10px 0; font-size: 0.95em;}
    .step-pass {background:#d1e7dd; border-left: 5px solid #0a3622; color:#0a3622;}
    .step-fail {background:#f8d7da; border-left: 5px solid #842029; color:#842029;}
    .step-fix  {background:#fff3cd; border-left: 5px solid #664d03; color:#664d03;}
    .step-info {background:#cfe2ff; border-left: 5px solid #084298; color:#084298;}
</style>
""", unsafe_allow_html=True)

st.title("🧭 Regresyon Tanı Sihirbazı")
st.caption("Adım adım regresyon varsayım kontrolü — her sorun tespit edilir, düzeltilir, sonuç raporlanır.")

# ============================================================
# İndikatör Fonksiyonları
# ============================================================

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_bollinger(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    bb_upper = sma + std_dev * std
    bb_lower = sma - std_dev * std
    return bb_upper, bb_lower, (bb_upper - bb_lower) / sma

def calc_roc(close, period=10):
    return ((close - close.shift(period)) / close.shift(period)) * 100

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calc_adx(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
    atr_s = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def calc_cci(high, low, close, period=20):
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (typical_price - sma_tp) / (0.015 * mean_dev)

def calc_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def calc_cmf(high, low, close, volume, period=20):
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    return (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

def calc_volume_roc(volume, period=10):
    return ((volume - volume.shift(period)) / volume.shift(period)) * 100

def calc_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    direction = typical_price.diff()
    pos_mf = raw_money_flow.where(direction > 0, 0.0)
    neg_mf = raw_money_flow.where(direction < 0, 0.0)
    pos_sum = pos_mf.rolling(window=period).sum()
    neg_sum = neg_mf.rolling(window=period).sum()
    return 100 - (100 / (1 + pos_sum / neg_sum))

def calc_amihud(close, volume):
    ret = np.log(close).diff().abs()
    return ret / volume.replace(0, np.nan)

def calc_mec(close, window=63):
    # T = ret_long_period / ret_short_period = 30 / 5 = 6
    T = 6
    ret_long = np.log(close / close.shift(30))
    ret_short = np.log(close / close.shift(5))
    var_long = ret_long.rolling(window=window).var()
    var_short = ret_short.rolling(window=window).var()
    return var_long / (T * var_short)

def calc_corwin_schultz(high, low):
    sqrt2 = np.sqrt(2)
    denom = 3 - 2 * sqrt2
    log_hl = np.log(high / low)
    log_hl2 = log_hl ** 2
    log_hl_prev = np.log(high.shift(1) / low.shift(1)) ** 2
    beta = log_hl2 + log_hl_prev
    h2 = pd.concat([high.shift(1), high], axis=1).max(axis=1)
    l2 = pd.concat([low.shift(1), low], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha = alpha.clip(lower=0)
    return 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

def calc_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = calc_rsi(close, rsi_period)
    min_rsi = rsi.rolling(window=stoch_period).min()
    max_rsi = rsi.rolling(window=stoch_period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi_k = stoch_rsi.rolling(window=k_smooth).mean() * 100
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_smooth).mean()
    return stoch_rsi_k, stoch_rsi_d

def build_indicators(df):
    c = df["Close"]; h = df["High"]; l = df["Low"]; v = df["Volume"]
    df["EMA_20"] = calc_ema(c, 20)
    df["EMA_50"] = calc_ema(c, 50)
    df["EMA_200"] = calc_ema(c, 200)
    df["RSI"] = calc_rsi(c)
    df["MACD"] = calc_macd(c)[0]
    df["ATR"] = calc_atr(h, l, c)
    df["BB_Upper"], df["BB_Lower"], df["BBW"] = calc_bollinger(c)
    df["Return"] = c.pct_change()
    df["ROC"] = calc_roc(c)
    df["Stoch_K"], df["Stoch_D"] = calc_stochastic(h, l, c)
    df["ADX"] = calc_adx(h, l, c)
    df["Williams_R"] = calc_williams_r(h, l, c)
    df["CCI"] = calc_cci(h, l, c)
    df["OBV"] = calc_obv(c, v)
    df["CMF"] = calc_cmf(h, l, c, v)
    df["Volume_ROC"] = calc_volume_roc(v)
    df["MFI"] = calc_mfi(h, l, c, v)
    df["StochRSI_K"], df["StochRSI_D"] = calc_stoch_rsi(c)
    df["Amihud"] = calc_amihud(c, v)
    df["MEC"] = calc_mec(c)
    df["CS_Spread"] = calc_corwin_schultz(h, l)
    df["Daily_Range"] = h - l
    return df

# ============================================================
# Yardımcı: Adım Kartı
# ============================================================

def step_card(step_no, title, status, detail, fix=None):
    """status: 'pass' | 'fail' | 'fix' | 'info'"""
    icons = {"pass": "✅", "fail": "❌", "fix": "🔧", "info": "ℹ️"}
    css   = {"pass": "step-pass", "fail": "step-fail", "fix": "step-fix", "info": "step-info"}
    icon  = icons.get(status, "")
    cls   = css.get(status, "step-info")
    fix_html = f"<br><b>Uygulanan düzeltme:</b> {fix}" if fix else ""
    st.markdown(f"""
    <div class="step-box {cls}">
        <b>{icon} Adım {step_no} — {title}</b><br>
        {detail}{fix_html}
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Giriş
# ============================================================

col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Sembol", placeholder="AAPL, THYAO.IS, BTC-USD")
    target = st.text_input("Hedef Değişken", value="Close")
with col2:
    start_date = st.date_input("Başlangıç",
                               value=pd.Timestamp("2020-01-01").date(),
                               min_value=pd.Timestamp("1970-01-01").date(),
                               max_value=pd.Timestamp.today().date())
    end_date   = st.date_input("Bitiş",
                               value=pd.Timestamp.today().date(),
                               min_value=pd.Timestamp("1970-01-01").date(),
                               max_value=pd.Timestamp.today().date())

col3, col4, col5 = st.columns(3)
with col3:
    corr_low  = st.slider("|ρ| düşük eşik",  0.05, 0.30, 0.15, 0.01)
with col4:
    corr_high = st.slider("|ρ| yüksek eşik", 0.90, 0.999, 0.995, 0.001, format="%.3f")
with col5:
    vif_thr   = st.slider("VIF eşiği", 5.0, 20.0, 10.0, 0.5)

col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    check_range = st.button("📅 Veri Aralığı", use_container_width=True)
with col_btn2:
    run = st.button("▶ Sihirbazı Başlat", type="primary", use_container_width=True)

if check_range and symbol:
    with st.spinner("Veri aralığı sorgulanıyor..."):
        try:
            _ticker = yf.Ticker(symbol)
            _hist   = _ticker.history(period="max", interval="1d", actions=False)
            if _hist.index.tz is not None:
                _hist.index = _hist.index.tz_localize(None)
            if _hist.empty:
                st.warning("Veri bulunamadı. Sembolü kontrol edin.")
            else:
                _start = _hist.index.min().date()
                _end   = _hist.index.max().date()
                _days  = len(_hist)
                st.info(
                    f"📅 **{symbol.upper()} mevcut veri aralığı** — "
                    f"En eski: `{_start}` · En yeni: `{_end}` · Toplam: `{_days:,}` gün"
                )

                # ── Veri Kalitesi Tanısı ──────────────────────────
                _n_raw        = len(_hist)
                _n_ohlc_all   = int(((_hist["Open"]==_hist["High"]) & (_hist["High"]==_hist["Low"]) & (_hist["Low"]==_hist["Close"])).sum()) if all(c in _hist.columns for c in ["Open","High","Low","Close"]) else 0
                _n_zero_range = int((_hist["High"]==_hist["Low"]).sum()) if all(c in _hist.columns for c in ["High","Low"]) else 0
                _n_zero_vol   = int((_hist["Volume"]==0).sum()) if "Volume" in _hist.columns else 0
                _n_nan        = int(_hist[["Open","High","Low","Close","Volume"]].isnull().any(axis=1).sum()) if all(c in _hist.columns for c in ["Open","High","Low","Close","Volume"]) else 0
                _n_usable     = _n_raw - _n_ohlc_all

                diag_rows = [
                    {"Kontrol": "Ham veri (tüm tarihler)",        "Satır": f"{_n_raw:,}",        "Durum": "ℹ️"},
                    {"Kontrol": "OHLC tümü eşit (donuk fiyat)",   "Satır": f"{_n_ohlc_all:,}",   "Durum": "✅ Temiz" if _n_ohlc_all == 0 else "⚠️ Çıkarılacak"},
                    {"Kontrol": "High = Low (sıfır range)",        "Satır": f"{_n_zero_range:,}", "Durum": "✅ Temiz" if _n_zero_range == 0 else "⚠️ ATR/CS_Spread/BBW bozulabilir"},
                    {"Kontrol": "Volume = 0",                      "Satır": f"{_n_zero_vol:,}",   "Durum": "✅ Temiz" if _n_zero_vol == 0 else "⚠️ Amihud/CMF/MFI bozulabilir"},
                    {"Kontrol": "Boş hücre (herhangi bir OHLCV)",  "Satır": f"{_n_nan:,}",        "Durum": "✅ Temiz" if _n_nan == 0 else "⚠️ dropna ile düşer"},
                    {"Kontrol": "Tahmini kullanılabilir satır",    "Satır": f"{_n_usable:,}",     "Durum": "ℹ️"},
                ]
                diag_df = pd.DataFrame(diag_rows)

                def _dq(val):
                    if not isinstance(val, str): return ""
                    if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
                    if val.startswith("⚠️"): return "background-color:#fff3cd; color:#664d03"
                    return ""

                st.markdown("**🔍 Veri Kalitesi Tanısı**")
                st.dataframe(
                    diag_df.style.map(_dq, subset=["Durum"]),
                    use_container_width=True, hide_index=True
                )

                if _n_zero_range > 0:
                    st.warning(f"⚠️ {_n_zero_range:,} gün High=Low — range-based indikatörler (ATR, CS_Spread, BBW, Daily_Range) bu günlerde 0 üretir. Regresyona girmeden önce VIF/Spearman filtresi bunları elemelidir.")
                if _n_zero_vol > 0:
                    st.warning(f"⚠️ {_n_zero_vol:,} gün Volume=0 — Amihud, CMF, MFI bu günlerde NaN/Inf üretir. dropna ile örneklemden düşer.")

        except Exception as e:
            st.error(f"Sorgu hatası: {e}")
elif check_range and not symbol:
    st.warning("Lütfen önce sembol girin.")

if run and symbol:
    st.divider()
    st.subheader(f"📋 {symbol.upper()} — Tanı Raporu")

    # ── Veri yükle ───────────────────────────────────────────
    with st.spinner("Veri indiriliyor..."):
        try:
            ticker   = yf.Ticker(symbol)
            is_intra = False
            hist     = ticker.history(period="max", interval="1d", actions=False)
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)

            # Sembolün tüm mevcut veri aralığını göster
            if not hist.empty:
                avail_start = hist.index.min().date()
                avail_end   = hist.index.max().date()
                avail_days  = len(hist)
                st.info(
                    f"📅 **{symbol.upper()} mevcut veri aralığı** — "
                    f"En eski: `{avail_start}` · En yeni: `{avail_end}` · "
                    f"Toplam: `{avail_days:,}` gün   "
                    f"_(Seçilen aralık: {start_date} → {end_date})_"
                )

            mask = (hist.index.date >= start_date) & (hist.index.date <= end_date)
            df   = hist.loc[mask].copy()
            if df.empty:
                st.error("Seçilen tarih aralığında veri bulunamadı.")
                st.stop()
            df = build_indicators(df)
            ohlc_mask = ~((df["Open"]==df["High"])&(df["High"]==df["Low"])&(df["Low"]==df["Close"]))
            df = df[ohlc_mask].dropna().copy()
        except Exception as e:
            st.error(f"Veri hatası: {e}")
            st.stop()

    if target not in df.columns:
        st.error(f"'{target}' sütunu veri setinde yok.")
        st.stop()

    notes      = []   # Rapor için notlar
    applied    = []   # Uygulanan düzeltmeler
    step       = 0

    candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    sub        = df[candidates + [target]].dropna()

    # ==============================================================
    # ADIM 1 — Spearman Korelasyon (Bonferroni düzeltmeli)
    # ==============================================================
    step += 1
    n_tests        = len(candidates)
    bonferroni_thr = corr_low / n_tests  # Bonferroni: α / n
    corr_vals      = sub[candidates].apply(lambda col: stats.spearmanr(col, sub[target])[0]).abs()
    low_list       = corr_vals[corr_vals < bonferroni_thr].index.tolist()
    fm             = sub[candidates].corr(method="spearman").abs()
    upper          = fm.where(np.triu(np.ones(fm.shape), k=1).astype(bool))
    high_list      = []
    for col in upper.columns:
        partners = upper.index[upper[col] > corr_high].tolist()
        for p in partners:
            drop = p if corr_vals.get(p,0) <= corr_vals.get(col,0) else col
            if drop not in high_list and drop not in low_list:
                high_list.append(drop)
    corr_remove = list(set(low_list + high_list))
    after_corr  = [f for f in candidates if f not in corr_remove]

    bonferroni_info = f"Bonferroni düzeltmesi: eşik = {corr_low:.2f} / {n_tests} = {bonferroni_thr:.4f}"

    if corr_remove:
        step_card(step, "Spearman Korelasyon", "fix",
                  f"{len(candidates)} feature test edildi. {bonferroni_info}. "
                  f"Düşük |ρ|: {low_list if low_list else 'Yok'}. Yüksek çapraz korelasyon: {high_list if high_list else 'Yok'}.",
                  f"{len(corr_remove)} feature çıkarıldı → {len(after_corr)} kaldı: `{'`, `'.join(after_corr)}`")
        notes.append(f"Spearman filtresi (Bonferroni): {len(corr_remove)} feature çıkarıldı. Eşik={bonferroni_thr:.4f}")
    else:
        step_card(step, "Spearman Korelasyon", "pass",
                  f"{len(candidates)} feature test edildi. {bonferroni_info}. Düşük korelasyon veya multicollinearity tespit edilmedi.")
        notes.append(f"Spearman filtresi (Bonferroni): tüm feature'lar geçti. Eşik={bonferroni_thr:.4f}")

    # ==============================================================
    # ADIM 2 — VIF (Iterative)
    # ==============================================================
    step += 1
    remaining = after_corr.copy()
    vif_rem   = []
    while True:
        sub_vif  = sub[remaining].dropna()
        X_v      = sub_vif.values.astype(float)
        vif_vals = {}
        for i, col in enumerate(remaining):
            try:    vif_vals[col] = variance_inflation_factor(X_v, i)
            except: vif_vals[col] = np.nan
        max_col = max(vif_vals, key=lambda c: vif_vals[c] if not np.isnan(vif_vals[c]) else 0)
        if vif_vals[max_col] > vif_thr:
            vif_rem.append(max_col)
            remaining.remove(max_col)
        else:
            break
    after_vif = remaining

    if vif_rem:
        step_card(step, "VIF — Çoklu Doğrusallık", "fix",
                  f"VIF > {vif_thr} olan feature'lar iteratif olarak çıkarıldı.",
                  f"Çıkarılanlar: `{'`, `'.join(vif_rem)}` → Kalan: `{'`, `'.join(after_vif)}`")
        notes.append(f"VIF: {len(vif_rem)} feature çıkarıldı.")
    else:
        step_card(step, "VIF — Çoklu Doğrusallık", "pass",
                  f"Tüm feature'ların VIF değeri ≤ {vif_thr}. Çoklu doğrusallık yok.")
        notes.append("VIF: tüm feature'lar eşik altında.")

    # ==============================================================
    # ADIM 3 — ADF (Durağanlık)
    # ==============================================================
    step += 1
    non_stat_feats = []
    target_stationary = True
    adf_rows = []

    for col in after_vif + [target]:
        series = df[col].dropna()
        try:
            stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
            stationary = pval < 0.05
            if not stationary:
                if col == target: target_stationary = False
                else: non_stat_feats.append(col)
            adf_rows.append({"Feature": col, "p-değeri": round(pval,4),
                             "Durum": "✅ Durağan" if stationary else "❌ Durağan Değil"})
        except:
            adf_rows.append({"Feature": col, "p-değeri": np.nan, "Durum": "⚠️ Hata"})

    adf_df = pd.DataFrame(adf_rows)

    use_return = False
    if not target_stationary or non_stat_feats:
        step_card(step, "ADF — Durağanlık", "fix",
                  f"Durağan olmayan: Target={'❌' if not target_stationary else '✅'}, Feature'lar: {non_stat_feats if non_stat_feats else 'Yok'}.",
                  "Target → pct_change() (Return). Durağan olmayan feature'lar → pct_change().")
        use_return = True
        notes.append(f"ADF: durağanlık sorunu → Return dönüşümü uygulandı.")
        applied.append("Return dönüşümü")
    else:
        step_card(step, "ADF — Durağanlık", "pass",
                  "Target ve tüm feature'lar durağan. Seviye regresyonu yapılabilir.")
        notes.append("ADF: tüm seriler durağan.")

    with st.expander("ADF detayları"):
        def _adf_c(val):
            if not isinstance(val, str): return ""
            if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
            if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
            return ""
        st.dataframe(adf_df.style.map(_adf_c, subset=["Durum"]), use_container_width=True, hide_index=True)

    # Veriyi hazırla
    working = df[after_vif + [target]].dropna().copy()
    if use_return:
        working[target] = working[target].pct_change()
        for col in non_stat_feats:
            if col in working.columns:
                working[col] = working[col].pct_change()
    working = working.dropna()
    y = working[target].values
    X = add_constant(working[after_vif].values.astype(float))

    # OLS fit (başlangıç)
    try:
        ols_base = OLS(y, X).fit()
    except Exception as e:
        st.error(f"OLS kurulamadı: {e}")
        st.stop()

    resid = ols_base.resid

    # ==============================================================
    # ADIM 4 — Otokorelasyon (Ljung-Box)
    # ==============================================================
    step += 1
    try:
        lb     = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_p   = float(lb["lb_pvalue"].iloc[0])
        has_ac = lb_p < 0.05
    except:
        lb_p = np.nan; has_ac = False

    # Kart — HAC kararı henüz verilmedi, sadece tespit göster
    if has_ac:
        step_card(step, "Ljung-Box — Otokorelasyon", "fix",
                  f"p = {lb_p:.4f} < 0.05 — artıklarda otokorelasyon var.",
                  "HAC (Newey-West, maxlags=5) standart hatalar uygulanacak.")
        notes.append("Otokorelasyon tespit edildi → HAC tetiklendi.")
    else:
        step_card(step, "Ljung-Box — Otokorelasyon", "pass",
                  f"p = {lb_p:.4f} ≥ 0.05 — artıklarda otokorelasyon yok.")
        notes.append("Otokorelasyon yok.")

    # ==============================================================
    # ADIM 5 — ARCH (Heteroskedasticity)
    # — Ham artıklar üzerinde test et, HAC kararından bağımsız
    # ==============================================================
    step += 1
    try:
        _, arch_p, _, _ = het_arch(resid, nlags=5)
        has_arch = arch_p < 0.05
    except:
        arch_p = np.nan; has_arch = False

    if has_arch:
        step_card(step, "ARCH — Heteroskedasticity", "fix",
                  f"p = {arch_p:.4f} < 0.05 — volatilite kümelenmesi var.",
                  "HAC (Newey-West, maxlags=5) standart hatalar uygulanacak — ARCH etkisini kısmen yönetir.")
        notes.append("ARCH etkisi tespit edildi → HAC tetiklendi.")
    else:
        step_card(step, "ARCH — Heteroskedasticity", "pass",
                  f"p = {arch_p:.4f} ≥ 0.05 — sabit varyans. OLS verimli.")
        notes.append("ARCH etkisi yok.")

    # HAC kararı: Ljung-Box VEYA ARCH başarısız olursa uygula
    use_hac = has_ac or has_arch
    try:
        if use_hac:
            ols_fit = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            hac_reason = []
            if has_ac:   hac_reason.append("otokorelasyon")
            if has_arch: hac_reason.append("ARCH etkisi")
            st.markdown(
                f'<div class="step-box step-fix">🔧 <b>HAC Uygulandı</b> — '
                f'Neden: {" + ".join(hac_reason)}. Newey-West (maxlags=5) standart hatalar aktif.</div>',
                unsafe_allow_html=True
            )
            applied.append(f"HAC standart hata ({', '.join(hac_reason)})")
        else:
            ols_fit = ols_base
        resid = ols_fit.resid
    except:
        ols_fit = ols_base

    # ==============================================================
    # ADIM 6 — Normallik (Jarque-Bera) — artıklara
    # ==============================================================
    step += 1
    try:
        _, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
        non_normal = jb_p < 0.05
    except:
        jb_p = np.nan; non_normal = False; jb_skew = np.nan; jb_kurt = np.nan

    if non_normal:
        if not target_stationary:
            step_card(step, "Jarque-Bera — Normallik", "fix",
                      f"p = {jb_p:.4f} — artıklar normal dağılmıyor (Çarpıklık={jb_skew:.2f}, Basıklık={jb_kurt:.2f}).",
                      "Eşbütünleşme mevcut ve HAC uygulandıysa CLT geçerli — normallik varsayımı hafifletilebilir.")
        else:
            step_card(step, "Jarque-Bera — Normallik", "fail",
                      f"p = {jb_p:.4f} — artıklar normal dağılmıyor (Çarpıklık={jb_skew:.2f}, Basıklık={jb_kurt:.2f}). Durağan serilerde CLT geçerli ise kabul edilebilir.")
        notes.append(f"Normallik: sağlanmıyor (çarpıklık={jb_skew:.2f}).")
    else:
        step_card(step, "Jarque-Bera — Normallik", "pass",
                  f"p = {jb_p:.4f} — artıklar normal dağılıyor.")
        notes.append("Normallik: sağlanıyor.")

    # ==============================================================
    # ADIM 7 — Doğrusallık (RESET)
    # ==============================================================
    step += 1
    try:
        rst    = linear_reset(ols_fit, power=2, use_f=True)
        rst_p  = rst.pvalue
        nonlin = rst_p < 0.05
    except:
        rst_p = np.nan; nonlin = False

    if nonlin:
        step_card(step, "RESET — Doğrusallık", "info",
                  f"p = {rst_p:.4f} < 0.05 — doğrusal olmayan ilişki tespit edildi. "
                  "Finansal serilerde beklenen bir sonuç. "
                  "Katsayılar yaklaşık yorumlanmalıdır. "
                  "Rejim değişikliği şüphesi varsa TAR/Markov Switching düşünülebilir.")
        notes.append("Doğrusallık: sağlanmıyor — katsayılar yaklaşık.")
    else:
        step_card(step, "RESET — Doğrusallık", "pass",
                  f"p = {rst_p:.4f} ≥ 0.05 — doğrusal ilişki yeterli.")
        notes.append("Doğrusallık: sağlanıyor.")

    # ==============================================================
    # ADIM 8 — Yapısal Kırılma (CUSUM)
    # ==============================================================
    step += 1
    try:
        _, cusum_p, _ = breaks_cusumolsresid(resid)
        has_break = cusum_p < 0.05
    except:
        cusum_p = np.nan; has_break = False

    if has_break:
        step_card(step, "CUSUM — Yapısal Kırılma", "info",
                  f"p = {cusum_p:.4f} < 0.05 — katsayılar zaman içinde değişiyor. "
                  "Tüm dönem için tek model varsayımı zayıflar. "
                  "Alt dönemlere bölme veya rolling window önerilir. "
                  "Akademik sunumda 'ortalama ilişki' olarak raporlanabilir.")
        notes.append("Yapısal kırılma: var — rolling window önerilir.")
    else:
        step_card(step, "CUSUM — Yapısal Kırılma", "pass",
                  f"p = {cusum_p:.4f} ≥ 0.05 — katsayılar stabil.")
        notes.append("Yapısal kırılma: yok.")

    # ==============================================================
    # ADIM 9 — Eşbütünleşme (Johansen)
    # ==============================================================
    step += 1
    johansen_n    = 0
    johansen_ok   = False
    johansen_note = ""

    # Johansen testi yalnızca seriler durağan DEĞİLSE anlamlıdır.
    # use_return=True → ADF durağan olmayan seri tespit etti → level veriyle Johansen geçerli.
    # use_return=False → tüm seriler zaten durağan → eşbütünleşme testi uygulanamaz,
    #   durağan seriler üzerinde Johansen çalıştırmak anlamsız sonuç üretir.
    if not use_return:
        step_card(step, "Johansen — Eşbütünleşme", "info",
                  "Tüm seriler ADF testinde durağan bulundu. "
                  "Eşbütünleşme testi yalnızca I(1) (durağan olmayan) seriler için geçerlidir — atlandı.")
        notes.append("Johansen: seriler durağan → test uygulanamaz, atlandı.")
    else:
        # Johansen seviye (level) verisiyle çalışır.
        # use_return=True ise seriler durağan değildi → ham df doğru girdi.
        try:
            cols_j = after_vif + [target]
            data_j = df[cols_j].dropna().values.astype(float)
            res_j  = coint_johansen(data_j, det_order=0, k_ar_diff=1)
            for i in range(len(res_j.lr1)):
                if res_j.lr1[i] is not None and res_j.lr1[i] > res_j.cvt[i, 1]:
                    johansen_n += 1
            johansen_ok   = johansen_n > 0
            johansen_note = "Ham (level) veri kullanıldı."
        except Exception:
            # Ham veri matris hatasına yol açtıysa z-score ile tekrar dene
            try:
                data_jz = scipy_zscore(data_j, axis=0)
                res_j   = coint_johansen(data_jz, det_order=0, k_ar_diff=1)
                for i in range(len(res_j.lr1)):
                    if res_j.lr1[i] is not None and res_j.lr1[i] > res_j.cvt[i, 1]:
                        johansen_n += 1
                johansen_ok   = johansen_n > 0
                johansen_note = "⚠️ Ham veri matris hatasına yol açtı — z-score ile tekrar çalıştırıldı. Sonuçlar gösterge niteliğindedir."
            except Exception:
                johansen_n = -1

        if johansen_n == -1:
            step_card(step, "Johansen — Eşbütünleşme", "info",
                      "Test çalıştırılamadı (matris hatası). Engle-Granger ikili testini deneyin.")
            notes.append("Johansen: hata.")
        elif johansen_ok:
            step_card(step, "Johansen — Eşbütünleşme", "pass",
                      f"{johansen_n} eşbütünleşme ilişkisi tespit edildi. "
                      "Feature'lar ve target arasında uzun vadeli gerçek ilişki var. "
                      f"OLS katsayıları güvenilir — sahte regresyon değil. {johansen_note}")
            notes.append(f"Eşbütünleşme: {johansen_n} ilişki — OLS güvenilir. {johansen_note}")
        else:
            step_card(step, "Johansen — Eşbütünleşme", "fail",
                      f"Eşbütünleşme tespit edilmedi. {johansen_note} "
                      "Seviye regresyonu sahte olabilir. Return modeli önerilir.")
            notes.append("Eşbütünleşme: yok — Return modeli önerilir.")

    # ==============================================================
    # ADIM 10 — MODEL SONUÇLARI
    # ==============================================================
    step += 1
    st.divider()
    st.subheader(f"🏁 Adım {step} — Final Model Sonuçları")

    model_name = "OLS + HAC (Newey-West)" if use_hac else "OLS"
    if use_return:
        model_name += " + Return"
    st.caption(f"Uygulanan model: **{model_name}** | Feature sayısı: {len(after_vif)}")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("R²",      f"{ols_fit.rsquared:.4f}")
    col_b.metric("Adj. R²", f"{ols_fit.rsquared_adj:.4f}")
    col_c.metric("AIC",     f"{ols_fit.aic:.2f}")
    col_d.metric("BIC",     f"{ols_fit.bic:.2f}")

    col_e, col_f = st.columns(2)
    col_e.metric("F-istatistiği", f"{ols_fit.fvalue:.4f}")
    col_f.metric("F p-değeri",    f"{ols_fit.f_pvalue:.4f}")

    # Katsayılar tablosu
    st.markdown("**Katsayılar**")
    feature_names = ["const"] + after_vif
    coef_rows = []
    for i, fname in enumerate(feature_names):
        if fname == "const": continue
        pval = ols_fit.pvalues[i]
        sig  = "✅ Anlamlı" if pval < 0.05 else ("⚠️ Sınırda" if pval < 0.10 else "❌ Anlamsız")
        coef_rows.append({
            "Feature":       fname,
            "Katsayı":       round(ols_fit.params[i], 6),
            "Std Hata":      round(ols_fit.bse[i], 6),
            "t-istatistiği": round(ols_fit.tvalues[i], 4),
            "p-değeri":      round(pval, 4),
            "Anlamlılık":    sig,
        })
    coef_df = pd.DataFrame(coef_rows)

    def _sc(val):
        if not isinstance(val, str): return ""
        if val.startswith("✅"): return "background-color:#d1e7dd; color:#0a3622"
        if val.startswith("⚠️"): return "background-color:#fff3cd; color:#664d03"
        if val.startswith("❌"): return "background-color:#f8d7da; color:#842029"
        return ""

    st.dataframe(
        coef_df.style
            .format({"Katsayı": "{:.6f}", "Std Hata": "{:.6f}", "t-istatistiği": "{:.4f}", "p-değeri": "{:.4f}"})
            .map(_sc, subset=["Anlamlılık"]),
        use_container_width=True, hide_index=True,
    )

    sig_f   = [r["Feature"] for r in coef_rows if r["Anlamlılık"].startswith("✅")]
    insig_f = [r["Feature"] for r in coef_rows if r["Anlamlılık"].startswith("❌")]
    if sig_f:   st.success(f"Anlamlı feature'lar: `{'`, `'.join(sig_f)}`")
    if insig_f: st.warning(f"Anlamsız feature'lar: `{'`, `'.join(insig_f)}` — modelden çıkarılabilir")

    # Artık grafiği
    st.markdown("**Artık Grafiği**")
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=list(range(len(resid))), y=resid, mode="lines",
                               line=dict(color="#0d6efd", width=1), name="Artık"))
    fig_r.add_hline(y=0, line_dash="dash", line_color="red")
    fig_r.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
                        xaxis_title="Gözlem", yaxis_title="Artık", hovermode="x unified")
    st.plotly_chart(fig_r, use_container_width=True)

    # Q-Q Plot
    st.markdown("**Q-Q Plot**")
    (osm, osr), (slope, intercept, _) = stats.probplot(resid)
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                marker=dict(color="#0d6efd", size=4), name="Artıklar"))
    fig_qq.add_trace(go.Scatter(x=osm, y=[slope*x+intercept for x in osm], mode="lines",
                                line=dict(color="red", dash="dash"), name="Normal"))
    fig_qq.update_layout(height=280, margin=dict(l=40,r=20,t=20,b=40),
                         xaxis_title="Teorik Kantiller", yaxis_title="Örnek Kantiller")
    st.plotly_chart(fig_qq, use_container_width=True)

    # ==============================================================
    # ÖZET RAPOR
    # ==============================================================
    st.divider()
    st.subheader("📄 Özet Rapor")

    passed_count = sum([
        True,                                          # Spearman her zaman çözüm üretir
        True,                                          # VIF iterative her zaman çözüm üretir
        True,                                          # ADF → Return uygulandı, çözüldü
        True,                                          # Otokorelasyon → HAC uygulandı, çözüldü
        not has_arch or use_hac,                       # ARCH → HAC varsa yönetildi
        not non_normal or (johansen_ok and use_hac),   # Normallik → eşbütünleşme+HAC ile hafifletildi
        not nonlin,                                    # RESET — düzeltme yok, ya geçer ya geçmez
        not has_break,                                 # CUSUM — düzeltme yok, ya geçer ya geçmez
        johansen_ok or not use_return,                 # Johansen — use_return=False ise zaten durağan, sorun yok
    ])
    total_steps = 9

    if passed_count >= 8:
        st.success(f"**{passed_count}/{total_steps} adım tamamlandı.** Model güvenilir.")
    elif passed_count >= 6:
        st.info(f"**{passed_count}/{total_steps} adım tamamlandı.** Uygulanan düzeltmelerle model kabul edilebilir.")
    else:
        st.warning(f"**{passed_count}/{total_steps} adım tamamlandı.** Model yorumlanırken dikkatli olunmalı.")

    st.markdown("**Uygulanan düzeltmeler:**")
    if applied:
        for a in applied:
            st.markdown(f"- {a}")
    else:
        st.markdown("- Düzeltme gerekmedi.")

    st.markdown("**Adım notları:**")
    for i, note in enumerate(notes, 1):
        st.markdown(f"{i}. {note}")

    # Akademik not
    with st.expander("📝 Akademik Metodoloji Notu"):
        hac_note    = "Otokorelasyon ve heteroskedasticity için HAC standart hatalar (Newey-West, 1987) uygulanmıştır. " if use_hac else ""
        return_note = "Durağanlık sağlamak amacıyla bağımlı ve durağan olmayan bağımsız değişkenler için yüzdesel getiri dönüşümü uygulanmıştır. " if use_return else ""
        coint_note  = f"Johansen (1988) eşbütünleşme testi {johansen_n} uzun vadeli ilişki tespit etmiştir; OLS katsayıları sahte regresyon içermemektedir. " if johansen_ok else ""
        reset_note  = "RESET testi doğrusal olmayan ilişki sinyali vermiştir; katsayılar yaklaşık olarak yorumlanmalıdır (Ramsey, 1969). " if nonlin else ""
        cusum_note  = "CUSUM testi yapısal kırılma sinyali vermiştir; bulgular tüm örneklem dönemi için ortalama ilişkiyi yansıtmaktadır. " if has_break else ""

        st.markdown(f"""
Bu çalışmada {symbol.upper()} için {start_date} — {end_date} dönemine ait günlük veri kullanılmıştır.
Çoklu doğrusallık Variance Inflation Factor (VIF > {vif_thr}) ile kontrol edilmiş, yüksek VIF değerine sahip değişkenler iteratif olarak çıkarılmıştır.
Çoklu test sorununun yalancı anlamlılık riskini azaltmak amacıyla Spearman korelasyon eşiğine Bonferroni düzeltmesi uygulanmıştır.
Durağanlık Augmented Dickey-Fuller (ADF) testi ile sınanmıştır.
{return_note}{hac_note}{coint_note}{reset_note}{cusum_note}
Final modelde {len(sig_f)} değişken istatistiksel olarak anlamlı bulunmuştur (p < 0.05): {', '.join(sig_f) if sig_f else 'Yok'}.

**Sınırlılıklar:** Bağımsız değişkenlerin büyük bölümü hedef değişkenden (kapanış fiyatı) türetilmiş teknik indikatörlerdir. Bu durum içsellik (endogeneity) sorununa yol açabilir ve OLS katsayılarının yanlı olmasına neden olabilir. Araç değişken (IV) tahmini için uygun enstrüman bulunamamıştır. Bu nedenle bulgular nedensellik değil, korelasyon ilişkisi olarak yorumlanmalıdır.
        """)
