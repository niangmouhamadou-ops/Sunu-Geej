"""
Observatoire côtier du Sénégal — Streamlit
==========================================
Carte interactive + espace administrateur + détection d'anomalies par
rupture de la structure de corrélation (sondes multiparamètres / ADCP).

Lancement :  streamlit run app.py
Mot de passe admin (optionnel) : variable d'environnement ADMIN_PASSWORD
ou fichier .streamlit/secrets.toml  ->  ADMIN_PASSWORD = "..."

Architecture du fichier
-----------------------
  1. Configuration & points stratégiques
  2. Persistance (registre JSON + fichiers bruts sur disque)
  3. Lecture / normalisation des fichiers (CSV, NetCDF)
  4. Moteur d'analyse : corrélations de référence + détection d'anomalies
  5. Graphiques (Plotly)
  6. Interface (carte, mode analyste, mode administrateur)
"""

from __future__ import annotations

import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import chi2, theilslopes
from sklearn.covariance import LedoitWolf
from streamlit_folium import st_folium

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
DATA_DIR = Path(os.environ.get("OBS_DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
REGISTRY_FILE = DATA_DIR / "registry.json"
SEED_DIR = Path("reference_data")  # jeux de référence injectés au 1er lancement

MAP_CENTER = (14.49, -14.45)  # centre du Sénégal

# Coordonnées approximatives (modifiables depuis l'espace administrateur).
DEFAULT_STATIONS = {
    "Dakar-Port": (14.683, -17.429),
    "Île de Gorée": (14.667, -17.398),
    "Cap-Vert": (14.743, -17.529),
    "Saint-Louis": (16.033, -16.482),
    "Casamance (Ziguinchor)": (12.583, -16.272),
    "Diakhanor (Saloum)": (13.80, -16.65),  # à vérifier / corriger
}
NEW_POINT = "➕ Nouveau point…"

st.set_page_config(page_title="Observatoire côtier du Sénégal", page_icon="🌊", layout="wide")


# =============================================================================
# 2. PERSISTANCE (registre JSON + fichiers bruts)
# =============================================================================
def load_registry() -> dict:
    """Charge le registre ; le crée (avec le jeu de référence Diakhanor) au besoin."""
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    reg = {"stations": {k: list(v) for k, v in DEFAULT_STATIONS.items()}, "datasets": []}
    for f in sorted(SEED_DIR.glob("diakhanor*")) if SEED_DIR.exists() else []:
        add_dataset(reg, "Diakhanor (Saloum)", "reference", f.name, f.read_bytes(), save=False)
    save_registry(reg)
    return reg


def save_registry(reg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def add_dataset(reg: dict, station: str, role: str, filename: str, raw: bytes, save: bool = True) -> dict:
    """Enregistre le fichier brut sur disque et l'associe à un point."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ds_id = uuid.uuid4().hex[:8]
    safe = re.sub(r"[^\w.\-]", "_", filename)
    path = UPLOAD_DIR / f"{ds_id}_{safe}"
    path.write_bytes(raw)
    entry = {
        "id": ds_id, "station": station, "role": role, "filename": filename,
        "path": str(path), "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    reg["datasets"].append(entry)
    if save:
        save_registry(reg)
    return entry


def delete_dataset(reg: dict, ds_id: str) -> None:
    for d in list(reg["datasets"]):
        if d["id"] == ds_id:
            Path(d["path"]).unlink(missing_ok=True)
            reg["datasets"].remove(d)
    save_registry(reg)


def ds_label(d: dict) -> str:
    role = "référence" if d["role"] == "reference" else "nouvelle mesure"
    return f"{d['filename']} — {role} — {d['uploaded_at']}"


# =============================================================================
# 3. LECTURE ET NORMALISATION DES FICHIERS
# =============================================================================
def _split_label(label: str) -> tuple[str, str]:
    """'Pressure(dbar)' -> ('Pressure', 'dbar')."""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", label)
    return (m.group(1).strip(), m.group(2).strip()) if m else (label.strip(), "")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    - détecte la colonne temps et la renomme 'time' ;
    - supprime les préfixes 'CHn:' ; les noms en double reçoivent le suffixe _CHn
      (ex. deux colonnes Temperature -> Temperature_CH1, Temperature_CH7) ;
    - convertit tout en numérique (virgule décimale acceptée).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    time_col = next((c for c in df.columns if re.search(r"time|date|heure", c, re.I)), None)
    out = pd.DataFrame(index=df.index)
    if time_col is not None:
        t = pd.to_datetime(df[time_col], errors="coerce")
        if t.notna().mean() > 0.5:
            out["time"] = t

    parsed = []
    for c in df.columns:
        if c == time_col:
            continue
        m = re.match(r"^CH(\d+):(.*)$", c)
        ch, label = (m.group(1), m.group(2)) if m else (None, c)
        name, unit = _split_label(label)
        parsed.append((c, ch, name, unit))
    counts = pd.Series([p[2] for p in parsed]).value_counts()

    for i, (orig, ch, name, unit) in enumerate(parsed):
        s = df[orig]
        if s.dtype == object:
            s = s.astype(str).str.replace(",", ".", regex=False)
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() == 0:
            continue
        if counts[name] > 1:
            name = f"{name}_CH{ch}" if ch is not None else f"{name}_{i}"
        out[f"{name} ({unit})" if unit else name] = s

    if "time" in out:
        out = out.dropna(subset=["time"]).sort_values("time")
    return out.reset_index(drop=True)


def parse_table(raw: bytes, filename: str) -> pd.DataFrame:
    """Lit un CSV (séparateur auto : ; , tab) ou un NetCDF et le normalise."""
    if filename.lower().endswith((".nc", ".cdf", ".netcdf")):
        import tempfile

        import xarray as xr

        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            tmp.write(raw)
            tmp.flush()
            with xr.open_dataset(tmp.name) as ds:
                df = ds.load().to_dataframe().reset_index()
        return normalize(df)

    text = raw.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text else ""
    sep = max([";", ",", "\t"], key=first.count)
    return normalize(pd.read_csv(io.StringIO(text), sep=sep))


@st.cache_data(show_spinner=False)
def load_dataset(path: str, mtime: float) -> pd.DataFrame:
    return parse_table(Path(path).read_bytes(), Path(path).name)


def read_entry(d: dict) -> pd.DataFrame:
    p = Path(d["path"])
    return load_dataset(str(p), p.stat().st_mtime)


def find_col(df: pd.DataFrame, *keys: str) -> str | None:
    for k in keys:
        for c in df.columns:
            if c.lower().startswith(k):
                return c
    return None


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "time"]


def apply_immersion_filter(df: pd.DataFrame, min_depth: float) -> pd.DataFrame:
    """Écarte les mesures de surface / hors d'eau (profondeur < seuil)."""
    dcol = find_col(df, "depth", "profondeur")
    if min_depth > 0 and dcol:
        return df[df[dcol] >= min_depth].reset_index(drop=True)
    return df


# =============================================================================
# 4. MOTEUR D'ANALYSE
# =============================================================================
def build_reference(df: pd.DataFrame, min_abs_r: float = 0.7) -> dict | None:
    """
    Construit la « signature » du point à partir des données de référence :
      * échelle robuste (médiane / MAD) de chaque paramètre ;
      * covariance régularisée (Ledoit-Wolf) -> distance de Mahalanobis ;
      * matrice de corrélation de Spearman ;
      * droites de régression robustes (Theil-Sen) pour les paires fortement corrélées.
    Les paramètres constants (ex. turbidité nulle) sont ignorés.
    """
    cols = [c for c in numeric_cols(df) if df[c].nunique() > 2 and df[c].std() > 1e-9]
    d = df[cols].dropna()
    if len(d) < 10 or len(cols) < 2:
        return None

    med = d.median()
    mad = 1.4826 * (d - med).abs().median()
    mad = mad.where(mad > 1e-9, d.std())
    z = ((d - med) / mad).clip(-10, 10)  # limite l'influence des pics de la référence
    cov = pd.DataFrame(LedoitWolf().fit(z.values).covariance_, index=cols, columns=cols)

    corr = d.corr(method="spearman")
    pairs = sorted(
        ((abs(corr.loc[a, b]), a, b) for i, a in enumerate(cols) for b in cols[i + 1:]
         if abs(corr.loc[a, b]) >= min_abs_r),
        reverse=True,
    )[:25]

    sample = d if len(d) <= 600 else d.sample(600, random_state=0)
    fits = []
    for _, a, b in pairs:
        slope, intercept, _, _ = theilslopes(sample[b], sample[a])
        res = d[b] - (intercept + slope * d[a])
        scale = max(1.4826 * np.median(np.abs(res - np.median(res))), 0.05 * d[b].std())
        fits.append({"x": a, "y": b, "slope": slope, "intercept": intercept,
                     "scale": scale, "rho_ref": corr.loc[a, b]})
    return {"cols": cols, "med": med, "mad": mad, "cov": cov, "corr": corr,
            "fits": fits, "n": len(d)}


def detect_anomalies(ref: dict, df: pd.DataFrame, q: float = 0.995, z_resid: float = 6.0,
                     min_pairs: int = 2, alert_rows: int = 3, alert_drop: float = 0.35) -> dict | None:
    """
    Compare un jeu de données à la signature de référence.

    Un point est anormal si :
      (a) sa distance de Mahalanobis dépasse le quantile q du χ² (point multivarié aberrant), OU
      (b) au moins `min_pairs` paires normalement corrélées s'écartent de leur droite de
          régression de référence de plus de `z_resid` écarts robustes (corrélation brisée).
    Indice de rupture globale = chute moyenne de |ρ| sur les paires fortes entre
    la référence et le jeu analysé.
    """
    common = [c for c in ref["cols"] if c in df.columns]
    if len(common) < 2 or len(df) == 0:
        return None
    df = df.reset_index(drop=True)

    z = ((df[common] - ref["med"][common]) / ref["mad"][common]).fillna(0).values
    prec = np.linalg.pinv(ref["cov"].loc[common, common].values)
    d2 = np.einsum("ij,jk,ik->i", z, prec, z)
    thr = chi2.ppf(q, len(common))

    n_broken = np.zeros(len(df), dtype=int)
    rows = []
    for f in ref["fits"]:
        if f["x"] not in df or f["y"] not in df:
            continue
        resid = (df[f["y"]] - (f["intercept"] + f["slope"] * df[f["x"]])) / f["scale"]
        bad = (resid.abs() > z_resid).fillna(False).values
        n_broken += bad
        r_new = df[[f["x"], f["y"]]].corr(method="spearman").iloc[0, 1] if len(df) >= 5 else np.nan
        drop = abs(f["rho_ref"]) - np.sign(f["rho_ref"]) * r_new
        rows.append({"Paire": f"{f['x']}  ↔  {f['y']}", "ρ référence": f["rho_ref"],
                     "ρ analysé": r_new, "Chute de corrélation": drop,
                     "% points hors tendance": 100 * bad.mean()})

    pairs_df = pd.DataFrame(rows)
    flag = (d2 > thr) | (n_broken >= min_pairs)
    scored = pd.DataFrame({"time": df["time"] if "time" in df else df.index,
                           "mahalanobis_d2": d2, "paires_rompues": n_broken, "anormal": flag})

    global_drop = float(np.nanmean(np.clip(pairs_df["Chute de corrélation"], 0, None))) if len(pairs_df) else 0.0
    n_flag = int(flag.sum())
    return {"scored": scored, "pairs": pairs_df, "threshold": thr, "n_flag": n_flag,
            "global_drop": global_drop, "n_rows": len(df),
            "alert": n_flag >= alert_rows or global_drop >= alert_drop}


# =============================================================================
# 5. GRAPHIQUES
# =============================================================================
def fig_timeseries(df: pd.DataFrame, cols: list[str], flags: np.ndarray | None) -> go.Figure:
    t = df["time"] if "time" in df else pd.Series(df.index, index=df.index)
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=cols)
    for i, c in enumerate(cols, 1):
        fig.add_trace(go.Scatter(x=t, y=df[c], mode="lines", name=c, showlegend=False), row=i, col=1)
        if flags is not None and flags.any():
            fig.add_trace(go.Scatter(x=t[flags], y=df[c][flags], mode="markers", name="Anomalie",
                                     marker=dict(color="red", size=7), showlegend=(i == 1)), row=i, col=1)
    fig.update_layout(height=230 * len(cols) + 60, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_profile(df: pd.DataFrame, cols: list[str], flags: np.ndarray | None) -> go.Figure | None:
    dcol = find_col(df, "depth", "profondeur", "pressure", "pression")
    if dcol is None:
        return None
    fig = make_subplots(rows=1, cols=len(cols), shared_yaxes=True, subplot_titles=cols)
    for i, c in enumerate(cols, 1):
        fig.add_trace(go.Scatter(x=df[c], y=df[dcol], mode="lines+markers", marker=dict(size=4),
                                 showlegend=False), row=1, col=i)
        if flags is not None and flags.any():
            fig.add_trace(go.Scatter(x=df[c][flags], y=df[dcol][flags], mode="markers",
                                     marker=dict(color="red", size=8), showlegend=False), row=1, col=i)
    fig.update_yaxes(autorange="reversed", title_text=dcol, row=1, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_corr(m: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(m, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    aspect="auto", title=title)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# =============================================================================
# 6. INTERFACE
# =============================================================================
def sidebar_params() -> dict:
    with st.sidebar.expander("⚙️ Paramètres de détection", expanded=False):
        return {
            "min_depth": st.slider("Profondeur min. analysée (m)", 0.0, 5.0, 0.5, 0.1,
                                   help="Écarte les mesures hors d'eau / de surface. 0 = désactivé."),
            "min_abs_r": st.slider("Corrélation minimale d'une paire (|ρ|)", 0.3, 0.95, 0.7, 0.05),
            "q": st.slider("Seuil Mahalanobis (quantile χ²)", 0.90, 0.9999, 0.995, 0.001, format="%.4f"),
            "z_resid": st.slider("Écart résiduel de régression (σ robustes)", 2.0, 10.0, 6.0, 0.5),
            "min_pairs": st.slider("Paires rompues min. par point", 1, 5, 2),
            "alert_rows": st.slider("Points anormaux min. pour alerter", 1, 20, 3),
        }


def show_alert(res: dict | None) -> None:
    """Alerte visuelle claire selon le résultat de la détection."""
    if res is None:
        st.warning("Analyse impossible : trop peu de paramètres communs avec la référence.")
    elif res["alert"]:
        st.error(
            "⚠️ **Variation inhabituelle détectée — Rupture de la structure de corrélation standard.**  \n"
            f"{res['n_flag']} point(s) anormaux sur {res['n_rows']} · "
            f"chute moyenne de corrélation : {res['global_drop']:.2f}"
        )
    elif res["n_flag"] > 0:
        st.warning(f"{res['n_flag']} point(s) isolé(s) atypique(s) sur {res['n_rows']} "
                   "— en dessous du seuil d'alerte.")
    else:
        st.success("✅ Structure de corrélation conforme à la référence.")


def render_station(reg: dict, station: str, p: dict, key: str = "main") -> None:
    dsets = [d for d in reg["datasets"] if d["station"] == station]
    if not dsets:
        st.info(f"Aucune donnée n'est encore associée à **{station}**. "
                "Un administrateur peut en téléverser depuis le mode Administrateur.")
        return

    refs = [d for d in dsets if d["role"] == "reference"]
    default = (([d for d in dsets if d["role"] != "reference"] or refs)[-1])
    target = st.selectbox("Jeu de données analysé", dsets, index=dsets.index(default),
                          format_func=ds_label, key=f"target_{key}_{station}")

    try:
        target_df = read_entry(target)
        ref_src = pd.concat([read_entry(d) for d in refs], join="inner", ignore_index=True) if refs else target_df
    except Exception as e:  # fichier corrompu / illisible
        st.error(f"Lecture impossible : {e}")
        return

    ref_f = apply_immersion_filter(ref_src, p["min_depth"])
    tgt_f = apply_immersion_filter(target_df, p["min_depth"])
    if len(tgt_f) == 0 or len(ref_f) == 0:
        st.warning("Le filtre de profondeur écarte toutes les mesures — réduisez la profondeur minimale.")
        return
    if not refs:
        st.warning("Aucune référence enregistrée : le jeu analysé sert de référence (auto-contrôle).")

    ref = build_reference(ref_f, p["min_abs_r"])
    if ref is None:
        st.warning("Référence insuffisante (≥ 10 lignes et ≥ 2 paramètres variables requis).")
        res = None
    else:
        res = detect_anomalies(ref, tgt_f, p["q"], p["z_resid"], p["min_pairs"], p["alert_rows"])
        if target["role"] == "reference":
            st.caption("ℹ️ Auto-contrôle : la référence est comparée à elle-même "
                       "(utile pour repérer les artefacts de capteur).")
    show_alert(res)

    flags = res["scored"]["anormal"].values if res else None
    params = numeric_cols(tgt_f)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lignes analysées", f"{len(tgt_f)} / {len(target_df)}")
    m2.metric("Lignes de référence", ref["n"] if ref else "—")
    m3.metric("Points anormaux", res["n_flag"] if res else "—")
    m4.metric("Chute moy. de corrélation", f"{res['global_drop']:.2f}" if res else "—")

    tab_ts, tab_prof, tab_corr, tab_anom = st.tabs(
        ["📈 Séries temporelles", "🌊 Profil vertical", "🔗 Corrélations", "🚨 Anomalies"])

    with tab_ts:
        sel = st.multiselect("Paramètres", params, default=params[:4], key=f"ts_{key}_{station}")
        if sel:
            st.plotly_chart(fig_timeseries(tgt_f, sel, flags), use_container_width=True)
    with tab_prof:
        sel = st.multiselect("Paramètres", params, default=params[:4], key=f"pf_{key}_{station}")
        fig = fig_profile(tgt_f, sel, flags) if sel else None
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune colonne profondeur / pression pour tracer un profil.")
    with tab_corr:
        if ref is None:
            st.info("Matrice indisponible (référence insuffisante).")
        else:
            choice = st.radio("Matrice", ["Référence", "Jeu analysé", "Différence (analysé − référence)"],
                              horizontal=True, key=f"cm_{key}_{station}")
            cols = ref["cols"]
            new_corr = tgt_f[[c for c in cols if c in tgt_f]].corr(method="spearman")
            if choice == "Référence":
                st.plotly_chart(fig_corr(ref["corr"], "Corrélation de Spearman — référence"), use_container_width=True)
            elif choice == "Jeu analysé":
                st.plotly_chart(fig_corr(new_corr, "Corrélation de Spearman — jeu analysé"), use_container_width=True)
            else:
                diff = (new_corr - ref["corr"].loc[new_corr.index, new_corr.columns]).clip(-1, 1)
                st.plotly_chart(fig_corr(diff, "Écart de corrélation"), use_container_width=True)
    with tab_anom:
        if res is None:
            st.info("Pas de résultat.")
        else:
            st.markdown("**Paires dont la corrélation habituelle est la plus altérée**")
            st.dataframe(res["pairs"].sort_values("% points hors tendance", ascending=False).round(3),
                         use_container_width=True, hide_index=True)
            bad = res["scored"][res["scored"]["anormal"]]
            st.markdown(f"**Points signalés** (seuil Mahalanobis d² = {res['threshold']:.1f})")
            if bad.empty:
                st.caption("Aucun point signalé.")
            else:
                st.dataframe(bad.round(2), use_container_width=True, hide_index=True)
    with st.expander("Données brutes normalisées"):
        st.dataframe(tgt_f, use_container_width=True)


# ---------------------------------------------------------------- mode analyste
def page_user(reg: dict, p: dict) -> None:
    st.title("🌊 Observatoire côtier du Sénégal")
    stations = reg["stations"]
    names = list(stations)
    if st.session_state.get("station") not in names:
        st.session_state["station"] = names[0]

    choice = st.selectbox("Point stratégique", names, index=names.index(st.session_state["station"]))
    st.session_state["station"] = choice

    has_data = {s for s in names if any(d["station"] == s for d in reg["datasets"])}
    m = folium.Map(location=MAP_CENTER, zoom_start=7, tiles="OpenStreetMap")
    for name, (lat, lon) in stations.items():
        color = "green" if name == st.session_state["station"] else ("blue" if name in has_data else "gray")
        n = sum(d["station"] == name for d in reg["datasets"])
        folium.Marker([lat, lon], tooltip=name, popup=f"{name} — {n} jeu(x) de données",
                      icon=folium.Icon(color=color, icon="tint", prefix="fa")).add_to(m)
    out = st_folium(m, height=430, use_container_width=True, key="map",
                    returned_objects=["last_object_clicked_tooltip"])
    st.caption("🟦 données disponibles · ⬜ pas encore de données · 🟩 point sélectionné")

    clicked = (out or {}).get("last_object_clicked_tooltip")
    if clicked in stations and clicked != st.session_state.get("last_click"):
        st.session_state["last_click"] = clicked
        if clicked != st.session_state["station"]:
            st.session_state["station"] = clicked
            st.rerun()

    st.subheader(f"📍 {st.session_state['station']}")
    render_station(reg, st.session_state["station"], p, key="user")


# ------------------------------------------------------------ mode administrateur
def admin_authenticated() -> bool:
    try:
        pwd = os.environ.get("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD")
    except Exception:
        pwd = os.environ.get("ADMIN_PASSWORD")
    if not pwd:
        st.warning("Aucun mot de passe défini (variable ADMIN_PASSWORD) : l'espace administrateur est ouvert.")
        return True
    if st.session_state.get("is_admin"):
        return True
    entered = st.text_input("Mot de passe administrateur", type="password")
    if entered:
        if entered == pwd:
            st.session_state["is_admin"] = True
            st.rerun()
        st.error("Mot de passe incorrect.")
    return False


def page_admin(reg: dict, p: dict) -> None:
    st.title("🛠️ Espace administrateur")
    if not admin_authenticated():
        return

    st.subheader("Téléverser des données pour un point")
    names = list(reg["stations"]) + [NEW_POINT]
    station = st.selectbox("Point stratégique", names)
    if station == NEW_POINT:
        name = st.text_input("Nom du nouveau point").strip()
        lat0, lon0 = MAP_CENTER
    else:
        name = station
        lat0, lon0 = reg["stations"][station]
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", value=float(lat0), format="%.4f")
    lon = c2.number_input("Longitude", value=float(lon0), format="%.4f")

    has_ref = any(d["station"] == name and d["role"] == "reference" for d in reg["datasets"])
    if has_ref:
        role_label = st.radio("Type de jeu", ["Nouvelle mesure (à contrôler)", "Nouvelle référence"], horizontal=True)
        role = "reference" if role_label == "Nouvelle référence" else "new"
    else:
        st.info("Ce point n'a pas encore de référence : ce jeu sera enregistré comme **référence**.")
        role = "reference"

    up = st.file_uploader("Fichier CSV ou NetCDF (sonde / ADCP)", type=["csv", "txt", "nc", "cdf"])
    if up is None:
        st.caption("En attente d'un fichier.")
    elif not name:
        st.warning("Donnez un nom au nouveau point.")
    else:
        raw = up.getvalue()
        try:
            df = parse_table(raw, up.name)
        except Exception as e:
            st.error(f"Fichier illisible : {e}")
            df = None
        if df is not None:
            if len(df) == 0 or len(numeric_cols(df)) == 0:
                st.error("Aucune donnée numérique exploitable dans ce fichier.")
            else:
                st.success(f"{len(df)} lignes · {len(numeric_cols(df))} paramètres détectés")
                st.dataframe(df.head(10), use_container_width=True)
                if st.button("💾 Enregistrer et analyser", type="primary"):
                    reg["stations"][name] = [lat, lon]
                    entry = add_dataset(reg, name, role, up.name, raw)
                    st.session_state["last_saved"] = entry["id"]
                    st.session_state["station"] = name
                    st.rerun()

    saved = next((d for d in reg["datasets"] if d["id"] == st.session_state.get("last_saved")), None)
    if saved:
        st.divider()
        st.subheader(f"Résultat de l'analyse — {saved['station']}")
        render_station(reg, saved["station"], p, key="admin")

    st.divider()
    st.subheader("Jeux de données enregistrés")
    if not reg["datasets"]:
        st.caption("Aucun jeu enregistré.")
    for d in list(reg["datasets"]):
        a, b = st.columns([5, 1])
        a.write(f"**{d['station']}** — {ds_label(d)}")
        if b.button("Supprimer", key=f"del_{d['id']}"):
            delete_dataset(reg, d["id"])
            st.rerun()


def main() -> None:
    reg = load_registry()
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Mode", ["👤 Utilisateur / Analyste", "🛠️ Administrateur"])
    p = sidebar_params()
    if mode.startswith("👤"):
        page_user(reg, p)
    else:
        page_admin(reg, p)


if __name__ == "__main__":
    main()
