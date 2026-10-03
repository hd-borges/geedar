# -*- coding: utf-8 -*-
"""
Provides a catalog of 'local algorithms' to be available for GEEDaR.

Local algorithms are those applied to the downloaded time series of satellite
data (in general, reflectance time series). Each algorithm has an id code.
They are used in GEEDaR as a parameter of Demand objects.*

You may edit this module to include your own algorithms.

Note: you may run GEEDaR with no local algorithm if you don't need to apply 
any model or algorithm to the downloaded time series, in which case you 
must use the local algorithm identified with number zero in the catalog below.
If you edit this module, I recommend you keep local the algorithm #0 as the 
one for "doing nothing".

* On Demand objects, see the class 'Demand' in 'geedar_core.demand'.

"""

__author__ = "Dhalton Ventura"
__copyright__ = "Copyright 2026 HidroSat Project"
__credits__ = ["Dhalton Ventura"]
__license__ = "MIT"
__version__ = "2.1"
__maintainer__ = "Dhalton Ventura"
__email__ = "dhalton.ventura@ana.gov.br"
__status__ = "Beta"


#%% Import

import math
import pandas as pd
import numpy as np
from geedar_core.local_algorithm import LocalAlgorithm


#%% Export

__all__ = ["local_algo_catalog"]


#%% Local algorithms to be used by GEEDaR

# The functions below must take a Pandas dataframe and a dictionary of
# options. And they must also return a Pandas dataframe corresponding to the 
# input dataframe with added columns for the result variables.

def do_nothing(df, options):
    return df

def madeira_2013(df, options):
    #return_df = df.assign(SSS=lambda c: 1020*(c["NIR"]/c["red"])**2.94)
    return_df = df.eval("SSS = 1020*(NIR/red)**2.94")
    return return_df


ARTICLE_COEF = 759.12
ARTICLE_EXP = 1.9189
RATIO_MIN = 0.03
RATIO_MAX = 1.20
TRANSITION_LOW = 80.0
TRANSITION_HIGH = 100.0
ALTO_RED_SLOPE = 0.032800385674


def _article_branch(red, nir):
    red = pd.to_numeric(red, errors="coerce")
    nir = pd.to_numeric(nir, errors="coerce")
    ratio = nir / red.replace(0, np.nan)
    valid = ratio.between(RATIO_MIN, RATIO_MAX)
    return ARTICLE_COEF * np.power(ratio.where(valid), ARTICLE_EXP)


def _smoothstep(driver):
    driver = pd.to_numeric(driver, errors="coerce")
    t = ((driver - TRANSITION_LOW) /
         (TRANSITION_HIGH - TRANSITION_LOW)).clip(lower=0.0, upper=1.0)
    return t * t * (3.0 - 2.0 * t)


def _apply_quality_filter(df):
    if "qual_flag" in df.columns:
        return pd.to_numeric(df["qual_flag"], errors="coerce").isin([1, 2])
    return pd.Series(True, index=df.index)


def sss_alto_solimoes_red_artigo(df, options):
    out = df.copy()
    red = pd.to_numeric(out["red"], errors="coerce")
    red = red.where(red > 0.0)
    red_branch = ALTO_RED_SLOPE * red
    article = _article_branch(red, out["NIR"])
    weight = _smoothstep(article)
    prediction = red_branch.copy()
    both = red_branch.notna() & article.notna()
    prediction.loc[both] = (
        (1.0 - weight.loc[both]) * red_branch.loc[both]
        + weight.loc[both] * article.loc[both]
    )
    out["SSS"] = prediction
    out.loc[~_apply_quality_filter(out), "SSS"] = np.nan
    return out


def sss_medio_solimoes_red_artigo(df, options):
    out = df.copy()
    red = pd.to_numeric(out["red"], errors="coerce")
    red = red.where(red > 0.0)
    red_branch = 0.09918086861006252 * red
    article = _article_branch(red, out["NIR"])
    weight = _smoothstep(red_branch)
    prediction = red_branch.copy()
    both = red_branch.notna() & article.notna() & weight.gt(0.0)
    prediction.loc[both] = (
        (1.0 - weight.loc[both]) * red_branch.loc[both]
        + weight.loc[both] * article.loc[both]
    )
    out["SSS"] = prediction
    out.loc[~_apply_quality_filter(out), "SSS"] = np.nan
    return out


def sss_baixo_amazonas_red_nir(df, options):
    out = df.copy()
    red = pd.to_numeric(out["red"], errors="coerce")
    red = red.where(red > 0.0)
    nir = pd.to_numeric(out["NIR"], errors="coerce")
    nir = nir.where(nir > 0.0)
    red_branch = 0.06920309102554381 * red
    nir_branch = 0.16802259391468988 * nir
    weight = _smoothstep(red_branch)
    prediction = red_branch.copy()
    both = red_branch.notna() & nir_branch.notna() & weight.gt(0.0)
    prediction.loc[both] = (
        (1.0 - weight.loc[both]) * red_branch.loc[both]
        + weight.loc[both] * nir_branch.loc[both]
    )
    out["SSS"] = prediction
    out.loc[~_apply_quality_filter(out), "SSS"] = np.nan
    return out


def ss_paraguai_red(df, options):
    out = df.copy()
    red = pd.to_numeric(out["red"], errors="coerce")
    out["SS"] = 0.0562 * red.where(red > 0.0)
    out.loc[~_apply_quality_filter(out), "SS"] = np.nan
    return out

def old_hidrosat_chla(df, options):
    return_df = df.eval("Chla = 4.3957 + 0.213*(red - (red**2)/green) + 0.0004*(red - (red**2)/green)**2")
    return return_df

def sss_solimoes_2018(df, options):
    return_df = df.eval("SSS = 759.12*(NIR/red)**1.9189")
    return return_df

def sss_obidos_2009(df, options):
    return_df = df.eval("SSS = 0.2019*NIR - 14.222")
    return return_df

def turb_paranapanema_2019(df, options):
    return_df = df.eval("Turb = 2.45 * exp(0.00223*red)")
    return return_df

def brumadinho_2020_simp(df, options):
    scale_factor = np.pi * 10000
    rejeito = (scale_factor / df['green']) - (scale_factor / df['red'])
    ind1 = (df['NIR'] / scale_factor) * (df['red'] / df['green'])
    ind2 = df['NIR'] / df['red']
    normal_case = 18381 * (ind1 ** 2) + 3874.8 * ind1
    special_case = 9205.5 * (ind2 ** 2) - 9253.8 * ind2
    condition = (ind2 >= 0.9) & (rejeito != 0)
    df['SSS'] = np.where(condition, special_case, normal_case)
    df['SSS'] = df['SSS'].replace([np.inf, -np.inf], np.nan)    
    return df

def acudes_vars_2020(df, options):
    df['ISS'] = (df['red'] - df['NIR']) * 0.059 + (df['green'] - df['NIR']) * -0.0245 + 0.74
    df['SSS'] = df['ISS'].clip(lower=0)
    df['SSS'] = (df['red'] - df['blue']) * 0.06318 + df['green'] * 0.009793 + 1.363
    df['SSS'] = np.maximum(df['SSS'], df['ISS'])
    df['OSS'] = df['SSS'] - df['ISS']
    df['Chla'] = df['green'] * 0.0937 + df['ISS'] * -3.752 - 10.92
    df['Chla'] = df['Chla'].clip(lower=0)
    df['Biomass'] = np.exp(df['Chla'] * 0.02386) * 1.55465
    df[['ISS', 'SSS', 'OSS', 'Chla', 'Biomass']] = df[['ISS', 'SSS', 'OSS', 'Chla', 'Biomass']].replace([np.inf, -np.inf], np.nan)
    return df

def acudes_chla_2022(df, options):
    return_df = df.eval("Chla = -4.227 + 0.1396*green - 0.1006*red")
    return return_df

def sss_doce_2022(df, options):
    c1 = (2180-645)/(2180-469)
    c2 = (645-469)/(2180-469)
    rh2 = df['red'] - (df['blue'] * c1 + df['wl2000'] * c2)
    sss_base = (df['NIR'] / df['green']) * 0.6798 + rh2 * 0.001391
    df['SSS'] = (np.exp(sss_base) * 6.44).clip(lower=0)
    df['SSS'] = df['SSS'].replace([np.inf, -np.inf], np.nan)
    return df

def madeira_2023(df, options):
    return_df = df.eval("SSS = 2.3574 * exp(6.1727 * NIR/red)")
    return return_df


# Médias de Rrs (sr^-1) dos 4 tipos ópticos de SPM (Cordeiro), bandas S2 B1-B8A, conforme
# getpak/data/Means_OWT_Cordeiro_S2A_SPM.json (GET-Pak, commit 7e04ba3).
GETPAK_OWT_MEANS = np.array([
    [0.0043, 0.0058, 0.0102, 0.0077, 0.0067, 0.0024, 0.0024, 0.0015, 0.0013],
    [0.0105, 0.0146, 0.0271, 0.0323, 0.0297, 0.0140, 0.0145, 0.0097, 0.0084],
    [0.0151, 0.0198, 0.0352, 0.0459, 0.0455, 0.0317, 0.0325, 0.0258, 0.0236],
    [0.0224, 0.0287, 0.0519, 0.0722, 0.0735, 0.0530, 0.0554, 0.0433, 0.0393]])
GETPAK_BANDS = ["wl440", "blue", "green", "red", "wl705", "wl740", "wl780", "NIR", "wl800"]


def sss_getpak_owt_blend(df, options):
    """SPM superficial pelas inversões do GET-Pak (Tavares et al., 2025) com mistura suave
    entre tipos ópticos: o espectro mediano da data (Rrs = valor * scale / pi) é comparado às
    médias dos 4 OWT por distância euclidiana; a estimativa é a média das inversões dos 2 OWT
    mais próximos ponderada por 1/d². OWT 1: Jiang 2021 verde; 2: Jiang 2021 vermelho;
    3: Zhang 2014 (B5); 4: Binding 2010 (B6). Experimental (coeficientes da literatura)."""
    import getpak_inversions as gp
    out = df.copy()
    scale = options.get("scale", 1e-4)
    rrs = {b: pd.to_numeric(out[b], errors="coerce").to_numpy(dtype=float) * scale / np.pi
           for b in GETPAK_BANDS}
    x = np.column_stack([rrs[b] for b in GETPAK_BANDS])
    valid = np.isfinite(x).all(axis=1) & (rrs["red"] > 0)
    sss = np.full(len(out), np.nan)
    if valid.any():
        xv = x[valid]
        d = np.linalg.norm(xv[:, None, :] - GETPAK_OWT_MEANS[None, :, :], axis=2)
        r = {b: rrs[b][valid] for b in GETPAK_BANDS}
        kw = dict(Aerosol=r["wl440"], Blue=r["blue"], Green=r["green"], Red=r["red"])
        with np.errstate(all="ignore"):
            alg = np.column_stack([gp.spm_jiang2021_green(**kw), gp.spm_jiang2021_red(**kw),
                                   gp.spm_zhang2014(RedEdge1=r["wl705"]),
                                   gp.spm_binding2010(RedEdge2=r["wl740"])])
        idx = np.argsort(d, axis=1)[:, :2]
        w = 1.0 / np.maximum(np.take_along_axis(d, idx, axis=1), 1e-9) ** 2
        w = w / w.sum(axis=1, keepdims=True)
        a2 = np.take_along_axis(alg, idx, axis=1)
        blend = np.where(np.isfinite(a2).all(axis=1), (w * a2).sum(axis=1), np.nan)
        sss[valid] = np.where(blend >= 0, blend, np.nan)
    out["SSS"] = sss
    out.loc[~_apply_quality_filter(out).to_numpy(), "SSS"] = np.nan
    return out


#%% Build a catalog of algorithms

_algo_list = [
     {
        "algo_code": 0,
        "name": "None",
        "description": "No local algorithm is applied.",
        "ref": "",
        "required_bands": [],
        "applicable_suffixes": [],
        "function": do_nothing,
        "options": None
     },
     {
        "algo_code": 1,
        "name": "Old HidroSat chla",
        "description": "Estiamtes chlorophyll-a concentration (ug/L) in "
            + "Brazilian semiarid reservoirs through this model: "
            + "4.3957 + 0.213*(R - R^2/G) + 0.0004*(R - R^2/G)^2",
        "ref": "",
        "required_bands": ["red", "green"],
        "applicable_suffixes": ["median","mean"],
        "function": old_hidrosat_chla,
        "options": None
     },
     {
        "algo_code": 2,
        "name": "SSS Solimões 2018",
        "description": "Estimates the surface suspended solids concentration "
            + "in the Solimões River. Model: 759.12*(NIR/red)^1.9189",
        "ref": "Villar, R.E. et al. Spatio-temporal monitoring of suspended "
            + "sediments in the Solimoes River (2000-2014). Comptes Rendus " 
            + "Geoscience, v. 350, n. 1-2, p. 4-12, 2018.",
        "required_bands": ["red", "NIR"],
        "applicable_suffixes": ["median","mean"],
        "function": sss_solimoes_2018,
        "options": None
     },
     {
        "algo_code": 3,
        "name": "SSS Madeira",
        "description": "Estimates the surface suspended solids concentration "
            + "in the Madeira River. Model: 1020*(NIR/red)^2.94",
        "ref": "Villar, R.E.; et al. A study of sediment transport in the "
            + "Madeira River, Brazil, using MODIS remote-sensing images. "
            + "Journal of South American Earth Sciences, v. 44, p. 45-54, "
            + "2013.",
        "required_bands": ["red", "NIR"],
        "applicable_suffixes": ["median","mean"],
        "function": madeira_2013,
        "options": None
     },
     {
        "algo_code": 4,
        "name": "SSS Óbidos 2009",
        "description": "Estimates the surface suspended solids concentration "
            + "in the Amazon River, near Óbidos. Model: "
            + "0.2019*NIR - 14.222",
        "ref": "Martinez, J. M. et al. Increase in suspended sediment "
            + "discharge of the Amazon River assessed by monitoring network "
            + "and satellite data. Catena, v. 79, n. 3, p. 257-264, 2009.",
        "required_bands": ["NIR"],
        "applicable_suffixes": ["median","mean"],
        "function": sss_obidos_2009,
        "options": None
     },
     {
        "algo_code": 5,
        "name": "Turb Paranapanema",
        "description": "Estimates the surface turbidity in reservoirs along "
            + "the Paranapnema river. Model: 2.45*EXP(0.00223*red)",
        "ref": "Condé, R.C. et al. Indirect Assessment of Sedimentation in "
            + "Hydropower Dams Using MODIS Remote Sensing Images. Remote "
            + "Sensing, v.11, n. 3, 2019.",
        "required_bands": ["red"],
        "applicable_suffixes": ["median","mean"],
        "function": turb_paranapanema_2019,
        "options": None
     },
     {
        "algo_code": 10,
        "name": "Brumadinho 2020 simp",
        "description": "Estimates the surface suspended solids concentration "
            + "in the Paraopeba River, accounting for the presence of mining "
            + "waste after the 2019 disaster.",
        "ref": "VENTURA, 2020 (Unpublished).",
        "required_bands": ["red", "green", "NIR"],
        "applicable_suffixes": ["median","mean"],
        "function": brumadinho_2020_simp,
        "options": None
     },
     {
        "algo_code": 11,
        "name": "Açudes SSS-ISS-OSS-Chla-Biomass 2020",
        "description": "Estimates four parameters for the waters of Brazilian "
            + "semiarid reservoirs: surface suspended solids, its organic and "
            + "inorganic fractions, and chlorophyll-a.",
        "ref": "VENTURA, 2020 (Unpublished).",
        "required_bands": ["blue", "green", "red", "NIR"],
        "applicable_suffixes": ["median","mean"],
        "function": acudes_vars_2020,
        "options": None
     },
     {
        "algo_code": 12,
        "name": "Açudes Chla 2022",
        "description": "Estimates chlorophyll-a in Brazilian semiarid "
            + "reservoirs. Model: -4.227 + 0.1396*G - 0.1006*R",
        "ref": "VENTURA, 2022 (Unpublished).",
        "required_bands": ["green", "red"],
        "applicable_suffixes": ["median","mean"],
        "function": acudes_chla_2022,
        "options": None
     },
     {
        "algo_code": 13,
        "name": "Rio Doce 2022",
        "description": "Estimates suspended sediment in the Doce river. Model "
            + "calibration was done with data covering a large range: 0-1000. "
            + "Model: 6.44*exp(0.6798*IR/G + 0.001391*Rh2)",
        "ref": "MENDES et al., 2022 (Unpublished).",
        "required_bands": ["blue", "green", "red", "NIR", "wl2000"],
        "applicable_suffixes": ["median","mean"],
        "function": sss_doce_2022,
        "options": None
     },
     {
        "algo_code": 14,
        "name": "Madeira 2023",
        "description": "Estimates suspended sediment along the Madeira river. "
            + "The algorithm adapts to the satellite product. Models were "
            + "calibrated for concentrations as high as 3500 mg/L which "
            + "occured at the Rurrenabaque station.",
        "ref": "VENTURA, 2023 (Unpublished).",
        "required_bands": ["blue", "green", "red", "NIR", "wl2000"],
        "applicable_suffixes": ["median","mean"],
        "function": madeira_2023,
        "options": None
     },
     {
        "algo_code": 20,
        "name": "SSS Alto Solimoes - RED/artigo",
        "description": "Regional SSS model with RED and NIR/RED branches.",
        "ref": "GEEDaR/HidroSat internal calibration, 2026.",
        "required_bands": ["red", "NIR", "qual_flag"],
        "applicable_suffixes": ["median"],
        "function": sss_alto_solimoes_red_artigo,
        "options": None
     },
     {
        "algo_code": 21,
        "name": "SSS Medio Solimoes - RED/artigo",
        "description": "Regional SSS model with RED and NIR/RED branches.",
        "ref": "GEEDaR/HidroSat internal calibration, 2026.",
        "required_bands": ["red", "NIR", "qual_flag"],
        "applicable_suffixes": ["median"],
        "function": sss_medio_solimoes_red_artigo,
        "options": None
     },
     {
        "algo_code": 22,
        "name": "SSS Baixo Amazonas - RED/NIR",
        "description": "Regional SSS model with linear RED and NIR branches.",
        "ref": "GEEDaR/HidroSat internal calibration, 2026.",
        "required_bands": ["red", "NIR", "qual_flag"],
        "applicable_suffixes": ["median"],
        "function": sss_baixo_amazonas_red_nir,
        "options": None
     },
     {
        "algo_code": 23,
        "name": "SS Bacia do Paraguai - RED",
        "description": "Regional SS model for the Paraguay River basin. "
            + "Model: 0.0562*RED",
        "ref": "",
        "required_bands": ["red", "qual_flag"],
        "applicable_suffixes": ["median"],
        "function": ss_paraguai_red,
        "options": None
     },
     {
        "algo_code": 24,
        "name": "SSS GET-Pak OWT mistura (experimental)",
        "description": "Experimental. Surface SPM from the GET-Pak inversions "
            + "(4 optical water types; Jiang 2021 green/red, Zhang 2014, "
            + "Binding 2010) with smooth blending between the two closest "
            + "types (1/d^2). Requires Sentinel-2 bands B1-B8A as reflectance "
            + "x 10000 (Rrs = value * 1e-4 / pi). Literature coefficients, "
            + "not calibrated for Brazilian rivers.",
        "ref": "Tavares, M.H. et al. A Framework to Retrieve Water Quality "
            + "Parameters in Small, Optically Diverse Freshwater Ecosystems "
            + "Using Sentinel-2 MSI Imagery. Remote Sensing 17(15):2729, 2025; "
            + "GET-Pak (SNO-HYBAM), commit 7e04ba3.",
        "required_bands": GETPAK_BANDS + ["qual_flag"],
        "applicable_suffixes": ["median"],
        "function": sss_getpak_owt_blend,
        "options": {"scale": 1e-4}
     }
]

local_algo_catalog = {a["algo_code"]: LocalAlgorithm(a) 
    for a in _algo_list}
