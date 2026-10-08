"""One shared, quiet plot style for every figure in the project (light surface, recessive grid and axes, one
blue series, orange only to highlight), plus the project's standard figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the reference palette
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RENT = "Rent (Rs / month)"


def apply_style():
    """Set the shared matplotlib style (call once per notebook; the plotting helpers call it too)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE]),
            "legend.frameon": False,
        }
    )


_PREFIXES = {
    "Point_of_Contact_Contact_": "Contact: ",
    "Furnishing_Status_": "Furnishing: ",
    "Tenant_Preferred_": "Tenant: ",
    "Area_Type_": "Area type: ",
    "City_": "City: ",
    "logsize_x_": "log size × ",
}
_NAMES = {
    "log_size": "log size",
    "log_bhk": "log BHK",
    "log_bath": "log bathrooms",
    "log_total_floors": "log total floors",
    "current_floor": "floor",
    "total_floors": "total floors",
    "floor_ratio": "floor ÷ total floors",
    "is_basement": "basement",
    "locality_target_enc": "locality (target-encoded)",
}


def pretty_feature(name):
    """Readable label for a feature name, e.g. 'Point_of_Contact_Contact_Agent' -> 'Contact: Agent'."""
    if name in _NAMES:
        return _NAMES[name]
    for prefix, label in _PREFIXES.items():
        if name.startswith(prefix):
            return label + name[len(prefix) :].replace("_", " ").replace("infrequent sklearn", "other (rare)")
    return name.replace("_", " ")


def _rupees(ax, axis="both"):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def _save(fig, save_path):
    if save_path:
        config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_diagnostics(model, pred, data, model_name, save_path=None):
    """Three panels on the test split of one seed: actual vs predicted, residuals vs predicted, and the
    model's coefficients or feature importances (top 15)."""
    apply_style()
    inner = (model.regressor_ if hasattr(model, "regressor_") else model).named_steps["model"]
    names = data["feature_names"]
    if hasattr(inner, "coef_"):
        importance, imp_label = pd.Series(np.ravel(inner.coef_), index=names), "Standardised coefficient (log rent)"
    elif hasattr(inner, "feature_importances_"):
        importance, imp_label = pd.Series(inner.feature_importances_, index=names), "Feature importance"
    else:
        importance, imp_label = None, ""
    actual, predicted = data["y_test"].to_numpy(), pred["test"]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    lim = [min(actual.min(), predicted.min()) * 0.9, max(actual.max(), predicted.max()) * 1.1]
    ax[0].plot(lim, lim, color=INK_2, lw=1, ls="--", label="perfect prediction")
    ax[0].scatter(actual, predicted, s=10, alpha=0.45, color=BLUE, edgecolors="none")
    ax[0].set(
        xscale="log",
        yscale="log",
        xlim=lim,
        ylim=lim,
        xlabel=f"Actual rent {RENT[5:]}",
        ylabel=f"Predicted rent {RENT[5:]}",
        title="Actual vs predicted",
    )
    _rupees(ax[0])
    ax[0].legend(loc="upper left")

    ax[1].axhline(0, color=INK_2, lw=1)
    ax[1].scatter(predicted, actual - predicted, s=10, alpha=0.45, color=BLUE, edgecolors="none")
    ax[1].set(
        xscale="log",
        xlabel=f"Predicted rent {RENT[5:]}",
        ylabel="Residual: actual − predicted (Rs)",
        title="Residuals vs predicted",
    )
    _rupees(ax[1])

    if importance is not None and np.any(importance != 0):
        top = importance.reindex(importance.abs().sort_values(ascending=False).index)[:15][::-1]
        ax[2].barh([pretty_feature(n) for n in top.index], top.values, color=BLUE, height=0.7)
        ax[2].axvline(0, color=INK_2, lw=0.8)
        ax[2].set(xlabel=imp_label, title="Top 15 features")
        ax[2].grid(axis="y", visible=False)
    else:
        ax[2].text(0.5, 0.5, "constant prediction —\nno features used", ha="center", va="center", color=INK_2)
        ax[2].set_axis_off()
    fig.suptitle(
        f"{model_name} · seed {data['seed']} · test split ({len(actual)} listings)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, save_path)


def plot_model_comparison(num, highlight=None, save_path=None):
    """Mean test RMSE per model with ± 1 std across seeds (bars sorted, best at the top).
    `num` is the numeric table from evaluation.read_results(); `highlight` is a model name to colour orange."""
    apply_style()
    d = num.sort_values("Test RMSE mean")
    colors = [ORANGE if m == highlight else BLUE for m in d["Model"]]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.2))
    ax.barh(
        d["Model"],
        d["Test RMSE mean"],
        xerr=d["Test RMSE std"],
        color=colors,
        height=0.65,
        error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3},
    )
    for y, (m, s) in enumerate(zip(d["Test RMSE mean"], d["Test RMSE std"], strict=True)):
        ax.text(
            m + s + d["Test RMSE mean"].max() * 0.01, y, f"{m:,.0f} ± {s:,.0f}", va="center", color=INK_2, fontsize=9
        )
    ax.invert_yaxis()
    ax.set_xlim(0, (d["Test RMSE mean"] + d["Test RMSE std"]).max() * 1.22)
    ax.set(
        xlabel="Test RMSE (Rs / month) — mean ± std over seeds " + ", ".join(map(str, config.SEEDS)),
        title="Test error by model (lower is better)",
    )
    ax.grid(axis="y", visible=False)
    _rupees(ax, "x")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_importance(importance, title, xlabel, save_path=None, top=15):
    """Horizontal bars of the `top` largest values of a Series (e.g. permutation importance)."""
    apply_style()
    s = importance.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(s) + 1.2))
    ax.barh([pretty_feature(n) for n in s.index], s.values, color=BLUE, height=0.65)
    ax.set(xlabel=xlabel, title=title)
    ax.grid(axis="y", visible=False)
    _rupees(ax, "x")
    fig.tight_layout()
    return _save(fig, save_path)
