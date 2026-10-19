"""Plotly figures. Every chart in the app is paired with a table view."""
import numpy as np
import plotly.graph_objects as go

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DE_EMPHASIS = "#c3c2b7"
CRITICAL = "#d03b3b"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 3},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _layout(fig, title, x_title=None, y_title=None, height=380, legend=True):
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": TEXT}, "x": 0, "xanchor": "left"},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": FONT, "color": TEXT_2, "size": 12},
        height=height, margin={"l": 10, "r": 30, "t": 56, "b": 10},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1, "font": {"color": TEXT_2}},
        hoverlabel={"bgcolor": "white", "font": {"family": FONT, "color": TEXT}, "bordercolor": GRID},
        bargap=0.35,
    )
    axis = {"gridcolor": GRID, "gridwidth": 1, "zeroline": False, "linecolor": AXIS, "tickfont": {"color": MUTED},
            "title": {"font": {"color": TEXT_2}}, "automargin": True}
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


def _method_labels(df, proposed_code):
    return [f"<b>{r.Code} · {r.Method}</b>" if r.Code == proposed_code else f"{r.Code} · {r.Method}" for r in df.itertuples()]


def rejected_by_method(table, proposed_code):
    """Horizontal stacked bars: students without a school, split by reason."""
    df = table.iloc[::-1]
    labels = _method_labels(df, proposed_code)
    fig = go.Figure()
    fig.add_bar(y=labels, x=df["rejected_full"], orientation="h", name="Schools in radius full",
                marker={"color": SERIES[0], "cornerradius": 4},
                hovertemplate="%{y}<br>Schools full: %{x:.1f}<extra></extra>")
    fig.add_bar(y=labels, x=df["rejected_no_school"], orientation="h", name="No school within radius",
                marker={"color": SERIES[1], "cornerradius": 4},
                hovertemplate="%{y}<br>No school in radius: %{x:.1f}<extra></extra>")
    fig.update_layout(barmode="stack", legend={"traceorder": "normal"})
    totals = df["rejected"].to_numpy()
    fig.add_scatter(y=labels, x=totals, mode="text", text=[f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}" for v in totals],
                    textposition="middle right", textfont={"color": TEXT_2}, showlegend=False, hoverinfo="skip")
    fig.update_xaxes(range=[0, max(totals.max() * 1.15, 1)])
    return _layout(fig, "Students without a school (✕) by method", "Students", None, height=360)


def metric_by_method(table, column, title, x_title, proposed_code, fmt=".3f"):
    df = table.iloc[::-1]
    labels = _method_labels(df, proposed_code)
    colors = [SERIES[0] if k == proposed_code else DE_EMPHASIS for k in df["Code"]]
    fig = go.Figure(go.Bar(y=labels, x=df[column], orientation="h", marker={"color": colors, "cornerradius": 4},
                           hovertemplate="%{y}<br>%{x:" + fmt + "}<extra></extra>"))
    proposed_value = table.loc[table["Code"] == proposed_code, column]
    if len(proposed_value):
        fig.add_annotation(y=labels[list(df["Code"]).index(proposed_code)], x=float(proposed_value.iloc[0]),
                           text=format(float(proposed_value.iloc[0]), fmt), showarrow=False, xanchor="left", xshift=6,
                           font={"color": TEXT})
    return _layout(fig, title, x_title, None, height=340, legend=False)


def lines_by_method(df, x, y, order, names, title, x_title, y_title, x_labels=None):
    fig = go.Figure()
    for idx, code in enumerate(order):
        part = df[df["Code"] == code].sort_values(x)
        xs = part[x].to_numpy()
        fig.add_scatter(x=xs, y=part[y], mode="lines+markers", name=f"{code} · {names[code]}",
                        line={"color": SERIES[idx], "width": 2}, marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                        hovertemplate=f"{code} · {names[code]}<br>{x_title}: %{{x}}<br>{y_title}: %{{y:.3f}}<extra></extra>")
        if len(part):
            fig.add_annotation(x=xs[-1], y=part[y].iloc[-1], text=code, showarrow=False, xanchor="left", xshift=8,
                               font={"color": TEXT_2, "size": 11})
    if x_labels:
        fig.update_xaxes(tickmode="array", tickvals=list(x_labels.keys()), ticktext=list(x_labels.values()))
    return _layout(fig, title, x_title, y_title)


def distance_histogram(distances, radius, title="Distance of admitted students"):
    fig = go.Figure(go.Histogram(x=distances, nbinsx=30, marker={"color": SERIES[0], "line": {"color": SURFACE, "width": 2}},
                                 hovertemplate="%{x} km<br>%{y} students<extra></extra>"))
    if radius is not None:
        fig.add_vline(x=radius, line={"color": TEXT_2, "width": 1, "dash": "dash"})
        fig.add_annotation(x=radius, y=1, yref="paper", text=f"Radius {radius:g} km", showarrow=False,
                           xanchor="right", xshift=-4, font={"color": TEXT_2})
    return _layout(fig, title, "Distance to school (km)", "Students", legend=False)


def utilization(util):
    df = util.sort_values("Fill rate (%)")
    fig = go.Figure()
    fig.add_bar(y=df["School"], x=df["Quota"], orientation="h", name="Quota",
                marker={"color": GRID, "cornerradius": 4}, hovertemplate="%{y}<br>Quota %{x}<extra></extra>")
    fig.add_bar(y=df["School"], x=df["Filled"], orientation="h", name="Filled",
                marker={"color": SERIES[0], "cornerradius": 4},
                customdata=df["Fill rate (%)"], hovertemplate="%{y}<br>Filled %{x} (%{customdata:.0f}%)<extra></extra>")
    fig.update_layout(barmode="overlay", bargap=0.25)
    return _layout(fig, "Seats filled per school", "Seats", None, height=max(420, 22 * len(df) + 80))


def scenario_grouped(df, order, names):
    fig = go.Figure()
    scenarios = list(dict.fromkeys(df["Scenario"]))
    for idx, code in enumerate(order):
        part = df[df["Code"] == code].set_index("Scenario").reindex(scenarios)
        fig.add_bar(x=scenarios, y=part["rejected"], name=f"{code} · {names[code]}",
                    marker={"color": SERIES[idx], "cornerradius": 4},
                    hovertemplate=f"{code} · {names[code]}<br>%{{x}}<br>✕ %{{y:.1f}} students<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    return _layout(fig, "Students without a school (✕) per capacity scenario", None, "Students", height=400)


def resampling_box(runs_df, order, names, proposed_code):
    fig = go.Figure()
    for code in order:
        part = runs_df[runs_df["Code"] == code]
        color = SERIES[0] if code == proposed_code else MUTED
        fig.add_box(y=part["rejected"], name=f"{code} · {names[code]}", marker={"color": color, "size": 6},
                    line={"color": color, "width": 2}, fillcolor="rgba(0,0,0,0)", boxpoints="all", jitter=0.4,
                    pointpos=0, hovertemplate="%{x}<br>✕ %{y} students<extra></extra>")
    return _layout(fig, "Students without a school (✕) in every run", None, "Students", legend=False, height=400)


def runtime_line(df):
    fig = go.Figure(go.Scatter(x=df["Students"], y=df["Proposed total runtime (s)"], mode="lines+markers",
                               line={"color": SERIES[0], "width": 2}, marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                               hovertemplate="%{x} students<br>%{y:.2f} s<extra></extra>"))
    last = df.iloc[-1]
    fig.add_annotation(x=last["Students"], y=last["Proposed total runtime (s)"], text=f"{last['Proposed total runtime (s)']:.2f} s",
                       showarrow=False, xanchor="left", xshift=8, font={"color": TEXT_2})
    return _layout(fig, "Runtime of the proposed method", "Students", "Seconds", legend=False)


def ablation_steps(df):
    ok = df[df["Status"] == "Optimal"]
    fig = go.Figure()
    fig.add_scatter(x=ok["Code"], y=ok["beyond_radius"], mode="lines+markers", name="Admitted beyond radius",
                    line={"color": SERIES[1], "width": 2}, marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                    hovertemplate="%{x}<br>Beyond radius: %{y}<extra></extra>")
    fig.add_scatter(x=ok["Code"], y=ok["rejected"], mode="lines+markers", name="No school (✕)",
                    line={"color": SERIES[0], "width": 2}, marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
                    hovertemplate="%{x}<br>✕: %{y}<extra></extra>")
    return _layout(fig, "Effect of each component: radius violations vs. students ✕", "Ablation configuration", "Students")


def ablation_distance(df):
    ok = df[df["Status"] == "Optimal"]
    fig = go.Figure(go.Bar(x=ok["Code"], y=ok["max_km"], marker={"color": SERIES[0], "cornerradius": 4},
                           hovertemplate="%{x}<br>Max distance %{y:.3f} km<extra></extra>"))
    return _layout(fig, "Longest trip of an admitted student", "Ablation configuration", "km", legend=False)


def sweep_x_labels(radii):
    finite = [r for r in radii if np.isfinite(r)]
    top = (max(finite) if finite else 1.0) + 0.5
    mapping = {r if np.isfinite(r) else top: (f"{r:g}" if np.isfinite(r) else "∞") for r in radii}
    return top, mapping
