#!/usr/bin/env python3
"""Build an offline UMAP atlas from a downloaded illumination archive."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import umap
from plotly.offline import plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    z = np.load(archive / "embeddings.npy", allow_pickle=False)
    novelty = np.load(archive / "novelty.npy", allow_pickle=False)
    selected = json.loads((archive / "representatives.json").read_text())
    assert z.shape[0] == len(novelty) == 5000
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.05, metric="cosine", random_state=0, n_components=2
    )
    xy = reducer.fit_transform(z)
    np.save(archive / "umap_coordinates.npy", xy)
    settings = dict(
        n_neighbors=30,
        min_dist=0.05,
        metric="cosine",
        random_state=0,
        n_components=2,
        umap_version=umap.__version__,
    )
    (archive / "umap_config.json").write_text(json.dumps(settings, indent=2))
    with (archive / "atlas.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("archive_index", "umap_x", "umap_y", "novelty"))
        writer.writerows(
            (i, float(x), float(y), float(novelty[i])) for i, (x, y) in enumerate(xy)
        )

    fig, ax = plt.subplots(figsize=(14, 10), dpi=180)
    dots = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=novelty,
        s=7,
        alpha=0.75,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        xy[selected, 0],
        xy[selected, 1],
        facecolors="none",
        edgecolors="red",
        s=75,
        linewidths=1.2,
    )
    ax.set(
        title="Lenia illumination · MobileCLIP2-S2 · 5,000 configurations",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
    )
    fig.colorbar(dots, ax=ax, label="Novelty (1 − nearest cosine)")
    fig.tight_layout()
    fig.savefig(archive / "atlas.png")
    plt.close(fig)

    custom = np.column_stack((np.arange(len(xy)), novelty))
    chart = go.Figure(
        go.Scattergl(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                opacity=0.8,
                color=novelty,
                colorscale="Viridis",
                colorbar=dict(title="Novelty"),
            ),
            customdata=custom,
            hovertemplate="Archive %{customdata[0]:.0f}<br>Novelty %{customdata[1]:.4f}<extra></extra>",
        )
    )
    chart.update_layout(
        title="Lenia illumination · MobileCLIP2-S2",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white",
        width=1100,
        height=800,
    )
    plot_div = plot(chart, include_plotlyjs=True, output_type="div")
    gallery = []
    for number, index in enumerate(selected, start=1):
        stem = f"{number:02d}-archive-{index:04d}"
        gallery.append(
            f'<a href="representative_videos/{stem}.mp4">'
            f'<img src="representative_videos/{stem}.png" alt="Archive {index}">'
            f"<span>#{index} · novelty {novelty[index]:.4f}</span></a>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Lenia illumination atlas</title><style>body{font:16px sans-serif;"
        "max-width:1160px;margin:auto;padding:24px} .gallery{display:grid;"
        "grid-template-columns:repeat(6,1fr);gap:12px} .gallery a{color:#222;"
        "text-decoration:none;text-align:center}.gallery img{width:100%;"
        "image-rendering:pixelated}.gallery span{display:block;font-size:13px}"
        "</style></head><body><h1>Lenia illumination atlas</h1>"
        "<p>5,000 configurations; color shows embedding novelty. "
        "Click a representative preview for its full simulation video.</p>"
        + plot_div
        + "<h2>Diverse representatives</h2><div class='gallery'>"
        + "".join(gallery)
        + "</div></body></html>"
    )
    (archive / "atlas.html").write_text(page)
    print(f"Saved UMAP atlas for {len(xy)} configurations to {archive}")


if __name__ == "__main__":
    main()
