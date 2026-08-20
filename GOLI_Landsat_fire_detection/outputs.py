from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as transform_coords
from shapely.geometry import Point


def write_raster(
    path: Path,
    array: np.ndarray,
    reference_profile: dict,
    dtype: str,
    nodata,
) -> None:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    output = array.copy()
    if np.issubdtype(np.dtype(dtype), np.floating):
        output = output.astype(dtype)
        output[~np.isfinite(output)] = nodata
    else:
        output = output.astype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)


def write_fire_points(
    path: Path,
    fire_mask: np.ndarray,
    fire_type: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
    ratio75: np.ndarray,
    selected_window: np.ndarray,
    scene_date: str,
    transform,
    crs,
) -> int:
    rows, cols = np.where(fire_mask == 1)

    path.parent.mkdir(parents=True, exist_ok=True)

    if rows.size == 0:
        empty = gpd.GeoDataFrame(
            {
                "longitude": [],
                "latitude": [],
                "fire_type": [],
                "red": [],
                "nir": [],
                "swir1": [],
                "swir2": [],
                "ratio75": [],
                "ctx_win": [],
                "date": [],
            },
            geometry=[],
            crs="EPSG:4326",
        )

        empty.to_file(path, encoding="utf-8")
        return 0

    xs, ys = xy(transform, rows, cols, offset="center")

    xs = list(xs)
    ys = list(ys)

    if crs is not None and str(crs).upper() not in {
        "EPSG:4326",
        "OGC:CRS84",
    }:
        longitudes, latitudes = transform_coords(
            crs,
            "EPSG:4326",
            xs,
            ys,
        )
    else:
        longitudes, latitudes = xs, ys

    geometries = [
        Point(float(lon), float(lat))
        for lon, lat in zip(longitudes, latitudes)
    ]

    gdf = gpd.GeoDataFrame(
        {
            "longitude": [float(v) for v in longitudes],
            "latitude": [float(v) for v in latitudes],
            "fire_type": [
                int(v) for v in fire_type[rows, cols]
            ],
            "red": [float(v) for v in red[rows, cols]],
            "nir": [float(v) for v in nir[rows, cols]],
            "swir1": [float(v) for v in swir1[rows, cols]],
            "swir2": [float(v) for v in swir2[rows, cols]],
            "ratio75": [
                float(v) if np.isfinite(v) else np.nan
                for v in ratio75[rows, cols]
            ],
            "ctx_win": [
                int(v) for v in selected_window[rows, cols]
            ],
            "date": [scene_date] * int(rows.size),
        },
        geometry=geometries,
        crs="EPSG:4326",
    )

    gdf.to_file(path, encoding="utf-8")

    return int(rows.size)


def write_summary(
    path: Path,
    rows: List[Dict],
) -> None:
    fieldnames = [
        "scene_folder",
        "scene_path",
        "image_prefix",
        "scene_date",
        "reflectance_mode",
        "status",
        "valid_pixels",
        "unambiguous_eq12_pixels",
        "recovered_eq13_pixels",
        "potential_raw_pixels",
        "contextual_potential_pixels",
        "final_fire_pixels",
        "fire_point_count",
        "output_dir",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
