from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (
    ROOT_DIR,
    OUTPUT_DIR,
    SUMMARY_CSV,
    SAVE_REFLECTANCE,
    SAVE_FIRE_TYPE,
    SAVE_FIRE_POINTS,
    SAVE_DEBUG_RASTERS,
)

from landsat_io import (
    detect_scene_prefix,
    discover_scene_dirs,
    get_scene_files,
    make_qa_mask,
    parse_mtl,
    read_band,
    scale_reflectance,
)

from goli_detection import detect_goli_fire

from outputs import (
    write_fire_points,
    write_raster,
    write_summary,
)


# =============================================================================
# 1. 日期提取
# =============================================================================
def extract_scene_date(prefix: str) -> str:
    parts = prefix.split("_")

    if len(parts) >= 4:
        value = parts[3]

        if len(value) == 8 and value.isdigit():
            return (
                f"{value[:4]}-"
                f"{value[4:6]}-"
                f"{value[6:8]}"
            )

        return value

    return ""


# =============================================================================
# 2. 单景处理
# =============================================================================
def process_scene(scene_dir: Path) -> Dict:
    prefix, mode = detect_scene_prefix(scene_dir)
    scene_date = extract_scene_date(prefix)
    scene_name = scene_dir.name

    files = get_scene_files(scene_dir, prefix, mode)

    print("\n" + "=" * 90)
    print(f"Scene       : {scene_dir}")
    print(f"Prefix      : {prefix}")
    print(f"Date        : {scene_date}")
    print(f"Mode        : {mode}")
    print("=" * 90)

    arrays_dn: Dict[str, np.ndarray] = {}
    nodata_values: Dict[str, Optional[float]] = {}

    profile = None
    shape = None

    for band_name in [
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
    ]:
        path = files[band_name]

        if path is None:
            raise FileNotFoundError(f"缺少波段：{band_name}")

        array, current_profile, nodata = read_band(path)

        if profile is None:
            profile = current_profile
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"波段尺寸不一致：{band_name}={array.shape}, "
                f"reference={shape}"
            )

        arrays_dn[band_name] = array
        nodata_values[band_name] = nodata

    if profile is None or shape is None:
        raise RuntimeError("未成功读取反射率波段。")

    mtl_values = None

    if mode == "L1_TOA":
        if files["mtl"] is None:
            raise FileNotFoundError(
                "Level-1 TOA 输入缺少 *_MTL.txt。"
            )

        mtl_values = parse_mtl(files["mtl"])

    band_numbers = {
        "blue": 2,
        "green": 3,
        "red": 4,
        "nir": 5,
        "swir1": 6,
        "swir2": 7,
    }

    reflectance: Dict[str, np.ndarray] = {}
    valid_masks: Dict[str, np.ndarray] = {}

    for band_name, band_number in band_numbers.items():
        refl, valid = scale_reflectance(
            arrays_dn[band_name],
            nodata_values[band_name],
            band_number,
            mode,
            mtl_values,
        )

        reflectance[band_name] = refl
        valid_masks[band_name] = valid

    valid = np.ones(shape, dtype=bool)

    for band_name in band_numbers:
        valid &= valid_masks[band_name]

    if files["qa"] is not None:
        qa, _, _ = read_band(files["qa"])

        if qa.shape != shape:
            raise ValueError(
                f"QA 尺寸不一致：qa={qa.shape}, reference={shape}"
            )

        valid &= make_qa_mask(qa)

    fire_mask, fire_type, debug = detect_goli_fire(
        blue=reflectance["blue"],
        green=reflectance["green"],
        red=reflectance["red"],
        nir=reflectance["nir"],
        swir1=reflectance["swir1"],
        swir2=reflectance["swir2"],
        valid=valid,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_fire = OUTPUT_DIR / f"{scene_name}_GOLI_fire_mask.tif"
    out_type = OUTPUT_DIR / f"{scene_name}_GOLI_fire_type.tif"
    out_points = OUTPUT_DIR / f"{scene_name}_GOLI_fire_pixels.shp"

    write_raster(
        out_fire,
        fire_mask,
        profile,
        dtype="uint8",
        nodata=0,
    )

    if SAVE_FIRE_TYPE:
        write_raster(
            out_type,
            fire_type,
            profile,
            dtype="uint8",
            nodata=0,
        )

    if SAVE_REFLECTANCE:
        for band_name in [
            "blue",
            "green",
            "red",
            "nir",
            "swir1",
            "swir2",
        ]:
            write_raster(
                OUTPUT_DIR
                / f"{scene_name}_{band_name}_reflectance.tif",
                reflectance[band_name],
                profile,
                dtype="float32",
                nodata=-9999.0,
            )

    if SAVE_DEBUG_RASTERS:
        uint8_debug = [
            "unambiguous_eq12",
            "recovered_eq13",
            "potential_eq14",
            "potential_eq15",
            "potential_raw",
            "contextual_potential",
            "water_background",
            "background_valid",
            "selected_window",
        ]

        for name in uint8_debug:
            write_raster(
                OUTPUT_DIR / f"{scene_name}_{name}.tif",
                debug[name],
                profile,
                dtype="uint8",
                nodata=0,
            )

        float_debug = [
            "swir2_nir_ratio",
            "context_ratio_mean",
            "context_ratio_std",
            "context_swir2_mean",
            "context_swir2_std",
        ]

        for name in float_debug:
            write_raster(
                OUTPUT_DIR / f"{scene_name}_{name}.tif",
                debug[name],
                profile,
                dtype="float32",
                nodata=-9999.0,
            )

    if SAVE_FIRE_POINTS:
        point_count = write_fire_points(
            path=out_points,
            fire_mask=fire_mask,
            fire_type=fire_type,
            red=reflectance["red"],
            nir=reflectance["nir"],
            swir1=reflectance["swir1"],
            swir2=reflectance["swir2"],
            ratio75=debug["swir2_nir_ratio"],
            selected_window=debug["selected_window"],
            scene_date=scene_date,
            transform=profile["transform"],
            crs=profile.get("crs"),
        )
    else:
        point_count = 0

    stats = {
        "scene_folder": scene_name,
        "scene_path": str(scene_dir),
        "image_prefix": prefix,
        "scene_date": scene_date,
        "reflectance_mode": mode,
        "status": "success",
        "valid_pixels": int(valid.sum()),
        "unambiguous_eq12_pixels": int(
            debug["unambiguous_eq12"].sum()
        ),
        "recovered_eq13_pixels": int(
            debug["recovered_eq13"].sum()
        ),
        "potential_raw_pixels": int(
            debug["potential_raw"].sum()
        ),
        "contextual_potential_pixels": int(
            debug["contextual_potential"].sum()
        ),
        "final_fire_pixels": int(fire_mask.sum()),
        "fire_point_count": point_count,
        "output_dir": str(OUTPUT_DIR),
        "error": "",
    }

    print("\nStatistics:")
    print(
        f"Unambiguous Eq.12    : "
        f"{stats['unambiguous_eq12_pixels']}"
    )
    print(
        f"Recovered Eq.13      : "
        f"{stats['recovered_eq13_pixels']}"
    )
    print(
        f"Raw potential        : "
        f"{stats['potential_raw_pixels']}"
    )
    print(
        f"Contextual potential : "
        f"{stats['contextual_potential_pixels']}"
    )
    print(
        f"Final fire pixels    : "
        f"{stats['final_fire_pixels']}"
    )

    return stats


# =============================================================================
# 3. 批处理入口
# =============================================================================
def main() -> None:
    print("=" * 90)
    print("GOLI Landsat-8 reflectance-based active fire detection")
    print("Single-scene core implementation: equations 8, 9 and 12-16")
    print("=" * 90)
    print(f"ROOT_DIR  : {ROOT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    scenes = discover_scene_dirs(ROOT_DIR)

    if not scenes:
        raise FileNotFoundError(
            "未找到 Landsat 场景文件夹。"
        )

    print(f"Found {len(scenes)} scene(s).")

    summary_rows: List[Dict] = []

    for index, scene in enumerate(scenes, start=1):
        print("\n" + "#" * 90)
        print(f"Processing {index}/{len(scenes)}: {scene.name}")
        print("#" * 90)

        try:
            summary_rows.append(process_scene(scene))

        except Exception as exc:
            print(
                f"处理失败：{type(exc).__name__}: {exc}"
            )

            summary_rows.append(
                {
                    "scene_folder": scene.name,
                    "scene_path": str(scene),
                    "image_prefix": "",
                    "scene_date": "",
                    "reflectance_mode": "",
                    "status": "failed",
                    "valid_pixels": "",
                    "unambiguous_eq12_pixels": "",
                    "recovered_eq13_pixels": "",
                    "potential_raw_pixels": "",
                    "contextual_potential_pixels": "",
                    "final_fire_pixels": "",
                    "fire_point_count": "",
                    "output_dir": str(OUTPUT_DIR),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        finally:
            gc.collect()

    write_summary(SUMMARY_CSV, summary_rows)

    success = sum(
        row["status"] == "success"
        for row in summary_rows
    )

    print("\n" + "=" * 90)
    print("Batch finished")
    print("=" * 90)
    print(f"Total  : {len(summary_rows)}")
    print(f"Success: {success}")
    print(f"Failed : {len(summary_rows) - success}")
    print(f"Summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
