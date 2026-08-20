from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from config import (
    OUTPUT_DIR,
    SEARCH_RECURSIVELY,
    REFLECTANCE_MODE,
    ALLOW_LEVEL2_SR_ADAPTATION,
    USE_QA_CLOUD_MASK,
    MASK_QA_WATER,
    SR_SCALE,
    SR_OFFSET,
)


# =============================================================================
# 1. 场景查找
# =============================================================================
def contains_landsat_scene(folder: Path) -> bool:
    if not folder.is_dir():
        return False

    names = [p.name.upper() for p in folder.iterdir() if p.is_file()]

    return any(
        name.endswith("_SR_B2.TIF") or name.endswith("_B2.TIF")
        for name in names
    )


def discover_scene_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在：{root_dir}")

    if SEARCH_RECURSIVELY:
        candidates = [p for p in root_dir.rglob("*") if p.is_dir()]
    else:
        candidates = [p for p in root_dir.iterdir() if p.is_dir()]

    result = []

    for folder in candidates:
        try:
            if folder.resolve() == OUTPUT_DIR.resolve():
                continue
        except OSError:
            pass

        if contains_landsat_scene(folder):
            result.append(folder)

    return sorted(result)


def detect_scene_prefix(input_dir: Path) -> Tuple[str, str]:
    """
    返回：
        prefix
        detected_mode: "L1_TOA" 或 "L2_SR"
    """
    files = [p for p in input_dir.iterdir() if p.is_file()]

    sr_matches = [
        p for p in files
        if p.name.upper().endswith("_SR_B2.TIF")
    ]

    l1_matches = [
        p for p in files
        if p.name.upper().endswith("_B2.TIF")
        and not p.name.upper().endswith("_SR_B2.TIF")
    ]

    requested = REFLECTANCE_MODE.upper()

    if requested == "AUTO":
        if len(sr_matches) == 1:
            suffix = "_SR_B2.TIF"
            return sr_matches[0].name[:-len(suffix)], "L2_SR"

        if len(l1_matches) == 1:
            suffix = "_B2.TIF"
            return l1_matches[0].name[:-len(suffix)], "L1_TOA"

    elif requested == "L2_SR":
        if len(sr_matches) == 1:
            suffix = "_SR_B2.TIF"
            return sr_matches[0].name[:-len(suffix)], "L2_SR"

    elif requested == "L1_TOA":
        if len(l1_matches) == 1:
            suffix = "_B2.TIF"
            return l1_matches[0].name[:-len(suffix)], "L1_TOA"

    else:
        raise ValueError(
            "REFLECTANCE_MODE 只能是 AUTO、L1_TOA 或 L2_SR。"
        )

    raise FileNotFoundError(
        f"无法唯一识别 Landsat 场景及输入模式：{input_dir}"
    )


# =============================================================================
# 2. 文件查找
# =============================================================================
def find_file(
    input_dir: Path,
    prefix: str,
    suffixes: List[str],
    required: bool = True,
) -> Optional[Path]:
    file_map = {
        p.name.lower(): p
        for p in input_dir.iterdir()
        if p.is_file()
    }

    for suffix in suffixes:
        name = f"{prefix}{suffix}".lower()

        if name in file_map:
            return file_map[name]

    if required:
        raise FileNotFoundError(
            f"找不到文件：prefix={prefix}, suffixes={suffixes}"
        )

    return None


def get_scene_files(
    input_dir: Path,
    prefix: str,
    mode: str,
) -> Dict[str, Optional[Path]]:
    if mode == "L2_SR":
        suffix = {
            "blue": "_SR_B2.TIF",
            "green": "_SR_B3.TIF",
            "red": "_SR_B4.TIF",
            "nir": "_SR_B5.TIF",
            "swir1": "_SR_B6.TIF",
            "swir2": "_SR_B7.TIF",
        }
    else:
        suffix = {
            "blue": "_B2.TIF",
            "green": "_B3.TIF",
            "red": "_B4.TIF",
            "nir": "_B5.TIF",
            "swir1": "_B6.TIF",
            "swir2": "_B7.TIF",
        }

    files: Dict[str, Optional[Path]] = {
        key: find_file(input_dir, prefix, [value])
        for key, value in suffix.items()
    }

    files["qa"] = find_file(
        input_dir,
        prefix,
        ["_QA_PIXEL.TIF", "_BQA.TIF"],
        required=False,
    )

    files["mtl"] = find_file(
        input_dir,
        prefix,
        ["_MTL.txt", "_MTL.TXT"],
        required=False,
    )

    return files


# =============================================================================
# 3. 栅格读取
# =============================================================================
def read_band(path: Path) -> Tuple[np.ndarray, dict, Optional[float]]:
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    return array, profile, nodata


# =============================================================================
# 4. MTL 解析和反射率计算
# =============================================================================
def parse_mtl(mtl_path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')

            try:
                values[key] = float(value)
            except ValueError:
                continue

    return values


def level1_toa_reflectance(
    dn: np.ndarray,
    nodata: Optional[float],
    band_number: int,
    mtl_values: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    mult_key = f"REFLECTANCE_MULT_BAND_{band_number}"
    add_key = f"REFLECTANCE_ADD_BAND_{band_number}"

    required = [mult_key, add_key, "SUN_ELEVATION"]

    missing = [key for key in required if key not in mtl_values]

    if missing:
        raise KeyError(
            "MTL 缺少 Level-1 TOA 反射率参数："
            + ", ".join(missing)
        )

    arr = dn.astype(np.float32)
    valid = np.isfinite(arr)

    if nodata is not None:
        valid &= arr != nodata

    valid &= arr > 0

    sun_elevation_rad = np.deg2rad(
        float(mtl_values["SUN_ELEVATION"])
    )

    sin_sun = np.sin(sun_elevation_rad)

    if sin_sun <= 0:
        raise ValueError("SUN_ELEVATION 无效。")

    reflectance = (
        arr * float(mtl_values[mult_key])
        + float(mtl_values[add_key])
    ) / sin_sun

    reflectance[~valid] = np.nan

    return reflectance.astype(np.float32), valid


def level2_surface_reflectance(
    dn: np.ndarray,
    nodata: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if not ALLOW_LEVEL2_SR_ADAPTATION:
        raise ValueError(
            "当前输入是 Level-2 SR，但配置禁止将论文 TOA 阈值用于 SR。"
        )

    arr = dn.astype(np.float32)
    valid = np.isfinite(arr)

    if nodata is not None:
        valid &= arr != nodata

    valid &= arr > 0

    reflectance = arr * SR_SCALE + SR_OFFSET
    reflectance[~valid] = np.nan

    return reflectance.astype(np.float32), valid


def scale_reflectance(
    dn: np.ndarray,
    nodata: Optional[float],
    band_number: int,
    mode: str,
    mtl_values: Optional[Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "L1_TOA":
        if mtl_values is None:
            raise FileNotFoundError(
                "Level-1 TOA 模式需要 MTL 文件。"
            )

        return level1_toa_reflectance(
            dn,
            nodata,
            band_number,
            mtl_values,
        )

    return level2_surface_reflectance(dn, nodata)


# =============================================================================
# 5. QA_PIXEL 掩膜
# =============================================================================
def make_qa_mask(qa: np.ndarray) -> np.ndarray:
    """
    适用于 Landsat Collection 2 QA_PIXEL。
    """
    qa = qa.astype(np.uint16)

    fill = (qa & (1 << 0)) != 0
    dilated_cloud = (qa & (1 << 1)) != 0
    cirrus = (qa & (1 << 2)) != 0
    cloud = (qa & (1 << 3)) != 0
    shadow = (qa & (1 << 4)) != 0
    snow = (qa & (1 << 5)) != 0
    water = (qa & (1 << 7)) != 0

    mask = ~fill

    if USE_QA_CLOUD_MASK:
        mask &= ~(dilated_cloud | cirrus | cloud | shadow | snow)

    if MASK_QA_WATER:
        mask &= ~water

    return mask
