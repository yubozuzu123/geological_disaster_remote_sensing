from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from config import (
    UNAMBIGUOUS_RED_SWIR2_SLOPE,
    UNAMBIGUOUS_RED_SWIR2_INTERCEPT,
    RECOVERY_RED_SWIR1_SLOPE,
    RECOVERY_RED_SWIR1_INTERCEPT,
    POTENTIAL_RED_SWIR2_SLOPE,
    POTENTIAL_RED_SWIR2_INTERCEPT,
    POTENTIAL_SWIR1_SWIR2_SLOPE,
    POTENTIAL_SWIR1_SWIR2_INTERCEPT,
    CONTEXT_MIN_WINDOW,
    CONTEXT_MAX_WINDOW,
    CONTEXT_WINDOW_STEP,
    CONTEXT_MIN_VALID_FRACTION,
    CONTEXT_RATIO_STD_FACTOR,
    CONTEXT_RATIO_MIN_INCREMENT,
    CONTEXT_SWIR2_STD_FACTOR,
    CONTEXT_SWIR2_MIN_INCREMENT,
    MIN_COMPONENT_PIXELS,
    MAX_COMPONENT_PIXELS,
)


# =============================================================================
# 1. 基础函数
# =============================================================================
def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=np.float32)

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-6)
    )

    out[valid] = numerator[valid] / denominator[valid]

    return out


def integral_image(array: np.ndarray) -> np.ndarray:
    """
    构建带一行一列零边界的二维积分图。
    """
    result = np.pad(
        array.astype(np.float64),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )

    result = np.cumsum(np.cumsum(result, axis=0), axis=1)

    return result


def rectangle_sum(
    integral: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    half_size: int,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r0 = np.maximum(rows - half_size, 0)
    r1 = np.minimum(rows + half_size + 1, height)

    c0 = np.maximum(cols - half_size, 0)
    c1 = np.minimum(cols + half_size + 1, width)

    values = (
        integral[r1, c1]
        - integral[r0, c1]
        - integral[r1, c0]
        + integral[r0, c0]
    )

    area = (r1 - r0) * (c1 - c0)

    return values, area


# =============================================================================
# 2. GOLI 自适应上下文检验
# =============================================================================
def contextual_test(
    potential_mask: np.ndarray,
    background_valid: np.ndarray,
    swir2: np.ndarray,
    swir2_nir_ratio: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """

    窗口从 5×5 增长到 61×61，选择第一个具有至少 25%
    可用周围背景像元的窗口。
    """
    height, width = swir2.shape

    candidate_rows, candidate_cols = np.where(potential_mask)

    accepted = np.zeros_like(potential_mask, dtype=np.uint8)

    selected_window = np.zeros_like(swir2, dtype=np.uint8)

    ratio_mean_out = np.full(
        swir2.shape,
        np.nan,
        dtype=np.float32,
    )
    ratio_std_out = np.full(
        swir2.shape,
        np.nan,
        dtype=np.float32,
    )
    swir2_mean_out = np.full(
        swir2.shape,
        np.nan,
        dtype=np.float32,
    )
    swir2_std_out = np.full(
        swir2.shape,
        np.nan,
        dtype=np.float32,
    )

    if candidate_rows.size == 0:
        return accepted, {
            "selected_window": selected_window,
            "context_ratio_mean": ratio_mean_out,
            "context_ratio_std": ratio_std_out,
            "context_swir2_mean": swir2_mean_out,
            "context_swir2_std": swir2_std_out,
        }

    bg_valid = (
        background_valid
        & np.isfinite(swir2)
        & np.isfinite(swir2_nir_ratio)
    )

    weight = bg_valid.astype(np.float64)

    ratio_filled = np.where(
        bg_valid,
        swir2_nir_ratio,
        0.0,
    ).astype(np.float64)

    swir2_filled = np.where(
        bg_valid,
        swir2,
        0.0,
    ).astype(np.float64)

    ii_count = integral_image(weight)

    ii_ratio = integral_image(ratio_filled)
    ii_ratio2 = integral_image(ratio_filled * ratio_filled)

    ii_swir2 = integral_image(swir2_filled)
    ii_swir22 = integral_image(swir2_filled * swir2_filled)

    unresolved = np.ones(candidate_rows.size, dtype=bool)

    for window_size in range(
        CONTEXT_MIN_WINDOW,
        CONTEXT_MAX_WINDOW + 1,
        CONTEXT_WINDOW_STEP,
    ):
        if not np.any(unresolved):
            break

        idx = np.where(unresolved)[0]
        rows = candidate_rows[idx]
        cols = candidate_cols[idx]

        half_size = window_size // 2

        count, clipped_area = rectangle_sum(
            ii_count,
            rows,
            cols,
            half_size,
            height,
            width,
        )

        surrounding_area = np.maximum(clipped_area - 1, 1)

        minimum_count = np.ceil(
            CONTEXT_MIN_VALID_FRACTION * surrounding_area
        )

        sufficient = count >= minimum_count

        if not np.any(sufficient):
            continue

        selected_idx = idx[sufficient]
        selected_rows = candidate_rows[selected_idx]
        selected_cols = candidate_cols[selected_idx]

        count_sel = count[sufficient]

        ratio_sum, _ = rectangle_sum(
            ii_ratio,
            selected_rows,
            selected_cols,
            half_size,
            height,
            width,
        )
        ratio_sum2, _ = rectangle_sum(
            ii_ratio2,
            selected_rows,
            selected_cols,
            half_size,
            height,
            width,
        )

        swir2_sum, _ = rectangle_sum(
            ii_swir2,
            selected_rows,
            selected_cols,
            half_size,
            height,
            width,
        )
        swir2_sum2, _ = rectangle_sum(
            ii_swir22,
            selected_rows,
            selected_cols,
            half_size,
            height,
            width,
        )

        ratio_mean = ratio_sum / np.maximum(count_sel, 1.0)
        ratio_var = np.maximum(
            ratio_sum2 / np.maximum(count_sel, 1.0)
            - ratio_mean * ratio_mean,
            0.0,
        )
        ratio_std = np.sqrt(ratio_var)

        swir2_mean = swir2_sum / np.maximum(count_sel, 1.0)
        swir2_var = np.maximum(
            swir2_sum2 / np.maximum(count_sel, 1.0)
            - swir2_mean * swir2_mean,
            0.0,
        )
        swir2_std = np.sqrt(swir2_var)

        ratio_threshold = (
            ratio_mean
            + np.maximum(
                CONTEXT_RATIO_STD_FACTOR * ratio_std,
                CONTEXT_RATIO_MIN_INCREMENT,
            )
        )

        swir2_threshold = (
            swir2_mean
            + np.maximum(
                CONTEXT_SWIR2_STD_FACTOR * swir2_std,
                CONTEXT_SWIR2_MIN_INCREMENT,
            )
        )

        current_ratio = swir2_nir_ratio[
            selected_rows,
            selected_cols,
        ]

        current_swir2 = swir2[
            selected_rows,
            selected_cols,
        ]

        passed = (
            current_ratio > ratio_threshold
        ) & (
            current_swir2 > swir2_threshold
        )

        accepted[
            selected_rows[passed],
            selected_cols[passed],
        ] = 1

        selected_window[
            selected_rows,
            selected_cols,
        ] = window_size

        ratio_mean_out[
            selected_rows,
            selected_cols,
        ] = ratio_mean.astype(np.float32)

        ratio_std_out[
            selected_rows,
            selected_cols,
        ] = ratio_std.astype(np.float32)

        swir2_mean_out[
            selected_rows,
            selected_cols,
        ] = swir2_mean.astype(np.float32)

        swir2_std_out[
            selected_rows,
            selected_cols,
        ] = swir2_std.astype(np.float32)

        unresolved[selected_idx] = False

    debug = {
        "selected_window": selected_window,
        "context_ratio_mean": ratio_mean_out,
        "context_ratio_std": ratio_std_out,
        "context_swir2_mean": swir2_mean_out,
        "context_swir2_std": swir2_std_out,
    }

    return accepted, debug


# =============================================================================
# 3. 可选斑块大小清理
# =============================================================================
def filter_components(mask: np.ndarray) -> np.ndarray:
    if MIN_COMPONENT_PIXELS <= 1 and MAX_COMPONENT_PIXELS is None:
        return mask.astype(np.uint8)

    structure = np.ones((3, 3), dtype=np.uint8)

    labels, number = ndimage.label(mask, structure=structure)

    output = np.zeros_like(mask, dtype=np.uint8)

    if number == 0:
        return output

    counts = np.bincount(labels.ravel())

    for component_id in range(1, number + 1):
        count = int(counts[component_id])

        if count < MIN_COMPONENT_PIXELS:
            continue

        if (
            MAX_COMPONENT_PIXELS is not None
            and count > MAX_COMPONENT_PIXELS
        ):
            continue

        output[labels == component_id] = 1

    return output


# =============================================================================
# 4. GOLI 主检测
# =============================================================================
def detect_goli_fire(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    实现论文 GOLI 单景核心算法：
        公式（12）-（16）
        公式（8）和（9）的上下文检验

    fire_type:
        0 = 非火点
        1 = 上下文检验通过的潜在火点
        2 = 公式（12）明确火点
        3 = 公式（13）相邻饱和/强火恢复像元
    """
    finite = (
        valid
        & np.isfinite(blue)
        & np.isfinite(green)
        & np.isfinite(red)
        & np.isfinite(nir)
        & np.isfinite(swir1)
        & np.isfinite(swir2)
    )

    # -------------------------------------------------------------------------
    # 论文公式（12）：明确火点
    # -------------------------------------------------------------------------
    unambiguous_eq12 = (
        finite
        & (
            red
            <= (
                UNAMBIGUOUS_RED_SWIR2_SLOPE * swir2
                + UNAMBIGUOUS_RED_SWIR2_INTERCEPT
            )
        )
    )

    # -------------------------------------------------------------------------
    # 论文公式（13）：与公式（12）火点相邻的强火恢复
    # -------------------------------------------------------------------------
    structure = np.ones((3, 3), dtype=np.uint8)
    adjacent_to_eq12 = ndimage.binary_dilation(
        unambiguous_eq12,
        structure=structure,
        iterations=1,
    )
    recovery_condition = (
        finite
        & (
            red
            <= (
                RECOVERY_RED_SWIR1_SLOPE * swir1
                + RECOVERY_RED_SWIR1_INTERCEPT
            )
        )
    )
    recovered_eq13 = (
        adjacent_to_eq12
        & recovery_condition
        & ~unambiguous_eq12
    )
    unambiguous_all = unambiguous_eq12 | recovered_eq13

    # -------------------------------------------------------------------------
    # 论文公式（14）和（15）：潜在火点初选
    # -------------------------------------------------------------------------
    potential_eq14 = (
        finite
        & (
            red
            <= (
                POTENTIAL_RED_SWIR2_SLOPE * swir2
                + POTENTIAL_RED_SWIR2_INTERCEPT
            )
        )
    )
    potential_eq15 = (
        finite
        & (
            swir1
            <= (
                POTENTIAL_SWIR1_SWIR2_SLOPE * swir2
                + POTENTIAL_SWIR1_SWIR2_INTERCEPT
            )
        )
    )
    potential_raw = (
        (potential_eq14 | potential_eq15)
        & ~unambiguous_all
    )

    # -------------------------------------------------------------------------
    # 论文公式（16）：上下文背景水体识别
    # Blue > Green > Red > NIR
    # -------------------------------------------------------------------------
    water_background = (
        finite
        & (blue > green)
        & (green > red)
        & (red > nir)
    )

    # 论文要求上下文背景排除：
    # 明确火点、潜在火点和水体。
    background_valid = (
        finite
        & ~unambiguous_all
        & ~potential_raw
        & ~water_background
    )

    swir2_nir_ratio = safe_ratio(swir2, nir)

    contextual_potential, context_debug = contextual_test(
        potential_mask=potential_raw,
        background_valid=background_valid,
        swir2=swir2,
        swir2_nir_ratio=swir2_nir_ratio,
    )

    fire_type = np.zeros(swir2.shape, dtype=np.uint8)

    fire_type[contextual_potential == 1] = 1
    fire_type[unambiguous_eq12] = 2
    fire_type[recovered_eq13] = 3

    fire_mask = (fire_type > 0).astype(np.uint8)

    fire_mask = filter_components(fire_mask)

    fire_type = np.where(
        fire_mask == 1,
        fire_type,
        0,
    ).astype(np.uint8)

    debug = {
        "unambiguous_eq12": unambiguous_eq12.astype(np.uint8),
        "recovered_eq13": recovered_eq13.astype(np.uint8),
        "potential_eq14": potential_eq14.astype(np.uint8),
        "potential_eq15": potential_eq15.astype(np.uint8),
        "potential_raw": potential_raw.astype(np.uint8),
        "contextual_potential": contextual_potential.astype(np.uint8),
        "water_background": water_background.astype(np.uint8),
        "background_valid": background_valid.astype(np.uint8),
        "swir2_nir_ratio": swir2_nir_ratio,
        **context_debug,
    }

    return fire_mask, fire_type, debug
