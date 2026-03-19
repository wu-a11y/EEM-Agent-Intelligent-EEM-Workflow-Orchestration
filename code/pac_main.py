# -*- coding: utf-8 -*-
"""
PAC  process/eem 


1)  data/process/eem  EEM  numpy 
2)  PARAFAC2~6 
3)  NPZ
4)  fingerprint  PARAFAC 


-  SampleLog PAC 
-  EEMs_toolkit “CC ”
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


PAC_EEM_INPUT_DIR = "../data/process/eem"
PAC_PROCESS_OUT_ROOT = "../data/process"
PAC_PROCESS_PIC_OUT_ROOT = "../data/process/picture"
PAC_OUTLIER_PIC_OUT_DIR = "../picture/outlier_samples"
PAC_RESULT_OUT_ROOT = "../data/result"
PAC_RESULT_PIC_OUT_ROOT = "../picture/fingerprints"

# PARAFAC 
PARAFAC_STARTS = 20
PARAFAC_SELECT_STRATEGY = "cc_aware"  # : "cc_aware" / "lowest_error"
PARAFAC_CC_SSE_RELAX = 0.03

# Preprocess switches for improving model trilinearity (and CC)
ENABLE_SCATTER_MASK = True
RAYLEIGH_1ST = [22, 22]
RAYLEIGH_2ND = [40, 40]
RAMAN_1ST = [20, 20]
RAMAN_2ND = [20, 20]
RAMAN_SHIFT = 3382
ENABLE_SMOOTH = False
SMOOTH_SIGMA = 0.6
ENABLE_SAMPLE_L2_NORM = False

#  __main__  [3,4,5]
PARAFAC_COMPONENTS = [2, 3, 4, 5, 6]


def _setup_matplotlib_no_show():
    """"""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show = lambda *args, **kwargs: None
    except Exception:
        pass


def _script_dir_as_cwd():
    """"""
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)


def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _print_path(label: str, path: str):
    """"""
    print(f"[] {label}: {os.path.abspath(path)}")


def _resolve_components():
    """ ([min,max], )"""
    comps = sorted({int(c) for c in PARAFAC_COMPONENTS})
    if not comps:
        raise ValueError("PARAFAC_COMPONENTS ")
    if comps[0] < 2:
        raise ValueError(f"PARAFAC_COMPONENTS  >= 2={comps[0]}")
    return [comps[0], comps[-1]], comps


def _read_eems_from_folder(eem_folder: str):
    """
     .xlsx EEM 

     Excel 
    -  0  1  Ex 
    -  0  1  Em 
    - =Em=Ex
    """
    files = [f for f in os.listdir(eem_folder) if f.lower().endswith(".xlsx") and not f.startswith("~$")]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .xlsx files found under folder: {eem_folder}")

    x_list: list[np.ndarray] = []
    file_stems: list[str] = []
    ex = None
    em = None
    expected_shape = None

    for idx, fn in enumerate(files, start=1):
        fp = f"{eem_folder}/{fn}"
        arr = pd.read_excel(fp, header=None).values
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            print(f"[] EEM : {fp}")
            continue

        if ex is None:
            ex = arr[0, 1:]
            em = arr[1:, 0]
            expected_shape = (len(em), len(ex))

        data = arr[1:, 1:]
        if expected_shape is not None and data.shape != expected_shape:
            print(f"[] : {fp} ={data.shape}, ={expected_shape}")
            continue

        x_list.append(data)
        file_stems.append(os.path.splitext(fn)[0])

        if idx % 200 == 0 or idx == len(files):
            print(f": {idx}/{len(files)} -> {fn}")

    if not x_list:
        raise RuntimeError(f"No valid EEM files were parsed under folder: {eem_folder}")

    x = np.array(x_list)
    return (x, ex, em), file_stems


def _save_eem_dataset_to_npz(data, save_path: str):
    """ EEMs_Dataset  NPZ"""
    data_dict = {
        "x": data.x,
        "ex": data.ex,
        "em": data.em,
        "file_list": np.array(list(getattr(data, "file_list", [])), dtype=object),
        "leverage": data.leverage,
        "sse": data.sse,
        "core_consistency": data.core_consistency,
        "factors": data.factors,
        "explanation_rate": data.explanation_rate,
        "model_selection": getattr(data, "model_selection", {}),
    }
    np.savez(save_path, **data_dict)


def _load_eem_dataset_from_npz(load_path: str):
    from EEMs_toolkit import EEMs_Dataset

    loaded = np.load(load_path, allow_pickle=True)
    x = loaded["x"]
    ex = loaded["ex"]
    em = loaded["em"]
    file_list = list(loaded["file_list"]) if "file_list" in loaded else None

    kwargs = {}
    if file_list:
        kwargs["file_list"] = file_list
    data = EEMs_Dataset(x, ex, em, **kwargs)

    data.leverage = loaded["leverage"].item()
    data.sse = loaded["sse"].item()
    data.core_consistency = loaded["core_consistency"].item()
    data.explanation_rate = loaded["explanation_rate"].item()
    data.factors = loaded["factors"].item()
    if "model_selection" in loaded:
        data.model_selection = loaded["model_selection"].item()
    return data


def _preprocess_dataset_for_parafac(data):
    """
    Apply optional preprocess steps before PARAFAC fitting.
    """
    if ENABLE_SCATTER_MASK:
        data.cut_ray_scatter(RAYLEIGH_1ST, RAYLEIGH_2ND)
        data.cut_ram_scatter(RAMAN_1ST, RAMAN_2ND, freq=RAMAN_SHIFT)

    if np.isnan(data.x).any():
        data.miss_value_interpolation()

    if ENABLE_SMOOTH:
        data.smooth_eem(sigma=SMOOTH_SIGMA)

    if ENABLE_SAMPLE_L2_NORM:
        flat = data.x.reshape(data.nSample, -1)
        norms = np.linalg.norm(flat, axis=1)
        norms[norms == 0] = 1.0
        data.x = np.einsum("ijk,i->ijk", data.x, 1.0 / norms)

    return data


def step0_load_and_save(eem_folder: str):
    """0 EEM  process """
    out_data_dir = PAC_PROCESS_OUT_ROOT
    _ensure_dir(out_data_dir)

    (x, ex, em), fl = _read_eems_from_folder(eem_folder)
    np.save(f"{out_data_dir}/0_eem_data_array.npy", x)
    np.save(f"{out_data_dir}/0_ex_wavelengths.npy", ex)
    np.save(f"{out_data_dir}/0_em_wavelengths.npy", em)
    np.save(f"{out_data_dir}/0_file_list.npy", np.array(fl, dtype=object))
    print(f"[PAC] 0: {out_data_dir}")
    _print_path("0-EEM", f"{out_data_dir}/0_eem_data_array.npy")
    _print_path("0-Ex", f"{out_data_dir}/0_ex_wavelengths.npy")
    _print_path("0-Em", f"{out_data_dir}/0_em_wavelengths.npy")
    _print_path("0-", f"{out_data_dir}/0_file_list.npy")


def step1_parafac_outlier_detection():
    """1PARAFAC +  process """
    from EEMs_toolkit import EEMs_Dataset

    data_dir = PAC_PROCESS_OUT_ROOT
    pic_dir = PAC_OUTLIER_PIC_OUT_DIR
    _ensure_dir(pic_dir)

    eem_data_array_path = f"{data_dir}/0_eem_data_array.npy"
    eem_data_array_cleaned_path = f"{data_dir}/0_eem_data_array_cleaned.npy"
    ex_wavelengths_path = f"{data_dir}/0_ex_wavelengths.npy"
    em_wavelengths_path = f"{data_dir}/0_em_wavelengths.npy"
    file_list_path = f"{data_dir}/0_file_list.npy"
    outlier_samples_save_path = f"{data_dir}/outlier_sample_indices.xlsx"

    x = np.load(eem_data_array_path)
    ex = np.load(ex_wavelengths_path)
    em = np.load(em_wavelengths_path)
    fl = np.load(file_list_path, allow_pickle=True)

    fit_range, fac_list = _resolve_components()

    # Preprocess for better trilinear structure before fitting.
    data_for_clean = EEMs_Dataset(x.copy(), ex, em, file_list=list(fl))
    data_for_clean = _preprocess_dataset_for_parafac(data_for_clean)
    np.save(eem_data_array_cleaned_path, data_for_clean.x)

    data = EEMs_Dataset(data_for_clean.x, ex, em, file_list=list(fl))
    data.multi_non_parafac_cal(
        fit_range,
        start=PARAFAC_STARTS,
        select_strategy=PARAFAC_SELECT_STRATEGY,
        cc_sse_relax=PARAFAC_CC_SSE_RELAX
    )

    outlier_records: dict[str, list[int]] = {}
    for fac in fac_list:
        if fac in data.leverage and fac in data.sse:
            save_path = f"{pic_dir}/outlier_{fac}_components.png"
            data.plot_outlier_test(fac, save_path=save_path)
            _print_path(f"1--{fac}", save_path)

            leverage = np.array(data.leverage[fac])
            sse = np.array(data.sse[fac])
            leverage_threshold = np.percentile(leverage, 95)
            sse_threshold = np.percentile(sse, 95)
            outlier_indices = np.where((leverage > leverage_threshold) | (sse > sse_threshold))[0]
            outlier_records[f"{fac}-component"] = outlier_indices.tolist()
        else:
            print(f"[PAC]  {fac}-component")

    outlier_df = pd.DataFrame({k: pd.Series(v) for k, v in outlier_records.items()})
    outlier_df.to_excel(outlier_samples_save_path, index=False)
    print(f"[PAC] : {outlier_samples_save_path}")
    _print_path("1-EEM", eem_data_array_cleaned_path)
    _print_path("1-", outlier_samples_save_path)

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


def step2_remove_outliers_and_split_analysis(residual_sample_count: int = 0):
    """2 +  +  process """
    from EEMs_toolkit import EEMs_Dataset

    data_dir = PAC_PROCESS_OUT_ROOT
    eem_data_array_path = f"{data_dir}/0_eem_data_array_cleaned.npy"
    ex_wavelengths_path = f"{data_dir}/0_ex_wavelengths.npy"
    em_wavelengths_path = f"{data_dir}/0_em_wavelengths.npy"
    file_list_path = f"{data_dir}/0_file_list.npy"
    outlier_samples_path = f"{data_dir}/outlier_sample_indices.xlsx"

    save_npz_path = f"{data_dir}/result_data.npz"
    split_summary_path = f"{data_dir}/split_half_summary.xlsx"

    x = np.load(eem_data_array_path)
    ex = np.load(ex_wavelengths_path)
    em = np.load(em_wavelengths_path)
    fl = np.array(list(np.load(file_list_path, allow_pickle=True)), dtype=object)

    fit_range, fac_list = _resolve_components()

    outlier_df = pd.read_excel(outlier_samples_path)
    outlier_indices = pd.unique(outlier_df.values.ravel())
    outlier_indices = outlier_indices[~pd.isna(outlier_indices)].astype(int)
    print(f"[PAC] : {outlier_indices.tolist()}")

    mask = np.ones(x.shape[0], dtype=bool)
    mask[outlier_indices] = False
    x_filtered = x[mask]
    fl_filtered = fl[mask].tolist()

    data = EEMs_Dataset(x_filtered, ex, em, file_list=fl_filtered)
    data = _preprocess_dataset_for_parafac(data)
    data.multi_non_parafac_cal(
        fit_range,
        start=PARAFAC_STARTS,
        select_strategy=PARAFAC_SELECT_STRATEGY,
        cc_sse_relax=PARAFAC_CC_SSE_RELAX
    )

    data.split_analysis(fit_range)
    data.save_split_analysis_summary(split_summary_path)
    data.plot_factor_similarity()
    data.plot_core_consistency_and_explanation()

    if residual_sample_count > 0:
        sample_id = list(range(1, min(residual_sample_count, data.nSample) + 1))
        for fac in fac_list:
            data.plot_residual_error(fac, sample_id=sample_id)

    _save_eem_dataset_to_npz(data, save_npz_path)
    print(f"[PAC]  NPZ: {save_npz_path}")
    _print_path("2-", split_summary_path)
    _print_path("2-NPZ", save_npz_path)

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


def step3_export_parafac_results():
    """3 fingerprint  PARAFAC  data/result"""
    data_dir = PAC_PROCESS_OUT_ROOT
    load_npz_path = f"{data_dir}/result_data.npz"

    output_dir = PAC_RESULT_OUT_ROOT
    picture_output_dir = PAC_RESULT_PIC_OUT_ROOT
    _ensure_dir(output_dir)
    _ensure_dir(picture_output_dir)

    _, fac_list = _resolve_components()

    data = _load_eem_dataset_from_npz(load_npz_path)
    for fac in fac_list:
        data.split_analysis([fac])
        pic_path = f"{picture_output_dir}/finger_{fac}.png"
        data.plot_fingers(
            fac,
            save_path=pic_path,
        )
        data.parafac_result_output(fac, output_dir)
        _print_path(f"3-fingerprint-{fac}()", pic_path)

    _print_path("3-", output_dir)
    _print_path("3-", picture_output_dir)

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


if __name__ == "__main__":
    _script_dir_as_cwd()
    _setup_matplotlib_no_show()

    # ----------  ----------
    EEM_INPUT_DIR = PAC_EEM_INPUT_DIR

    RUN_STEP0 = True
    RUN_STEP1 = True
    RUN_STEP2 = True
    RUN_STEP3 = True

    #  [3, 4, 5] list(range(2, 7))
    PARAFAC_COMPONENTS = [2]

    #  N 0 
    RESIDUAL_SAMPLE_COUNT = 0
    # --------------------------------

    print("\n===  ===")
    _print_path("EEM_INPUT_DIR", EEM_INPUT_DIR)
    _print_path("PAC_PROCESS_OUT_ROOT", PAC_PROCESS_OUT_ROOT)
    _print_path("PAC_PROCESS_PIC_OUT_ROOT", PAC_PROCESS_PIC_OUT_ROOT)
    _print_path("PAC_OUTLIER_PIC_OUT_DIR", PAC_OUTLIER_PIC_OUT_DIR)
    _print_path("PAC_RESULT_OUT_ROOT", PAC_RESULT_OUT_ROOT)
    _print_path("PAC_RESULT_PIC_OUT_ROOT", PAC_RESULT_PIC_OUT_ROOT)
    fit_range, fac_list = _resolve_components()
    print(f": {fac_list}")
    print(f": {fit_range}")

    if not os.path.isdir(EEM_INPUT_DIR):
        raise FileNotFoundError(f" EEM : {EEM_INPUT_DIR}")

    print("\n===== DATASET: PAC =====")
    if RUN_STEP0:
        print("\n=== STEP 0 ===")
        step0_load_and_save(EEM_INPUT_DIR)
    if RUN_STEP1:
        print("\n=== STEP 1 ===")
        step1_parafac_outlier_detection()
    if RUN_STEP2:
        print("\n=== STEP 2 ===")
        step2_remove_outliers_and_split_analysis(residual_sample_count=RESIDUAL_SAMPLE_COUNT)
    if RUN_STEP3:
        print("\n=== STEP 3 ===")
        step3_export_parafac_results()
