import pandas as pd
import numpy as np
import os
import re
from scipy.interpolate import interp1d


# Step 1. Define Tucker Congruence calculation
def calculate_tucker_q(x1, x2):
    numerator = np.sum(x1 * x2)
    denominator = np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2))
    if denominator == 0:
        return 0
    return numerator / denominator


# Step 2. Compare similarity after interpolation
def calculate_similarity_overlap(t_x, t_col, s_x, s_col):
    overlap_min = max(np.min(t_x), np.min(s_x))
    overlap_max = min(np.max(t_x), np.max(s_x))
    if overlap_min >= overlap_max:
        return 0

    overlap_x = np.arange(overlap_min, overlap_max + 1, 1)
    f_target = interp1d(t_x, t_col, kind='linear', bounds_error=False, fill_value="extrapolate")
    f_standard = interp1d(s_x, s_col, kind='linear', bounds_error=False, fill_value="extrapolate")
    t_col_interp = f_target(overlap_x)
    s_col_interp = f_standard(overlap_x)

    q = calculate_tucker_q(t_col_interp, s_col_interp)
    return q


# =================== Path Configuration ===================
# PAC root directory
pac_root = "../data/result"

# PAC subfolders to process. If empty, scan all subfolders under pac_root.
selected_samples = ["."]

# Standard library sample folder
standard_folder = r"..\docs\openflour_knowledge_base\openflour_database"


def _has_component_results(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    return any(re.match(r"^(\d+)-component result\.xlsx$", f) for f in os.listdir(folder))


def _resolve_result_dir(sample_dir: str):
    """Resolve result directory, supporting both current and nested layouts."""
    if _has_component_results(sample_dir):
        return sample_dir

    nested_result = os.path.join(sample_dir, "result")
    if _has_component_results(nested_result):
        return nested_result

    return None


def process_one_sample(sample_dir: str):
    """Process one PAC sample directory with auto result directory detection."""
    target_result_dir = _resolve_result_dir(sample_dir)
    if target_result_dir is None:
        print(f"[Skip] {sample_dir}: no usable result directory found")
        return

    # Output directory
    output_folder = os.path.join(target_result_dir, "comparison_results")
    os.makedirs(output_folder, exist_ok=True)

    component_files = sorted(
        [f for f in os.listdir(target_result_dir) if f.endswith("-component result.xlsx")]
    )
    if not component_files:
        print(f"[Skip] {target_result_dir}: no component result files found")
        return

    # Extract and sort component numbers
    component_nums = []
    for fn in component_files:
        m = re.match(r"^(\d+)-component result\.xlsx$", fn)
        if m:
            n = int(m.group(1))
            if n >= 2:
                component_nums.append(n)

    if not component_nums:
        print(f"[Skip] {target_result_dir}: no valid component indices found")
        return

    # Load standard library file names
    files = os.listdir(standard_folder)
    emission_files = [f for f in files if 'emissiontable' in f and not f.startswith('~$')]

    for i in sorted(component_nums):
        target_file = os.path.join(target_result_dir, f"{i}-component result.xlsx")
        output_path = os.path.join(output_folder, f"{i}_comparison_results.xlsx")

        print(f"\nProcessing target file: {target_file}")

        # Read target file
        target_df = pd.read_excel(target_file, sheet_name=None)
        target_em = target_df['Em loadings']
        target_ex = target_df['Ex loadings']

        target_em_x = target_em.iloc[:, 0].values
        target_em_y = target_em.iloc[:, 1:].values
        target_ex_x = target_ex.iloc[:, 0].values
        target_ex_y = target_ex.iloc[:, 1:].values

        # Initialize result container
        grouped_results = {f"C{idx+1}": [] for idx in range(target_em_y.shape[1])}

        for em_file in emission_files:
            base_name = em_file.replace('_emissiontable.xlsx', '')
            ex_file = base_name + '_excitationtable.xlsx'

            em_path = os.path.join(standard_folder, em_file)
            ex_path = os.path.join(standard_folder, ex_file)

            if not os.path.exists(ex_path):
                print(f"Missing matching excitation file, skipped: {ex_file}")
                continue

            try:
                standard_em = pd.read_excel(em_path)
                standard_ex = pd.read_excel(ex_path)

                standard_em_x = standard_em.iloc[:, 0].values
                standard_em_y = standard_em.iloc[:, 1:].values
                standard_ex_x = standard_ex.iloc[:, 0].values
                standard_ex_y = standard_ex.iloc[:, 1:].values

                # Iterate target components
                for k in range(target_em_y.shape[1]):
                    t_em_col = target_em_y[:, k]
                    t_ex_col = target_ex_y[:, k]
                    target_component = f"C{k+1}"

                    # Match against each standard component
                    for j in range(standard_em_y.shape[1]):
                        s_em_col = standard_em_y[:, j]
                        s_ex_col = standard_ex_y[:, j]
                        sample_component = f"C{j+1}"

                        q_em = calculate_similarity_overlap(
                            target_em_x, t_em_col,
                            standard_em_x, s_em_col
                        )
                        q_ex = calculate_similarity_overlap(
                            target_ex_x, t_ex_col,
                            standard_ex_x, s_ex_col
                        )
                        overall_q = q_em * q_ex

                        grouped_results[target_component].append({
                            "ID": base_name,
                            "Target component": target_component,
                            "Standard component": sample_component,
                            "Emission q": round(q_em, 4),
                            "Excitation q": round(q_ex, 4),
                            "Overall q (q_ex * q_em)": round(overall_q, 4)
                        })

                print(f"Completed: {base_name}")
            except Exception as e:
                print(f"Failed: {base_name}, reason: {e}")

        # Save results
        with pd.ExcelWriter(output_path) as writer:
            for component, records in grouped_results.items():
                df = pd.DataFrame(records)
                df = df.sort_values(by="Overall q (q_ex * q_em)", ascending=False)
                df.to_excel(writer, sheet_name=component, index=False)

        print(f"Saved comparison result: {output_path}")


print(f"PAC root: {pac_root}")

if selected_samples:
    target_samples = selected_samples
else:
    target_samples = sorted(
        [d for d in os.listdir(pac_root) if os.path.isdir(os.path.join(pac_root, d))]
    )
    if _has_component_results(pac_root):
        target_samples = ["."] + target_samples

for sample in target_samples:
    # Accept full directories or subfolder names under pac_root
    sample_dir = sample if os.path.isabs(sample) else os.path.join(pac_root, sample)
    sample_result_dir = _resolve_result_dir(sample_dir)
    if sample_result_dir is None:
        print(f"[Skip] No valid result directory found: {sample_dir}")
        continue
    print(f"\nStart processing sample: {sample}")
    process_one_sample(sample_result_dir)

print("\nAll processing completed")
