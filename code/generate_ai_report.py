import os
import json
import re
import pandas as pd
from openai import OpenAI


def read_api_key(api_file_path):
    """Read API key from file."""
    try:
        with open(api_file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Failed to read API key: {e}")
        return None


def get_relevant_samples(excel_file):
    """Get best-match sample IDs and corresponding fluorescence components."""
    try:
        df = pd.read_excel(excel_file)
        # Required columns from the generated analysis table.
        required_columns = ["Best sample ID (global q)", "Fluorescence component (sheet)"]
        if not all(col in df.columns for col in required_columns):
            print(f"Excel file is missing required columns: {required_columns}")
            return []

        result = []
        for _, row in df.iterrows():
            sample_id = row["Best sample ID (global q)"]
            component = row["Fluorescence component (sheet)"]
            result.append({"id": sample_id, "component": component})

        return result
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return []


def normalize_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def split_components(cell_value):
    text = normalize_text(cell_value)
    if not text:
        return set()
    parts = re.split(r"[,;\s]+", text)
    return {p for p in parts if p}


def pick_col(df, candidates):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def find_category_for_component(classify_row, component, id_col):
    comp = normalize_text(component)
    if not comp:
        return ""

    for col in classify_row.index:
        if col == id_col:
            continue
        if comp in split_components(classify_row[col]):
            return str(col)
    return ""


def lookup_meta(meta_df, sample_id):
    row = meta_df[meta_df["ID"] == sample_id]
    if row.empty:
        return "", "", ""
    rec = row.iloc[0]
    return str(rec.get("Reference", "")), str(rec.get("Sources", "")), str(rec.get("Ecozones", ""))


def build_title_lookup(meta_file):
    """Build lookup dictionary: ID -> literature title."""
    if not os.path.exists(meta_file):
        return {}

    meta_df = pd.read_excel(meta_file)
    id_col = pick_col(meta_df, ["ID"])
    title_col = pick_col(meta_df, ["Title"])
    if id_col is None or title_col is None:
        return {}

    title_map = {}
    for _, row in meta_df[[id_col, title_col]].iterrows():
        sid = normalize_id(row[id_col])
        if sid:
            title_map[sid] = "" if pd.isna(row[title_col]) else str(row[title_col]).strip()
    return title_map


def build_analysis_excel(compare_file, classify_file, meta_file, output_file):
    """Generate analysis_result_2.xlsx from comparison results."""
    if not os.path.exists(compare_file):
        raise FileNotFoundError(f"Comparison result file not found: {compare_file}")
    if not os.path.exists(classify_file):
        raise FileNotFoundError(f"Classification result file not found: {classify_file}")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"OpenFluor metadata file not found: {meta_file}")

    classify_df = pd.read_excel(classify_file)
    id_col_cls = pick_col(classify_df, ["ID"])
    if id_col_cls is None:
        raise ValueError("Classification file is missing ID column")
    classify_df[id_col_cls] = classify_df[id_col_cls].map(normalize_id)

    meta_df = pd.read_excel(meta_file)
    id_col_meta = pick_col(meta_df, ["ID"])
    ref_col = pick_col(meta_df, ["Reference"])
    src_col = pick_col(meta_df, ["Sources"])
    eco_col = pick_col(meta_df, ["Ecozones"])
    if None in [id_col_meta, ref_col, src_col, eco_col]:
        raise ValueError("OpenFluor metadata is missing ID/Reference/Sources/Ecozones columns")

    meta_df = meta_df[[id_col_meta, ref_col, src_col, eco_col]].copy()
    meta_df.columns = ["ID", "Reference", "Sources", "Ecozones"]
    meta_df["ID"] = meta_df["ID"].map(normalize_id)

    workbook = pd.read_excel(compare_file, sheet_name=None)
    result_rows = []

    for sheet_name, df in workbook.items():
        if df is None or df.empty:
            continue

        id_col = pick_col(df, ["ID"])
        q_col = pick_col(df, ["Overall q (q_ex * q_em)"])
        std_comp_col = pick_col(df, ["Standard component"])

        if None in [id_col, q_col, std_comp_col]:
            print(f"[Skip] Sheet {sheet_name} is missing required columns (ID/Overall q/Standard component)")
            continue

        work = df.copy()
        work[id_col] = work[id_col].map(normalize_id)
        work[q_col] = pd.to_numeric(work[q_col], errors="coerce")
        work = work.dropna(subset=[q_col])
        if work.empty:
            continue

        cnt_over_07 = int((work[q_col] > 0.7).sum())
        idx_global = work[q_col].idxmax()
        global_id = normalize_id(work.loc[idx_global, id_col])
        global_q = float(work.loc[idx_global, q_col])
        global_std_comp = str(work.loc[idx_global, std_comp_col])

        cls_row_global = classify_df[classify_df[id_col_cls] == global_id]
        matched_category = ""
        if not cls_row_global.empty:
            matched_category = find_category_for_component(cls_row_global.iloc[0], sheet_name, id_col_cls)

        ref_all, src_all, eco_all = lookup_meta(meta_df, global_id)

        cat_best_id = ""
        cat_best_q = None
        cat_best_std_comp = ""
        ref_cat = ""
        src_cat = ""
        eco_cat = ""

        if matched_category:
            category_rows = []
            for _, r in work.iterrows():
                rid = normalize_id(r[id_col])
                cls_row = classify_df[classify_df[id_col_cls] == rid]
                if cls_row.empty:
                    continue
                r_cat = find_category_for_component(cls_row.iloc[0], sheet_name, id_col_cls)
                if r_cat == matched_category:
                    category_rows.append(r)

            if category_rows:
                cat_df = pd.DataFrame(category_rows)
                idx_cat = cat_df[q_col].idxmax()
                cat_best_id = normalize_id(cat_df.loc[idx_cat, id_col])
                cat_best_q = float(cat_df.loc[idx_cat, q_col])
                cat_best_std_comp = str(cat_df.loc[idx_cat, std_comp_col])
                ref_cat, src_cat, eco_cat = lookup_meta(meta_df, cat_best_id)

        result_rows.append(
            {
                "Fluorescence component (sheet)": sheet_name,
                "Sample count with q > 0.7": cnt_over_07,
                "Best sample ID (global q)": global_id,
                "Max global q": round(global_q, 4),
                "Best standard component (global q)": global_std_comp,
                "Global-Reference": ref_all,
                "Global-Sources": src_all,
                "Global-Ecozones": eco_all,
                "Matched category": matched_category,
                "Best sample ID (category q)": cat_best_id,
                "Max category q": "" if cat_best_q is None else round(cat_best_q, 4),
                "Best standard component (category q)": cat_best_std_comp,
                "Category-Reference": ref_cat,
                "Category-Sources": src_cat,
                "Category-Ecozones": eco_cat,
            }
        )

    result_df = pd.DataFrame(result_rows)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_excel(output_file, index=False)
    print(f"Analysis table generated: {output_file}")


def read_excel_to_markdown(excel_file):
    """Read an Excel file and convert it to a Markdown table."""
    try:
        df = pd.read_excel(excel_file)
        return dataframe_to_markdown(df)
    except Exception as e:
        print(f"Failed to read Excel file {excel_file}: {e}")
        return f"Unable to read file {excel_file}: {str(e)}"


def read_ipynb_to_markdown(ipynb_file):
    """Read a Jupyter notebook file and convert it to Markdown."""
    try:
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        markdown_content = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                # Append markdown cell content.
                markdown_content.extend(cell['source'])
            elif cell['cell_type'] == 'code':
                # Wrap code cell content in a fenced code block.
                markdown_content.append('\n```python')
                markdown_content.extend(cell['source'])
                markdown_content.append('```\n')

        return ''.join(markdown_content)
    except Exception as e:
        print(f"Failed to read Jupyter notebook file {ipynb_file}: {e}")
        return f"Unable to read file {ipynb_file}: {str(e)}"


def read_md_file(md_file_path):
    """Read Markdown file content."""
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read Markdown file {md_file_path}: {e}")
        return None


def read_sample_md(sample_id, md_paths):
    """Read a sample markdown file from multiple candidate paths."""
    for base_path in md_paths:
        md_file_path = f"{base_path}/{sample_id}.md"
        content = read_md_file(md_file_path)
        if content:
            return content, md_file_path
    return None, ""


def extract_md_title(md_content, default_title=""):
    """Extract the first level-1 heading as the literature title."""
    if not md_content:
        return default_title

    for line in md_content.splitlines():
        text = line.strip()
        if text.startswith("# "):
            title = text[2:].strip()
            if title:
                return title
    return default_title


def strip_first_h1(md_content):
    """Remove the first level-1 heading from markdown text for cleaner embedding."""
    if not md_content:
        return md_content

    lines = md_content.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("# "):
            cleaned = lines[:idx] + lines[idx + 1:]
            while cleaned and not cleaned[0].strip():
                cleaned.pop(0)
            return "\n".join(cleaned)
    return md_content


def analyze_with_deepseek(client, content, component):
    """Analyze fluorescence component semantics using DeepSeek API."""
    try:
        prompt = f"""Please analyze the document below and explain what \"{component}\" refers to.

Context:
- C1, C2, C3... represent fluorescence component identifiers from EEM/PARAFAC.
- They are not molecular formulas and not sample IDs.

Please interpret this fluorescence component in the literature context,
including potential source, property, and environmental implication.

{content}

Please provide a clear and specific explanation for \"{component}\"."""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional literature analysis assistant. Interpret C1/C2/C3 as fluorescence component identifiers in EEM/PARAFAC context."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek API call failed: {e}")
        return f"Error while analyzing {component}: {str(e)}"


def dataframe_to_markdown(df):
    """Convert DataFrame to markdown table format."""
    if df is None or df.empty:
        return "No data"

    # Build table header.
    md = "| " + " | ".join(df.columns) + " |\n"
    # Build separator row.
    md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    # Build body rows.
    for _, row in df.iterrows():
        md += "| " + " | ".join([str(x) for x in row.values]) + " |\n"

    return md


def main():
    # File path configuration
    compare_excel = r"..\data\result\comparison_results\2_comparison_results.xlsx"
    classify_excel = r"..\docs\classification_results.xlsx"
    openfluor_excel = r"..\docs\openflour_knowledge_base\OpenFluor_Measurements.xlsx"
    analysis_excel = r"..\data\result\analysis_result_2.xlsx"
    split_analysis_excel = r"..\data\process\split_half_summary.xlsx"
    metric_formulas_md = r"..\docs\component_analysis_metric_formulas.md"
    api_file = r"..\docs\API.txt"
    md_paths = [
        r"..\docs\openflour_knowledge_base\openflour_literature\md_knowledge_base",
        r"..\docs\openflour_knowledge_base\openflour_literature\process"
    ]
    output_md = r"..\outputs\fluorescence_analysis_report.md"

    # Build analysis table first, to guarantee downstream inputs.
    build_analysis_excel(compare_excel, classify_excel, openfluor_excel, analysis_excel)

    # Initialize DeepSeek client
    api_key = read_api_key(api_file)
    if not api_key:
        print("Failed to get API key, exiting")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    print("DeepSeek client initialized")

    # Read best-matched sample records
    samples = get_relevant_samples(analysis_excel)
    if not samples:
        print("Warning: no valid sample records found; continue with remaining content")

    # Read and convert all files needed for report content
    print("Reading analysis_result_2...")
    analysis_excel_content = read_excel_to_markdown(analysis_excel)

    print("Reading split_half_summary...")
    split_analysis_content = read_excel_to_markdown(split_analysis_excel)

    print("Reading component_analysis_metric_formulas.md...")
    metric_formulas_content = read_md_file(metric_formulas_md)
    if not metric_formulas_content:
        metric_formulas_content = "Metric formula markdown was not found or could not be read."
    else:
        metric_formulas_content = strip_first_h1(metric_formulas_content)

    # Process each sample
    all_results = []
    for sample in samples:
        sample_id = normalize_id(sample["id"])
        component = str(sample["component"])
        print(f"Processing sample {sample_id}, component {component}...")

        # Read corresponding literature markdown (primary path first, fallback second).
        md_content, md_file_path = read_sample_md(sample_id, md_paths)
        if not md_content:
            print(f"No literature found for sample {sample_id}, skipped")
            continue

        md_title = extract_md_title(md_content, default_title="")

        # Run DeepSeek analysis.
        analysis_result = analyze_with_deepseek(client, md_content, component)

        # Save analysis result (without raw markdown body).
        all_results.append({
            "sample_id": sample_id,
            "component": component,
            "title": md_title,
            "source_md": md_file_path,
            "analysis": analysis_result
        })

    # Write final markdown report.
    try:
        os.makedirs(os.path.dirname(output_md), exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("# Fluorescence Component Analysis Report\n\n")

            # Section 1: Metric formula notes markdown
            f.write("## 1. Component Analysis Metric Formula Notes\n\n")
            f.write(metric_formulas_content)
            f.write("\n\n---\n\n")

            # Section 2: Analysis table
            f.write("## 2. analysis_result_2\n\n")
            f.write(analysis_excel_content)
            f.write("\n\n---\n\n")

            # Section 3: Split-half summary
            f.write("## 3. Split-half Summary\n\n")
            f.write(split_analysis_content)
            f.write("\n\n---\n\n")

            # Section 4: DeepSeek interpretation results
            f.write("## 4. Literature Interpretation by Component\n\n")
            for result in all_results:
                f.write(f"### Sample {result['sample_id']} - Component {result['component']}\n\n")
                title_text = result["title"] if result["title"] else f"Title not found (source: {result['source_md']})"
                f.write(f"Literature title: {title_text}\n\n")
                f.write(result['analysis'])
                f.write("\n\n---\n\n")

            if not all_results:
                f.write("No usable literature samples were found in this run, so no API interpretation content was generated.\n\n")

        print(f"Analysis report saved to {output_md}")
    except Exception as e:
        print(f"Failed to save analysis report: {e}")


if __name__ == "__main__":
    main()
