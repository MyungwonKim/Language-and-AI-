import pandas as pd

d_mask = pd.read_csv("C:\\Users\\jelic\\Desktop\\L&AI\\MBTI_self_descriptions_only.csv")
print(d_mask.head())
d_masked = d_mask.copy()
PLACEHOLDER = "[MBTI_SELF_DESC]"

def mask_self_desc(row):
    text = row.get("final_text", "")
    match = row.get("matched_text", None)

    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # If no match (NaN/None/empty), return original text unchanged
    if match is None or (isinstance(match, float) and pd.isna(match)) or (isinstance(match, str) and match.strip() == ""):
        return text

    match = str(match)

    # Replace ONLY the first occurrence to avoid wiping repeated mentions accidentally
    return text.replace(match, PLACEHOLDER, 1)

d_masked["final_text_masked"] = d_masked.apply(mask_self_desc, axis=1)

# Quick sanity check
pd.set_option("display.max_colwidth", None)
print(d_masked["final_text_masked"].head(1))
# d_masked.to_csv("dataset_masked.csv", index=False)