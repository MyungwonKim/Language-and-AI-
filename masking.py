import pandas as pd

d_mask = pd.read_csv("C:\\Users\\jelic\\Desktop\\L&AI\\this.csv")
d_masked = d_mask.copy()
PLACEHOLDER = "[MBTI_SELF_DESC]"
print(d_mask.head(20))
def mask_self_desc(row):
    text = row.get("clean_text", "")
    match = row.get("matched_text", None)

    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # if no match, return original text unchanged
    if match is None or (isinstance(match, float) and pd.isna(match)) or (isinstance(match, str) and match.strip() == ""):
        return text

    match = str(match)

    # replace  the first occurrence to avoid wiping repeated mentions accidentally
    return text.replace(match, PLACEHOLDER, 1)

d_masked["final_text_masked"] = d_masked.apply(mask_self_desc, axis=1)

# check
# pd.set_option("display.max_colwidth", None)
print(d_masked["final_text_masked"].head(1)) #Since text is long ctrl + f and type: "I've completely" and you will see the masked part in the sentence
d_masked.to_csv("dataset_masked_v2.csv", index=False)
print(len(d_masked))