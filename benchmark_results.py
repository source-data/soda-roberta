import os
import glob
import json
import pandas as pd

FOLDER = './data/ray_results/'
OUTPUT_FOLDER = "/home/jabreu/data/hp_results"
keep_columns = ["eval_accuracy_score", "eval_precision", "eval_recall", "eval_f1", "config"]
result_df = pd.DataFrame({})
for folder in glob.glob(f"{FOLDER}/*"):
    for f in glob.glob(f"{folder}/*"):
        if os.path.isdir(f):
            os.system(f"cp {os.path.join(f,'result.json')} {OUTPUT_FOLDER}/result_{f.split('/')[-1]}.json")
            df = pd.read_json(os.path.join(f,'result.json'), lines=True)
            try:
                df = df[keep_columns]
            except KeyError:
                continue
            df['model'] = folder.split('/')[-1]
            df['task'] = folder.split('/')[-1].split('_')[-1]
            df = df.iloc[df["eval_f1"].idxmax()]
            result_df = result_df.append(df, ignore_index=True)

aggregate_df = pd.DataFrame(result_df.groupby(["model", "task"])[["eval_precision","eval_recall","eval_f1"]].max().reset_index())
print(aggregate_df.sort_values("eval_f1", ascending=False))
