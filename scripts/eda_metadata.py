import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/eda")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(os.path.join(args.root, "BraTS-PEDs_metadata.tsv"), sep="\t")

    plt.figure(figsize=(6, 4))
    sns.histplot(df["Age at imaging (days)"].dropna(), bins=30, kde=True)
    plt.title("Age Distribution (days)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "age_hist.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(4, 4))
    sns.countplot(data=df, x="Sex_at_birth")
    plt.title("Sex at Birth")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "sex_count.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
