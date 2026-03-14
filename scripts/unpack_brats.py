import argparse
import os
import zipfile


def unpack(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".zip"):
                zpath = os.path.join(dirpath, name)
                out_dir = os.path.splitext(zpath)[0]
                if os.path.exists(out_dir):
                    continue
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    unpack(args.root)


if __name__ == "__main__":
    main()
