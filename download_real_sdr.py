"""Download the public measured SDR OFDM validation dataset."""

from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/rikluost/sdr_ofdm_dataset_v1/main/ofdm_valset_4_8_sdr.pth"


def main():
    out = Path("data") / "ofdm_valset_4_8_sdr.pth"
    out.parent.mkdir(exist_ok=True)
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"Dataset already exists: {out}")
        return
    print("Downloading ~27 MB measured SDR OFDM dataset...")
    urlretrieve(URL, out)
    print(f"Saved: {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
