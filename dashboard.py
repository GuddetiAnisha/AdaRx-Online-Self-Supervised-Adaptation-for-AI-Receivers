import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot(csv_path, save=None):
    df=pd.read_csv(csv_path)
    fig, axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for name,g in df.groupby("method"):
        axes[0,0].plot(g.frame,g.ber,label=name); axes[0,1].plot(g.frame,g.bler,label=name)
        axes[1,0].plot(g.frame,g.throughput,label=name); axes[1,1].plot(g.frame,g.weight_drift,label=name)
    titles=["Bit error rate","Block error rate","Normalized throughput","Model weight drift"]
    for ax,title in zip(axes.ravel(),titles): ax.set_title(title); ax.grid(alpha=.3); ax.legend()
    axes[1,0].set_xlabel("Frame"); axes[1,1].set_xlabel("Frame"); fig.tight_layout()
    if save: fig.savefig(save,dpi=160,bbox_inches="tight")
    else: plt.show()

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--csv",required=True); p.add_argument("--save"); a=p.parse_args(); plot(a.csv,a.save)

