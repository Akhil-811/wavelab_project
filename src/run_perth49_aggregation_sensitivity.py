from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("data/raw/WEC_Perth_49.csv")
OUT = Path("reports/sensitivity_perth49")
SEEDS = [42]
SPLIT_SEED = 42
EPOCHS = 60
PATIENCE = 15
BATCH = 256

class CoordinateMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,256), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def score(y, p):
    return {
        "mae": float(mean_absolute_error(y,p)),
        "rmse": float(np.sqrt(mean_squared_error(y,p))),
        "r2": float(r2_score(y,p)),
    }

def train_eval(Xtr, ytr, Xv, yv, Xte, yte, seed, label):
    seed_all(seed)
    device=torch.device("cpu")
    xs=StandardScaler(); ys=StandardScaler()
    Xtr=xs.fit_transform(Xtr).astype("float32")
    Xv=xs.transform(Xv).astype("float32")
    Xte=xs.transform(Xte).astype("float32")
    ytr_s=ys.fit_transform(ytr.reshape(-1,1)).astype("float32")
    yv_s=ys.transform(yv.reshape(-1,1)).astype("float32")
    loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr_s)),
                      batch_size=BATCH,shuffle=True)
    Xt=torch.from_numpy(Xv); yt=torch.from_numpy(yv_s)
    model=CoordinateMLP(Xtr.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loss_fn=nn.MSELoss()
    best=float("inf"); best_state=None; best_epoch=0; stale=0
    for epoch in range(1,EPOCHS+1):
        model.train()
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred=ys.inverse_transform(model(Xt).numpy()).ravel()
        val_mae=mean_absolute_error(yv,pred)
        if val_mae < best:
            best=float(val_mae); best_epoch=epoch; stale=0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=PATIENCE: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred=ys.inverse_transform(model(torch.from_numpy(Xte)).numpy()).ravel()
    return pred, best_epoch, best

def main():
    global DATA, OUT, SEEDS, SPLIT_SEED, EPOCHS, PATIENCE
    parser = argparse.ArgumentParser(description="Paired Perth 49 row-level vs layout-mean target sensitivity experiment.")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[21, 42, 123])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()
    DATA, OUT = Path(args.data), Path(args.output_dir)
    SEEDS, SPLIT_SEED = args.seeds, args.split_seed
    EPOCHS, PATIENCE = args.epochs, args.patience
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA)
    xcols=[]
    for i in range(1,50): xcols += [f"X{i}",f"Y{i}"]
    target="Total_Power"
    X=df[xcols].to_numpy(dtype=np.float32)
    y=df[target].to_numpy(dtype=np.float64)
    # Stable exact-layout key; no rounding or coordinate modification.
    groups=pd.util.hash_pandas_object(df[xcols],index=False).to_numpy(dtype="uint64")
    # Fixed layout partitions shared by original-row and aggregated-target arms.
    s1=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SPLIT_SEED)
    trv,te=next(s1.split(X,y,groups))
    s2=GroupShuffleSplit(n_splits=1,test_size=.1765,random_state=SPLIT_SEED)
    trrel,vrel=next(s2.split(X[trv],y[trv],groups[trv]))
    tr=trv[trrel]; va=trv[vrel]
    split={"train":tr,"validation":va,"test":te}
    # Build layout-level table once, preserving row-to-layout mapping.
    layout_df=df[xcols].copy()
    layout_df["_layout_id"]=groups
    layout_df["_target"]=y
    agg=layout_df.groupby("_layout_id",sort=True).agg(
        **{c:(c,"first") for c in xcols},
        mean_target=("_target","mean"),
        median_target=("_target","median"),
        rows_in_layout=("_target","size"),
        target_std=("_target","std"),
        target_min=("_target","min"),
        target_max=("_target","max")
    ).reset_index()
    agg.to_csv(OUT/"perth49_layout_mean_targets.csv",index=False)
    # For each partition, retain corresponding layout IDs and aggregate records.
    layout_part={}
    for name,idx in split.items():
        layout_part[name]=np.unique(groups[idx])
    result=[]
    pred_rows=[]
    for seed in SEEDS:
        # Original row-level target training, grouped split.
        p_orig, ep_orig, val_orig=train_eval(
            X[tr],y[tr],X[va],y[va],X[te],y[te],seed,"original")
        m_orig=score(y[te],p_orig)
        # Aggregated one-row-per-layout training/validation.
        agg_by=agg.set_index("_layout_id")
        def select(ids):
            part=agg_by.loc[ids]
            return part[xcols].to_numpy(np.float32),part["mean_target"].to_numpy(np.float64)
        Xa,ya=select(layout_part["train"])
        Xv,yv=select(layout_part["validation"])
        Xtest_layout,ytest_layout=select(layout_part["test"])
        p_agg_layout,ep_agg,val_agg=train_eval(Xa,ya,Xv,yv,Xtest_layout,ytest_layout,seed,"aggregated")
        m_agg_mean=score(ytest_layout,p_agg_layout)
        # Evaluate aggregate model on every original test row as well, mapping by layout.
        id_to_pred=dict(zip(layout_part["test"],p_agg_layout))
        p_agg_raw=np.array([id_to_pred[g] for g in groups[te]],dtype=float)
        m_agg_raw=score(y[te],p_agg_raw)
        result += [
          {"seed":seed,"arm":"original_row_level","evaluation":"original_test_rows",
           "train_examples":len(tr),"validation_examples":len(va),"test_examples":len(te),
           "train_layouts":len(layout_part["train"]),"validation_layouts":len(layout_part["validation"]),
           "test_layouts":len(layout_part["test"]),"best_epoch":ep_orig,"best_validation_mae":val_orig,**m_orig},
          {"seed":seed,"arm":"layout_mean_aggregated","evaluation":"original_test_rows",
           "train_examples":len(Xa),"validation_examples":len(Xv),"test_examples":len(te),
           "train_layouts":len(layout_part["train"]),"validation_layouts":len(layout_part["validation"]),
           "test_layouts":len(layout_part["test"]),"best_epoch":ep_agg,"best_validation_mae":val_agg,**m_agg_raw},
          {"seed":seed,"arm":"layout_mean_aggregated","evaluation":"layout_mean_targets",
           "train_examples":len(Xa),"validation_examples":len(Xv),"test_examples":len(Xtest_layout),
           "train_layouts":len(layout_part["train"]),"validation_layouts":len(layout_part["validation"]),
           "test_layouts":len(layout_part["test"]),"best_epoch":ep_agg,"best_validation_mae":val_agg,**m_agg_mean},
        ]
        for j,row_idx in enumerate(te):
            pred_rows.append({"seed":seed,"row_index":int(row_idx),"layout_id":int(groups[row_idx]),
              "actual_original_target":float(y[row_idx]),"prediction_original_model":float(p_orig[j]),
              "prediction_aggregated_model":float(p_agg_raw[j])})
        print(f"seed {seed}: original MAE={m_orig['mae']:.2f}; aggregated vs raw MAE={m_agg_raw['mae']:.2f}; aggregated vs layout-mean MAE={m_agg_mean['mae']:.2f}")
    res=pd.DataFrame(result)
    res.to_csv(OUT/"sensitivity_metrics_by_seed.csv",index=False)
    summary=res.groupby(["arm","evaluation"]).agg(
       runs=("seed","count"),train_examples=("train_examples","mean"),
       test_examples=("test_examples","mean"),mae_mean=("mae","mean"),mae_sd=("mae","std"),
       rmse_mean=("rmse","mean"),r2_mean=("r2","mean")
    ).reset_index()
    summary.to_csv(OUT/"sensitivity_summary.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(OUT/"test_predictions_comparison.csv",index=False)
    metadata={
      "source_file":str(DATA),"source_sha256":__import__("hashlib").sha256(DATA.read_bytes()).hexdigest(),
      "source_rows":len(df),"unique_exact_layouts":int(len(np.unique(groups))),
      "original_train_rows":len(tr),"original_validation_rows":len(va),"original_test_rows":len(te),
      "train_layouts":len(layout_part["train"]),"validation_layouts":len(layout_part["validation"]),
      "test_layouts":len(layout_part["test"]),"seeds":SEEDS,"split_seed":SPLIT_SEED,"max_epochs":EPOCHS,"early_stopping_patience":PATIENCE,
      "aggregation":"mean Total_Power per exact coordinate layout, after layout-level partition assignment",
      "original_data_modified":False,
      "caveat":"Aggregated model is also evaluated against original row-level test labels; layout-mean evaluation is secondary and answers a different target question."
    }
    (OUT/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2))
    print("\nSUMMARY\n",summary.to_string(index=False))
    print("\nWrote",OUT)

if __name__=="__main__": main()
