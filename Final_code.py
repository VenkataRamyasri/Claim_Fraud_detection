import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
url = "https://zenodo.org/record/13289814/files/Health%20Insurance%20Fraud%20Claims.xlsx?download=1"
df = pd.read_excel(url)
print("Data shape:", df.shape)
df['ClaimLegitimacy'] = df['ClaimLegitimacy'].astype(str).str.lower()
label_map = {"legitimate":0,"real":0,"fraud":1,"fraudulent":1}
df['ClaimLegitimacy'] = df['ClaimLegitimacy'].replace(label_map)
y = df['ClaimLegitimacy'].astype(int).values
print("Unique labels:", np.unique(y))
# Detect datetime columns
datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
print("Datetime columns:", datetime_cols)

# Convert datetime to numeric (seconds since epoch)
for c in datetime_cols:
    df[c] = df[c].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)
# Numeric features
num_cols = ['ClaimAmount', 'PatientAge', 'PatientIncome', 'Cluster']

# Categorical features (everything else except label and ClaimID)
cat_cols = [c for c in df.columns if c not in num_cols + ['ClaimLegitimacy', 'ClaimID']]

# Fill missing numeric values
X_num = df[num_cols].fillna(df[num_cols].median())

# One-hot encode categorical
X_cat = pd.get_dummies(df[cat_cols].fillna("__MISSING__"))

# Combine numeric + categorical
X = np.hstack([X_num.values, X_cat.values])
y = df['ClaimLegitimacy'].values

print("Final X shape:", X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fit k-NN on training data
k = 10
knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(X_train)

# Compute graph score for train and test
graph_train = 1.0 / (1.0 + knn.kneighbors(X_train)[0].mean(axis=1))
graph_test  = 1.0 / (1.0 + knn.kneighbors(X_test)[0].mean(axis=1))

print("Graph feature computed.")
# One-hot encode categorical features
X_cat = pd.get_dummies(df[cat_cols].fillna("__MISSING__"))

# Save the trained one-hot column order
trained_feature_cols = X_cat.columns.tolist()

# Combine numeric + categorical for training
X_num = df[num_cols].fillna(df[num_cols].median())
X = np.hstack([X_num.values, X_cat.values])
y = df['ClaimLegitimacy'].values
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Define Autoencoder
class AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae = AE(X_train_scaled.shape[1])
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train Autoencoder on majority class only (label=0)
loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled[y_train==0], dtype=torch.float32)),
                    batch_size=64, shuffle=True)

EPOCHS = 20
for epoch in range(EPOCHS):
    for (batch,) in loader:
        optimizer.zero_grad()
        loss = loss_fn(ae(batch), batch)
        loss.backward()
        optimizer.step()

# Compute reconstruction error
with torch.no_grad():
    ae_train_err = ((ae(torch.tensor(X_train_scaled,dtype=torch.float32)) - torch.tensor(X_train_scaled,dtype=torch.float32))**2).mean(1).numpy()
    ae_test_err  = ((ae(torch.tensor(X_test_scaled,dtype=torch.float32)) - torch.tensor(X_test_scaled,dtype=torch.float32))**2).mean(1).numpy()

print("Autoencoder trained.")
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

# Combine original features + graph score + autoencoder error
X_train_final = np.hstack([X_train, graph_train.reshape(-1,1), ae_train_err.reshape(-1,1)])
X_test_final  = np.hstack([X_test,  graph_test.reshape(-1,1),  ae_test_err.reshape(-1,1)])

# Train XGBoost
dtrain = xgb.DMatrix(X_train_final, label=y_train)
dtest  = xgb.DMatrix(X_test_final,  label=y_test)

params = {"objective":"binary:logistic", "eval_metric":"auc"}
bst = xgb.train(params, dtrain, num_boost_round=200)

# Predict
pred_probs = bst.predict(dtest)
preds = (pred_probs >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, pred_probs))
def predict_claim(claim_dict):
    # 1️⃣ Numeric features
    x_num = np.array([[claim_dict.get(c, 0) for c in num_cols]])

    # 2️⃣ Categorical features
    claim_cat = pd.get_dummies(pd.DataFrame([claim_dict]))

    # ⚡ Key fix: match trained one-hot columns
    claim_cat = claim_cat.reindex(columns=trained_feature_cols, fill_value=0)

    # 3️⃣ Combine numeric + categorical
    x_feat = np.hstack([x_num, claim_cat.values])

    # 4️⃣ Graph score
    g = 1.0 / (1.0 + knn.kneighbors(x_feat)[0].mean())

    # 5️⃣ Autoencoder error
    x_scaled = scaler.transform(x_feat)
    ae_err = ((ae(torch.tensor(x_scaled, dtype=torch.float32)) - torch.tensor(x_scaled, dtype=torch.float32))**2).mean(1).item()

    # 6️⃣ Combine all features for XGBoost
    x_final = np.hstack([x_feat, [[g]], [[ae_err]]])
    prob = bst.predict(xgb.DMatrix(x_final))[0]

    return "Fraud" if prob >= 0.5 else "Real"
while True:
    s = input("Enter claim as Python dict (or type 'exit'): ")
    if s.lower() == 'exit':
        break

    try:
        # Convert input string to dictionary
        claim = eval(s)
        result = predict_claim(claim)
        print("Prediction:", result)
    except Exception as e:
        print("Error:", e)
        print("Make sure you enter a valid Python dictionary with all fields.")
