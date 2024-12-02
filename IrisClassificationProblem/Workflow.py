import torch
from pathlib import Path
import kaggle
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from timeit import default_timer as timer
from IrisDataset import IrisDataset
from IrisModel import IrisModel as IM

kaggle.api.authenticate()
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("data/")
image_path = data_path/"iris"

if image_path.is_dir():
    print(f"{image_path} already exist... Skipping Downloading")
else:
    print(f"{image_path} does not exist... Creating One")
    image_path.mkdir(parents=True, exist_ok=True)

if(os.path.isfile(image_path/"Iris.csv")):
    print("File already exist")
else:
    kaggle.api.dataset_download_files("uciml/iris", path = image_path, unzip = True)

csv_file = image_path/"Iris.csv"

df = pd.read_csv(csv_file)

print(df.head())
df = df.drop(columns=["Id"])
print(df.head())
print(df.describe())
print(df.info())
print(df["Species"].value_counts())
print(df.isnull().sum())

# df['SepalLengthCm'].hist()
#Scatter Colors
colors = ["red", "green", "blue"]
species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
# for i in range(len(species)):
#     x = df[df['Species'] == species[i]]
#     plt.scatter(x["SepalLengthCm"], x["SepalWidthCm"], c = colors[i], label = species[i])
#     # x["PetalLengthCm"], x["PetalWidthCm"]
# plt.xlabel("Sepal Length")
# plt.ylabel("Sepal Width")
# plt.legend()
# plt.show()
#
# for i in range(len(species)):
#     x = df[df['Species'] == species[i]]
#     plt.scatter(x["PetalLengthCm"], x["PetalWidthCm"], c = colors[i], label = species[i])
#     # x["PetalLengthCm"], x["PetalWidthCm"]
# plt.xlabel("Petal Length")
# plt.ylabel("Petal Width")
# plt.legend()
# plt.show()
corr = df.corr(numeric_only=True)

le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])
print(df.head())

X = df.drop(columns=["Species"], axis=1)
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.DataFrame(X_train, columns=X.columns)
train_df["Species"] = y_train.values
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["Species"] = y_test.values

train_df.to_csv("Iris_train.csv", index = False)
test_df.to_csv("Iris_test.csv", index=False)

train_dataset = IrisDataset(csv_file="Iris_train.csv")
test_dataset = IrisDataset(csv_file="Iris_test.csv")

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for batch in train_dataloader:
    print(batch["features"], batch["label"])
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
model_1 = IM(input_shape=int(X_train.shape[1]), hidden_units=25, output_shape=3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


start_time = timer()
epochs = 35
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_losses = []
test_losses = []
train_acc = []
test_acc = []

#Beginning the Training Loop
for epoch in tqdm(range(epochs)):
    model_1.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch in train_dataloader:
        X_batch, y_batch = batch['features'].to(device), batch["label"].to(device)

        #Forward Pass
        y_logits = model_1(X_batch).squeeze()
        y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

        #Calculates the loss
        loss = loss_fn(y_logits, y_batch)
        running_loss += loss.item() * X_batch.size(0)
        acc = accuracy_fn(y_true=y_batch, y_pred=y_preds)
        correct_train += acc * X_batch.size(0)
        total_train += X_batch.size(0)

        #Backward pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Training Metrics
    train_losses.append(running_loss/total_train)
    train_acc.append(correct_train/total_train)

    #Evaluation Mode
    model_1.eval()
    correct_test = 0
    running_test_loss = 0.0
    total_test = 0

    #Testing
    with torch.inference_mode():
        for batch in test_dataloader:
            X_batch, y_batch = batch['features'].to(device), batch["label"].to(device)

            #Forward Pass
            test_logits = model_1(X_batch).squeeze()
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

            #Calculate the Loss
            test_loss = loss_fn(test_logits, y_batch)
            running_test_loss += test_loss.item() * X_batch.size(0)
            acc_test = accuracy_fn(y_true=y_batch, y_pred=test_pred)
            correct_test += acc_test * X_batch.size(0)
            total_test += X_batch.size(0)

        #Test Metrics
        test_losses.append(running_test_loss/total_test)
        test_acc.append(correct_test/total_test)

        if epoch % 10 == 0:
            print(f"\nEpoch: {epoch}, Loss: {loss.item():5f}, Test Loss: {test_loss.item():5f}, "
                  f"Accuracy: {acc:2f}, Accuracy Test: {acc_test:2f}")
    end_time = timer()


#Visualizing the Output
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(epochs), train_losses, label="Train Loss")
plt.plot(range(epochs), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()


plt.subplot(1,2,2)
plt.plot(range(epochs), train_acc, label="Train Accuracy")
plt.plot(range(epochs), test_acc, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

