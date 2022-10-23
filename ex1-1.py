#The purpose of the model is trying to classify the deposit 
# for a given customer and campaign combination
import torch.nn as nn
import torch
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix ,accuracy_score
from sklearn.metrics import precision_score ,recall_score ,roc_curve ,auc ,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#Load dataset into memory using pandas
data = pd.read_csv('./bank.csv')
device=torch.device("cuda")

#print("Data shape:", data.shape)

#print out the target attribute in the dataset
#print("Distribution of Target Values in Dateset - ", data.deposit.value_counts())

#Check if we have "NA" values within the dataset
#print(data.isna().sum())

#Check the distinct datatypes within the dataset
#print(data.dtypes.value_counts())

#Extract categorical columns from dataset
categorical_columns = data.select_dtypes(include = "object").columns
#print("Categorical cols:", list(categorical_columns))

#For each categorical column if values in (Tes/No) convert into a 1/0 flag
for col in categorical_columns:
    if data[col].nunique() == 2:
        data[col] = np.where(data[col] == 'yes' ,1 ,0)
#print(data.head())

#For the remaining categorical variables; create one-hot encoded version of the dataset
new_data = pd.get_dummies(data)
#Define target and predictors for the model
target = "deposit"
predictors = set(new_data.columns) - set([target])
#print(new_data.head())

#Convert all datatypes within pandas dataframe to Float32 (Compatibility with Pytorch Tensor)
new_data = new_data.astype(np.float32)

#Split dataset into Train/Test[80:20]
x_train, x_test, y_train, y_test = train_test_split(new_data[predictors], new_data[target], test_size = 0.2)

#Convert Pandas dataframe, first to numpy and then to Torch Tensors
x_train = torch.from_numpy(x_train.values)
x_test = torch.from_numpy(x_test.values)
y_train = torch.from_numpy(y_train.values).reshape(-1,1)
y_test = torch.from_numpy(y_test.values).reshape(-1,1)

#Check the dataset size to verify
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#Define the function to train the model        
def train_network(model, optimizer, loss_function, num_epochs, batch_size, x_train, y_train, lambda_L1 = 0.0):
    loss_across_epochs = []
    for epoch in range(num_epochs):
        train_loss = 0.0

        #Explicitly(明確的) start model training
        model.train()
        for i in range(0,x_train.shape[0], batch_size):
            #Extract train batch from X and Y
            #prevent input data from getting over the number of training data
            input_data = x_train[i:min(x_train.shape[0], i + batch_size)].to(device)
            labels = y_train[i:min(y_train.shape[0], i + batch_size)].to(device)

            #Set the gradients to zero before starting to do backpropragation
            optimizer.zero_grad()

            #Forward pass
            output_data = model(input_data)

            #Calculate loss
            loss = loss_function(output_data, labels)
            L1_loss = 0

            #Compute L1 penalty to be add with loss
            for p in model.parameters():
                L1_loss = L1_loss + p.abs().sum()

            # Add L1 penalty to loss      
            L1_loss = loss + lambda_L1*L1_loss

            #Backpropogation
            loss.backward()

            #Update weights
            optimizer.step()

            train_loss += loss.item()*input_data.size(0)
            
        loss_across_epochs.append(train_loss/x_train.size(0))
        if epoch%100 == 0:
            print("Epoch:{} - Loss:{:.4f}".format(epoch, train_loss/x_train.size(0)))
    return(loss_across_epochs)
    
#Define function for evaluating NN
def evaluate_model(model, x_test, y_test, x_train, y_train, loss_list):
    model.eval() #Explicitly set to evaluate mode

    #Predict on Train and Validation Datasets
    y_test_prob = model(x_test)
    y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    y_train_prob = model(x_train)
    y_train_pred = np.where(y_train_prob > 0.5, 1, 0)

    #Compute Training and Validation Metrics
    print("\n Model Performance - ")
    print("Training Accuracy - ", round(accuracy_score(y_train, y_train_pred), 3))
    print("Taining Precision - ", round(precision_score(y_train, y_train_pred), 3))
    print("Training Recall - ", round(recall_score(y_train, y_train_pred), 3))
    print("Training ROCAUC", round(roc_auc_score(y_train, y_train_prob.detach().numpy()), 3)) #detach()去掉梯度資訊，畫圖不需要梯度!!
    print("Validation Accuracy - ", round(accuracy_score(y_test, y_test_pred), 3))
    print("Validation Precision - ", round(precision_score(y_test, y_test_pred), 3))
    print("Validation Recall - ", round(recall_score(y_test, y_test_pred), 3))
    print("Validation ROCAUC", round(roc_auc_score(y_test, y_test_prob.detach().numpy()), 3)) #detach()去掉梯度資訊，畫圖不需要梯度!!
    print("\n")

    #Plot the loss curve and ROC curve
    plt.figure(figsize = (20,5))
    plt.subplot(1,2,1)
    plt.plot(loss_list)
    plt.title("Loss across epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    plt.subplot(1,2,2)
    #Validation
    fpr_v, tpr_v, _ =  roc_curve(y_test, y_test_prob.detach().numpy())
    roc_auc_v = auc(fpr_v, tpr_v)

    #Training
    fpr_t, tpr_t = roc_curve(y_train, y_train_prob.detach().numpy())
    roc_auc_t = auc(fpr_t, tpr_t)

    plt.title("Receiver Operating Characteristic: Validation")
    plt.plot(fpr_v, tpr_v, "b", label = "Validation AUC = %0.2f" %roc_auc_v)
    plt.plot(fpr_t, tpr_t, "r", label = "Validation AUC = %0.2f" %roc_auc_t)
    plt.legend(loc = "lower right")
    plt.plot([0,1],[0,1], "r--")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

#Define Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2000)
        self.fc1 = nn.Linear(48,96)
        self.fc2 = nn.Linear(96,192)
        self.fc3 = nn.Linear(192,384)
        self.fc4 = nn.Linear(384,1)
        self.relu = nn.ReLU()
        self.final = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        y = self.final(out)
        return y

#Define Training Variable
num_epochs = 500
batch_size = 128
loss_function = nn.BCELoss()

#Hyperparameters
weight_decay = 0.0 #set to 0 ;no L2 regularizer; passed into the optimizer
lambda_L1 = 0.0 #set to 0 ; no L1 regularizer; manually added in Loss (train_network)

#Create a model instance
model = NeuralNetwork().to(device)
#Define Optimizer
adam_optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = weight_decay)

#Train model 
adam_loss = train_network(model, adam_optimizer, loss_function, num_epochs,
batch_size, x_train, y_train, lambda_L1 = 0.0)

#Evaluate Model 
evaluate_model(model, x_test, y_test, x_train, y_train, adam_loss)