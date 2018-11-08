import numpy as np
import pandas as pd
import sklearn as sp
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score


# import data from csv
df = pd.read_csv('C:/Users/Kamal/Downloads/Assignment Dataset/train_u6lujuX_CVtuZ9i.csv', header=0);

# Feature Preprocessing

# Covert Category values to numeric
df.Gender = pd.Categorical(df.Gender)
df['Gender_Mod'] = df.Gender.cat.codes

df.Married = pd.Categorical(df.Married)
df['Married_Mod'] = df.Married.cat.codes

df.Loan_ID = pd.Categorical(df.Loan_ID)
df['Loan_ID_Mod'] = df.Loan_ID.cat.codes

df.Education = pd.Categorical(df.Education)
df['Education_Mod'] = df.Education.cat.codes

df.Self_Employed = pd.Categorical(df.Self_Employed)
df['Self_Employed_Mod'] = df.Self_Employed.cat.codes

df.Property_Area = pd.Categorical(df.Property_Area)
df['Property_Area_Mod'] = df.Property_Area.cat.codes

df.Loan_Status = pd.Categorical(df.Loan_Status)
df['Loan_Status_Mod'] = df.Loan_Status.cat.codes

# Feature Generation
for index, row in df.iterrows():
   if row["Dependents"] == 0:
       df.loc[index , 'PerIncome']  = float(row["ApplicantIncome"]) + float(row["CoapplicantIncome"])

   else:
       df.loc[index, 'PerIncome'] = (float(row["ApplicantIncome"]) + float(row["CoapplicantIncome"]))/float(row["Dependents"])+1


df_mod = df[['Loan_ID_Mod','Gender_Mod','Dependents','Education_Mod','Self_Employed_Mod','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area_Mod','Loan_Status_Mod','PerIncome']]

x_mod = df[['Loan_ID_Mod','Gender_Mod','Dependents','Education_Mod','Self_Employed_Mod','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area_Mod','PerIncome']]
y_mod = df[['Loan_Status_Mod']]

# create correlation mattrix for feature selection
# This code has been commented as we are ploting graphs at later stage of code and this code can impact its look  and feel.
# fig = plt.figure(figsize=(15,8))
# ax1 = fig.add_subplot(111)
# plt.imshow(x_mod.corr(), cmap='hot', interpolation='nearest')
# plt.colorbar()
# labels = x_mod.columns.tolist()
# ax1.set_xticks(np.arange(len(labels)))
# ax1.set_yticks(np.arange(len(labels)))
# ax1.set_xticklabels(labels,rotation=90, fontsize=10)
# ax1.set_yticklabels(labels,fontsize=10)
# plt.show()

# alter dataset after correlation mattrix
x_mod = df[['Loan_ID_Mod','Gender_Mod','Dependents','Education_Mod','Self_Employed_Mod','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area_Mod','PerIncome']]

# print ('Count of test y is : ', y_mod.shape[1])
# print ('Count of test y is : ', x_mod.shape[1])

y_mod = np.array(y_mod).reshape((len(y_mod)), y_mod.shape[1])
x_mod = np.array(x_mod).reshape((len(x_mod)), x_mod.shape[1])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_mod,y_mod, test_size=0.05, random_state=101)

# # Min Max Normalization of the data
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler1.fit(X_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
X_train = scaler1.transform(X_train)
X_test = scaler1.transform(X_test)

# MLPClassifier uses cross-entropy loss function by default
# Tuning of the model is handled in below code
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50, 50),activation="tanh", max_iter=500, learning_rate="constant", learning_rate_init=0.0001, verbose=False)
# mlp.loss('MSE')
mlp.fit(X_train, Y_train)

# Accuracy using 8-fold cross validation
scores = cross_val_score(mlp, X_train, Y_train, cv=8,scoring='accuracy')
print ("Accuracy of the classifier is" , scores.mean())

# Recall using 8-fold cross validation
score = cross_val_score(mlp, X_train, Y_train, cv=8,scoring='recall')
print ("Recall of the classifier is" , score.mean())


# print(mlp1.loss_)
# print(mlp1.classes_)
# print(mlp1.n_iter_)
# print(mlp1.n_layers_)
# print(mlp1.n_outputs_)
# print(mlp1.out_activation_)
# print(mlp1.loss)

# Plot for loss function versus number of iterations
loss_values= mlp.loss_curve_
plt.xlabel('N iteration')
plt.ylabel('loss values')
plt.plot(loss_values)
plt.show()

