import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras import Sequential
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# Load the dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop rows with missing values
data.dropna(axis=0, inplace=True)  # axis=0 means dropping rows
print(f"After dropping null values, the shape of the Dataset is {data.shape}")

# Check for remaining missing values
print(f"After dropping null values, null values of the Dataset:")
print(data.isna().sum().to_frame().T)

# Convert data types
data["age"] = data["age"].astype("int")

# Remove rows with specific values
data = data[data["gender"] != "Other"]

# Replace values for better readability
data["hypertension"] = data["hypertension"].replace({0: "No", 1: "Yes"})
data["heart_disease"] = data["heart_disease"].replace({0: "No", 1: "Yes"})
data["stroke"] = data["stroke"].replace({0: "No", 1: "Yes"})
data["ever_married"] = data["ever_married"].replace({"No": "Unmarried", "Yes": "Married"})
data["work_type"] = data["work_type"].replace({
    "Self-employed": "Self Employed",
    "children": "Children",
    "Govt_job": "Government Job",
    "Private": "Private Job",
    "Never_worked": "Unemployed"
})
data["smoking_status"] = data["smoking_status"].replace({
    "never smoked": "Never Smoked",
    "formerly smoked": "Formerly Smoked",
    "smokes": "Smokes"
})

# Rename columns for consistency
data.rename(columns={
    "gender": "Gender",
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "ever_married": "Marital Status",
    "work_type": "Occupation Type",
    "Residence_type": "Residence Type",
    "avg_glucose_level": "Average Glucose Level",
    "bmi": "BMI",
    "smoking_status": "Smoking Status",
    "stroke": "Stroke"
}, inplace=True)

# Rearrange column order
data = data[[
    "Age", "Gender", "Marital Status", "BMI", "Occupation Type",
    "Residence Type", "Smoking Status", "Hypertension",
    "Heart Disease", "Average Glucose Level", "Stroke"
]]

# Display a preview of the final dataset
print("After preprocessing, let's have a glimpse of the final dataset:")
print(data.head())

# Display summary statistics for numerical columns
print("After preprocessing, let's have a look at the summary of the dataset:")
print(data.describe().T)

# Display summary statistics for categorical columns
print("Summary of categorical columns:")
print(data.describe(include="object").T)

# Separate features and target variable
x = data.drop(["Stroke"], axis=1)  # Features
y = data["Stroke"]  # Target variable

feature_names = x.columns

x = pd.get_dummies(x, columns=x.select_dtypes(include='object').columns)
sc = StandardScaler()
x = sc.fit_transform(x)

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Resample the data using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print(f"Original training data shape: {x_train.shape}, {y_train.shape}")
print(f"Resampled training data shape: {x_train_smote.shape}, {y_train_smote.shape}")

# Model 1: Logistic Regression with class weights
lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train_smote, y_train_smote)
lr_pred = lr.predict(x_test)
lr_conf = confusion_matrix(y_test, lr_pred)
lr_report = classification_report(y_test, lr_pred)
lr_acc = round(accuracy_score(y_test, lr_pred) * 100, 2)
print(f"Logistic Regression Confusion Matrix:\n{lr_conf}")
print(f"Logistic Regression Classification Report:\n{lr_report}")
print(f"Logistic Regression Accuracy: {lr_acc}%")

# Model 2: Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train_smote, y_train_smote)
gnb_pred = gnb.predict(x_test)
gnb_conf = confusion_matrix(y_test, gnb_pred)
gnb_report = classification_report(y_test, gnb_pred)
gnb_acc = round(accuracy_score(y_test, gnb_pred) * 100, 2)
print(f"Gaussian Naive Bayes Confusion Matrix:\n{gnb_conf}")
print(f"Gaussian Naive Bayes Classification Report:\n{gnb_report}")
print(f"Gaussian Naive Bayes Accuracy: {gnb_acc}%")

# Model 3: Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(x_train_smote, y_train_smote)
bnb_pred = bnb.predict(x_test)
bnb_conf = confusion_matrix(y_test, bnb_pred)
bnb_report = classification_report(y_test, bnb_pred)
bnb_acc = round(accuracy_score(y_test, bnb_pred) * 100, 2)
print(f"Bernoulli Naive Bayes Confusion Matrix:\n{bnb_conf}")
print(f"Bernoulli Naive Bayes Classification Report:\n{bnb_report}")
print(f"Bernoulli Naive Bayes Accuracy: {bnb_acc}%")

# Model 4: Support Vector Machine with class weights
svm = SVC(C=100, gamma=0.002, class_weight='balanced')
svm.fit(x_train_smote, y_train_smote)
svm_pred = svm.predict(x_test)
svm_conf = confusion_matrix(y_test, svm_pred)
svm_report = classification_report(y_test, svm_pred)
svm_acc = round(accuracy_score(y_test, svm_pred) * 100, 2)
print(f"SVM Confusion Matrix:\n{svm_conf}")
print(f"SVM Classification Report:\n{svm_report}")
print(f"SVM Accuracy: {svm_acc}%")

# Model 5: Random Forest with class weights
rfg = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rfg.fit(x_train_smote, y_train_smote)
rfg_pred = rfg.predict(x_test)
rfg_conf = confusion_matrix(y_test, rfg_pred)
rfg_report = classification_report(y_test, rfg_pred)
rfg_acc = round(accuracy_score(y_test, rfg_pred) * 100, 2)
print(f"Random Forest Confusion Matrix:\n{rfg_conf}")
print(f"Random Forest Classification Report:\n{rfg_report}")
print(f"Random Forest Accuracy: {rfg_acc}%")

# Model 6: K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train_smote, y_train_smote)
knn_pred = knn.predict(x_test)
knn_conf = confusion_matrix(y_test, knn_pred)
knn_report = classification_report(y_test, knn_pred)
knn_acc = round(accuracy_score(y_test, knn_pred) * 100, 2)
print(f"KNN Confusion Matrix:\n{knn_conf}")
print(f"KNN Classification Report:\n{knn_report}")
print(f"KNN Accuracy: {knn_acc}%")

# Model 7: Neural Network
x = data.drop(["Stroke"],axis =1)
x = pd.get_dummies(x, columns=x.select_dtypes(include='object').columns)

sc = StandardScaler()
x = sc.fit_transform(x)
y=data["Stroke"].map({"No": 0, "Yes": 1}).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = sc.fit_transform(x_train)
smote = SMOTE(sampling_strategy='auto', random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
x_test = sc.transform(x_test)

regularization_parameter = 0.003
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

neural_model = Sequential([tf.keras.layers.Dense(units=32, input_dim=(x_train.shape[-1]), activation="relu", kernel_regularizer = regularizers.l1(regularization_parameter)),
                    tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer = regularizers.l1(regularization_parameter)),
                    tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer = regularizers.l1(regularization_parameter)),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(units=16,activation="relu", kernel_regularizer = regularizers.l1(regularization_parameter)),
                    tf.keras.layers.Dense(units=1, activation="sigmoid")
                    ])

print(neural_model.summary())
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("accuracy") == 1.0):
            print("\nAccuracy is 100% so canceling training!")
            self.model.stop_training = True

callbacks = myCallback()


neural_model.compile(optimizer = Adam(learning_rate = 0.001),
                     loss = "binary_crossentropy",
                     metrics = ["accuracy"])

history = neural_model.fit(x_train, y_train,
                           epochs = 150,
                           verbose = 1,
                           batch_size = 64,
                           validation_data = (x_test, y_test),
                           callbacks = early_stopping)


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc)) # number of epochs

plt.figure(figsize=(20, 12))
plt.subplots_adjust(hspace=1.2)
plt.subplot(2,1,1)
plt.tight_layout(pad=8.0)

# Set background color to white with grid for the plot
sns.set_style("whitegrid")

# Define the colors for the different curves
palette = ["#764a23", "#f7941d", "#6c9a76", "#f25a29", "#cc4b57"]

# Plot the curves
plt.figure(figsize=(20, 12))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2, 1, 1)
plt.tight_layout(pad=8.0)

# Plot accuracy curves with grid
plt.plot(epochs, acc, color=palette[0], label="Training Accuracy")
plt.plot(epochs, val_acc, color=palette[1], label="Validation Accuracy")
plt.yscale("linear")
plt.title("\nTraining and validation accuracy", fontsize=15)
plt.xlabel("\nEpoch", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.legend(edgecolor="black")

# Add grid to the plot
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(epochs, loss, color=palette[2], label="Training Loss")
plt.plot(epochs, val_loss, color=palette[3], label="Validation Loss")
plt.title("Training and validation loss\n", fontsize=15)
plt.xlabel("\nEpoch", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(edgecolor="black")

# Add grid to the plot
plt.grid(True)

# Ensure a clean look with no unnecessary lines
sns.despine(left=True, bottom=True)

# Display the plot
plt.show()

y_pred = (neural_model.predict(x_test) > 0.5).astype("int32")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Stroke", "Stroke"]))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix (optional)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
