from sklearn.model_selection import train_test_split
import os
import shutil

X_folder = "./end/train/labelled/image"
y_folder = "./end/train/labelled/label"
X_train_dst = "./dataset/train"
X_test_dst = "./dataset/val"
y_train_dst = "./dataset/trainannot"
y_test_dst = "./dataset/valannot"

X = os.listdir(X_folder)
y = os.listdir(y_folder)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/9, random_state=42)

for i in range(len(X_train)):
    X_train_src = os.path.join(X_folder, X_train[i])
    y_train_src = os.path.join(y_folder, y_train[i])
    shutil.copy(X_train_src, X_train_dst)
    shutil.copy(y_train_src, y_train_dst)
    
for j in range(len(X_test)):
    X_test_src = os.path.join(X_folder, X_test[j])
    y_test_src = os.path.join(y_folder, y_test[j])
    shutil.copy(X_test_src, X_test_dst)
    shutil.copy(y_test_src, y_test_dst)
    

