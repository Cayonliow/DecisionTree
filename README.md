# Data Mining Homework2 - Decision Tree
###### tags: `datamining`
---

## Step 1: Design a set of rules to classify data
* The problem designed whether he/she is charming

* There are 5 features for the problem
    * `Height`
    * `Weight`
    * `Kindness` whether is kindness
    * `IQ`
    * `Smoker` - whether a smoker

* The number of data = 10000

### Absolutely right rule
Height (>170cm), Weight(>50kg, <100kg), Kindness(yes), IQ(>80), Smoker(no) --->>> Charming

step:
* Write the features into `csv` format
* Load the data from the generated `.csv` file 
    *    `dataset.csv` and `label.csv` are used instead of `dataset_latest.csv` and `label_latest.csv`
`dataset_latest.csv` and `label_latest.csv` above are showing how the data being generated 

#### Cross Validation
* training data = 67% 
* testing data = 33%

## Step 2: Use the data generated in Step 1 to construct the classification model 
* Desicion Tree is used
    * Accuracy = 99% 

### Visualization with graphviz

![](https://i.imgur.com/PxPXkfd.png)

## Step 3: Compare the rules in the decision tree from Step 2 and the rules used to generate the  ‘absolutely right’ data 

### The absolutely right rule
* Height (>170cm) and Weight(>50kg, <100kg) and Kindness(yes) and IQ(>80) and Smoker(no) --->>> Charming(yes)

### The rule generated from the Decision Tree
* Height(<160.15) and Weight(<50.5) and Kindness(no) --->>> Charming(no)

### Discussion
The rule generated from the Decision Tree are similiar to The absolutely right rule

Since the data is generated randomly, there are affecting the final result 

In the result of Decision tree, the level of IQ and whether is a smoker are not included in the consideration of being Charming

## Step 4: Discuss anything you can 

### Random Forest
Accuracy of using Random Forest is 99.8% which is slightly lower than that of Decision Tree
```python=
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf_rf.fit(X_train, Y_train)
y_rf = clf_rf.predict(X_test)
print('Accuracy = ', accuracy_score(y_rf, Y_test))
```

### SVC
Accuracy of using  SVC is 96.5% which is slightly lower than that of Decision Tree
```python=
classifier_svc = SVC(gamma='auto')
classifier_svc.fit(X_train, Y_train)
y_pred = classifier_svc.predict(X_test)
print('Accuracy = ', accuracy_score(y_pred, Y_test))
```

### PCA to 2 dimension of features
the accuracy proves the PCA doesn't get much more help on Decision Tree
```python=
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train_pca, Y_train)
y = clf.predict(X_test_pca)
print('Accuracy = ', accuracy_score(y, Y_test))
```

![](https://i.imgur.com/iEfI2Ec.png)

### Reference
* [Decision Tree and Random Forest](https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda)
* [A brief look at sklearn.tree.DecisionTreeClassifier](https://hackernoon.com/a-brief-look-at-sklearn-tree-decisiontreeclassifier-c2ee262eab9a)


