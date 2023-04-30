#!/usr/bin/env python
# coding: utf-8

# # Data Preparation and Model Training
# 
# In this notebook, we load the [Iris plants dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris), extract features, and train a Random Forest classifier.
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import mlflow
from sklearn import datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ## Data Preparation
# 
# We use the [Iris plants dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris) to predict the class of Iris flowers.
# It contains 150 samples (50 for each instance), and the following 4 numeric, predictive attributes.
# 
# - sepal length in cm
# - sepal width in cm
# - petal length in cm
# - petal width in cm
# 
# The class names are Iris-Setosa (`0`), Iris-Versicolour (`1`), and Iris-Virginica (`2`).
# 

# In[2]:


import os
homedir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


# In[3]:


berlin_gpd = gpd.read_file(homedir+"\\frontend\crime_pp_cafes.gpkg")
X = berlin_gpd[["Robbery", "bp_weight","restaurant"]].to_numpy()
y = berlin_gpd["Land_Value"].to_numpy()


# In[8]:


X.shape, y.shape


# In[2]:


X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape


# In[3]:





# In[3]:


X.min(axis=0), X.max(axis=0)


# In[4]:


col_mean = np.nanmean(X, axis=0)


# In[5]:


inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])


# Let's plot the data to gain some insights.
# 

# In[12]:


fig = plt.figure(figsize=(12, 6))

# Petal Length and Width
ax = fig.add_subplot(131)
ax.set_title('Robbery and bp_weight')
ax.set_xlabel('Robbery')
ax.set_ylabel('bp_weight')
ax.grid(True, linestyle='-', color='0.5')
ax.scatter(X[:, 0], X[:, 1], s=32, c=y, marker='o')

# Sepal Length and Width
ax = fig.add_subplot(132)
ax.set_title("bp_weight and restaurant")
ax.set_xlabel('bp_weight')
ax.set_ylabel('restaurant')
ax.grid(True, linestyle='-', color='0.5')
ax.scatter(X[:, 1], X[:, 2], s=32, c=y, marker='s')

# Sepal Length and Width
ax = fig.add_subplot(133)
ax.set_title("Robbery and restaurant")
ax.set_xlabel('Robbery')
ax.set_ylabel('restaurant')
ax.grid(True, linestyle='-', color='0.5')
ax.scatter(X[:, 0], X[:, 2], s=32, c=y, marker='s')


# **Task:** Split the data into train and test sets. Use $60%$ samples for training.
# 

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)
X_train.shape, X_test.shape


# In[ ]:


col_mean = np.nanmean(X, axis=0)


# ## ML Training
# 
# ### Model Training and Logging
# 
# Let's use random forest regression to predict house prices.
# Therefore, we want to find a reasonable maximum depth for the single decision trees.
# We keep track of several tries and their mean squared error using MLflow.
# 
# Please note that we ignore best practices like cross validation, feature selection and randomised parameter search for demonstration purposes.
# 
# **Task:** Setup the pipeline factory with a random forest classifier using `max_depth=3` and a specified number of estimators. You might add additional pipeline steps.
# 

# In[7]:


def create_pipeline(n_estimators: int) -> Pipeline:
    return Pipeline(
        steps=[('scalar', StandardScaler()),
               ('model', KernelRidge(alpha=1.0))])


# **Task:** Choose reasonable hyperparameters to try, and execute the training process. Log the accuracy and according parameters. You might add further metrics.
# 

# In[8]:


n_alphas_to_try = [0.5, 1, 1.5]
REGISTERED_MODEL_NAME = "krr"

for alpha in n_alphas_to_try:
    with mlflow.start_run():
        # build a pipeline with a ridge regression model
        model_pipeline = create_pipeline(n_estimators=alpha)
        model_pipeline.fit(X_train, y_train)

        # calculaye metrics using the test data
        y_pred = model_pipeline.predict(X=X_test)
        r2 = r2_score(y_true=y_test, y_pred=y_pred)

        # log parameters, metrics and the model
        mlflow.log_param(key="alpha", value=alpha)
        mlflow.log_metric(key="R2", value=r2)
        mlflow.sklearn.log_model(
            sk_model=model_pipeline, artifact_path="dlr", registered_model_name=REGISTERED_MODEL_NAME)

        print(
            f"Model saved in run {mlflow.active_run().info.run_uuid}. R2={r2}")


# ### Assessing the Runs in the MLflow Web-UI
# 
# **Task:** Inspect the training runs with their parameters and metrics with MLflow's web-UI.
# Store the best model in the model registry, and stage it for production (either in the web UI, or using the Python interface).
# 
# Just execute this cell and visit the uri in your web browser.
# Terminate this cell or the notebook to stop the server.
# 

# In[9]:


get_ipython().system('mlflow ui -p 5002')


# In[9]:


client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME, version=3, stage="Production"
)


# ## Model Deployment
# 
# Let's use our production model to generate a Docker image for the model endpoint.
# 
# Please note that the first run of this cell might take some minutes.
# In the meantime, you can start with the next tasks.
# 

# In[10]:


mlflow.models.build_docker(
    model_uri=f"models:/{REGISTERED_MODEL_NAME}/Production", name="iris_model_api", env_manager="conda", install_mlflow=True)


# In[ ]:




