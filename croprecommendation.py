# %%
import pandas as pd   #imprting panda library for data analysis
import numpy as np    #importing numpy library for mathematical operations
import seaborn as sns #importing seaborn library for data visualization
import matplotlib.pyplot as plt  #matplotlib work like a MATLAB
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('Crop_recommendation.csv')
df

# %%
df.shape

# %%
df.size

# %%
df.head(10)

# %%
df.tail(10)

# %%
df.describe()

# %%
df.info()

# %%
df.isna().sum()

# %%
"""
# EDA
"""

# %%
N_wise=df.N.value_counts()  
N_wise                              

# %%
K_wise=df.K.value_counts()  
K_wise                              

# %%
P_wise=df.P.value_counts()  
P_wise                              

# %%
temperature_wise=df.temperature.value_counts()  
temperature_wise                              

# %%
humidity_wise=df.humidity.value_counts()  
humidity_wise 

# %%
ph_wise=df.ph.value_counts()  
ph_wise 

# %%
rainfall_wise=df.rainfall.value_counts()  
rainfall_wise 

# %%
#setting a grey background
sns.set(style="darkgrid")
#creating subplots with 10*10 figure size
fig,axs=plt.subplots(2,2,figsize=(10,10))
#ploting subplots with numerical variables
sns.histplot(data=df,x="P",kde=True,color="red",ax=axs[0,0])
sns.histplot(data=df,x="N",kde=True,color="olive",ax=axs[0,1])
sns.histplot(data=df,x="K",kde=True,color="gold",ax=axs[1,0])
sns.histplot(data=df,x="ph",kde=True,color="black",ax=axs[1,1])
plt.show()

# %%
fig,axs=plt.subplots(1,2,figsize=(10,5))
#ploting subplots with numerical variables
sns.histplot(data=df,x="temperature",kde=True,color="red",ax=axs[0])  #histogram for low column
sns.histplot(data=df,x="humidity",kde=True,color="Navy",ax=axs[1])   #histogram for Total_Sales
plt.show()

# %%
#setting a grey background
sns.set(style="darkgrid")
#creating subplots with 10*10 figure size
fig,axs=plt.subplots(2,2,figsize=(10,10))
#ploting subplots with variables
sns.boxplot(data=df,x="N",color="red",ax=axs[0,0])
sns.boxplot(data=df,x="P",color="olive",ax=axs[0,1])
sns.boxplot(data=df,x="K",color="gold",ax=axs[1,0])
sns.boxplot(data=df,x="ph",color="black",ax=axs[1,1])
plt.show()

# %%
#setting a grey background
sns.set(style="darkgrid")
#creating subplots with 10*10 figure size
fig,axs=plt.subplots(1,2,figsize=(10,5))
#ploting subplots with variables
sns.boxplot(data=df,x="temperature",color="red",ax=axs[0])
sns.boxplot(data=df,x="humidity",color="olive",ax=axs[1])
plt.show()

# %%
import pingouin as pg
pg.qqplot(df['P'],dist='norm')

# %%
#QQ plot for Salary
import pingouin as pg
pg.qqplot(df['N'],dist='norm')

# %%
#QQ plot for Salary
import pingouin as pg
pg.qqplot(df['K'],dist='norm')

# %%
#QQ plot for Salary
import pingouin as pg
pg.qqplot(df['ph'],dist='norm')

# %%
#QQ plot for Salary
import pingouin as pg
pg.qqplot(df['temperature'],dist='norm')

# %%
#QQ plot for Salary
import pingouin as pg
pg.qqplot(df['rainfall'],dist='norm')

# %%
#QQ plot for Salary
import pingouin as pg
pg.qqplot(df['humidity'],dist='norm')

# %%
"""
# finding out the correlation between variables using spearman rank correlation
"""

# %%
df_numeric=df.select_dtypes(include='number')

# %%
#spearman rank correlation heapmap
df_numeric.corr(method="spearman") #selecting the method as a spearman
plt.figure(figsize=(8,6)) #setting the figuresize

heatmap=sns.heatmap(df_numeric.corr(method='spearman').round(3),vmin=-1,
                                vmax=1,annot=True) # annot=True means writting the data value in each cell
font2={'family':'serif','color':'green','size':20}
plt.title("Spearman Rank Correlation",font2)
plt.show() #displayed heatmap
                                

# %%
plt.figure(figsize=(9,8))  #size of the figure
sns.boxplot(data=df,x="N",y="label", color="Red")  
plt.title("label vs Nitrogen")  #plot title

# %%
plt.figure(figsize=(9,8))  #size of the figure
sns.boxplot(data=df,x="K",y="label", color="green")  
plt.title("label vs Potassium")  #plot title

# %%
plt.figure(figsize=(9,8))  #size of the figure
sns.boxplot(data=df,x="P",y="label", color="blue")  
plt.title("label vs Phosphorus")  #plot title

# %%
plt.figure(figsize=(9,8))  #size of the figure
sns.boxplot(data=df,x="ph",y="label", color="pink")  
plt.title("label vs ph")  #plot title

# %%
plt.figure(figsize=(9,8))  #size of the figure
sns.boxplot(data=df,x="temperature",y="label", color="olive")  
plt.title("label vs temperature")  #plot title

# %%
plt.figure(figsize=(9,8))  #size of the figure
sns.boxplot(data=df,x="humidity",y="label", color="lightgreen")  
plt.title("label vs humidity")  #plot title

# %%
plt.figure(figsize=(9,8))  #size of the figure
sns.boxplot(data=df,x="rainfall",y="label", color="lime")  
plt.title("label vs rainfall")  #plot title

# %%
plt.figure(figsize=(9,7)) #size of the figure
figure=df.boxplot(column="P")  

# %%
plt.figure(figsize=(9,7)) #size of the figure
figure=df.boxplot(column="K")  

# %%
plt.figure(figsize=(9,7)) #size of the figure
figure=df.boxplot(column="N")  

# %%
plt.figure(figsize=(9,7)) #size of the figure
figure=df.boxplot(column="ph")  

# %%
plt.figure(figsize=(9,7)) #size of the figure
figure=df.boxplot(column="temperature")  

# %%
plt.figure(figsize=(9,7)) #size of the figure
figure=df.boxplot(column="humidity")  

# %%
plt.figure(figsize=(9,7)) #size of the figure
figure=df.boxplot(column="rainfall")  

# %%
# looking to the percentage of correlation
df_numeric.corr(method='spearman')*100

# %%
"""
# Model Building
"""

# %%
x=df.drop(['label'],axis=1) #dropping highly co-related features
x #new datas set

# %%
y=df['label'] 
y 

# %%
#split x and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# %%
x_train

# %%
y_train

# %%
x_test

# %%
y_test

# %%
#data normalization with sklearn
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
#fit scalar on training data
x_train=sc.fit_transform(x_train)

#transform testing data
x_test=sc.transform(x_test)

# %%
x_train

# %%
x_test

# %%
"""
# Machine learning technique
"""

# %%
from sklearn.linear_model import LogisticRegression

# %%
model=LogisticRegression()
model.fit(x_train,y_train)

# %%
y_pred1=model.predict(x_test)

# %%
from sklearn.metrics import accuracy_score
logistic_reg_acc=accuracy_score(y_test,y_pred1)
print("logistic accuracy is "+str(logistic_reg_acc))

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
model2=DecisionTreeClassifier()
model2.fit(x_train,y_train)
y_pred2=model2.predict(x_test)
decision_acc=accuracy_score(y_test,y_pred2)
print("Decision tree accuracy is  "+ str(decision_acc))

# %%
from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier()
model3.fit(x_train,y_train)
y_pred3=model3.predict(x_test)
random_acc=accuracy_score(y_test,y_pred3)
print("Random Forest accuracy is  "+ str(decision_acc))

# %%
import pickle
from sklearn.ensemble import RandomForestClassifier
# example: train a model (RandomForestClassifier as an example)
model3=RandomForestClassifier()
model3.fit(x_train,y_train)
# save the trained mode to a file usoing pickle
with open('crop.pkl','wb') as file:
    pickle.dump(model3,file)
print("model saved successfull")

# %%
with open('sccrop.pkl','wb') as scaler_file:
    pickle.dump(sc,scaler_file)

# %%
import os

streamlit_code = """
import streamlit as st
import pickle
import numpy as np

# Load the trained RandomForest model and scaler
with open('crop.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('sccrop.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the web app
st.title('Crop Recommendation App')

# Input fields
N = st.number_input('Nitrogen', min_value=0.0, value=0.0, step=1.0)
P = st.number_input('Phosphorus', min_value=0.0, value=0.0, step=1.0)
K = st.number_input('Potassium', min_value=0.0, value=30.0, step=1.0)
temperature = st.number_input('Temperature', min_value=0.0, value=25.0, step=0.1)
humidity = st.number_input('Humidity', min_value=0.0, value=50.0, step=0.1)
ph = st.number_input('PH', min_value=0.0, value=6.5, step=0.1)
rainfall = st.number_input('Rainfall', min_value=0.0, value=100.0, step=1.0)

# Button to trigger prediction
if st.button('Predict Crop'):
    # Prepare the feature vector
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=np.float64)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Crop Recommendation
    predicted_crop = model.predict(features_scaled)

    # Display the prediction
    st.write(f'Predicted Crop: {predicted_crop[0]}')
"""

file_path = os.path.join(os.getcwd(), 'crop1app.py')

try:
    with open(file_path, 'w') as file:
        file.write(streamlit_code)
    print(f"File '{file_path}' has been saved.")
except Exception as e:
    print(f"Error saving file: {e}")


# %%
