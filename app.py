import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data'
column_names = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long',
    'fM3Trans', 'fAlpha', 'fDist', 'class'
]
df = pd.read_csv(url, names=column_names)

# Map 'g' to 1 and 'h' to 0
df['class'] = df['class'].map({'g': 1, 'h': 0})

# Detect and remove outliers using IQR method
def detect_outliers(df, features):
    outliers = []
    for feature in features:
        Q1 = np.percentile(df[feature], 25)
        Q3 = np.percentile(df[feature], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outliers.extend(df[(df[feature] < Q1 - outlier_step) | (df[feature] > Q3 + outlier_step)].index)
    return outliers

features = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
outliers = detect_outliers(df, features)
df_no_outliers = df.drop(outliers).reset_index(drop=True)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']
X_no_outliers = df_no_outliers.drop('class', axis=1)
y_no_outliers = df_no_outliers['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_no_outliers, y_no_outliers, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Pre-trained models dictionary
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

# Streamlit app
st.title('Magic Gamma Telescope Classification')

# STAR Method Section
st.header('Project Overview')
st.markdown("""
**Situation**: Developed a machine learning model to classify gamma particles for a data science project.

**Task**: Preprocess data, perform EDA, and build an accurate model.

**Action**: Cleaned data, conducted EDA, optimized a RandomForestClassifier, and used TPOT for tuning.

**Result**: Achieved 87% accuracy with Random Forest Classifier Using TPOT to find best Parameters.
""")

# Display space for Kaggle, Github, & LinkedIn
st.subheader('Connect with me:')
st.markdown('[Kaggle](https://www.kaggle.com/fakelone)')
st.markdown('[GitHub](https://github.com/Abdullah-0-3)')
st.markdown('[LinkedIn](https://www.linkedin.com/in/muhammadabdullahabrar)')

# Display boxplots with outliers
st.subheader('Boxplots with Outliers')
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i, feature in enumerate(features):
    sns.boxplot(x=df[feature], ax=axs[i//5, i%5])
st.pyplot(fig)

# Display boxplots after removing outliers
st.subheader('Boxplots After Removing Outliers')
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i, feature in enumerate(features):
    sns.boxplot(x=df_no_outliers[feature], ax=axs[i//5, i%5])
st.pyplot(fig)

st.markdown('Used IQR method to remove outliers.')

# Sidebar for model selection and parameter tuning
st.sidebar.header('Model Selection and Parameter Tuning')

selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))

# Display heatmap
st.subheader('Heatmap of Correlation Matrix')
corr_matrix = df.corr()
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
st.plotly_chart(fig)

params = dict()
if selected_model == 'Random Forest':
    params['n_estimators'] = st.sidebar.slider('Number of Estimators', 10, 200, 100)
    params['max_depth'] = st.sidebar.slider('Max Depth', 1, 20, 10)
elif selected_model == 'SVM':
    params['C'] = st.sidebar.slider('C', 0.01, 10.0, 1.0)
    params['kernel'] = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
elif selected_model == 'K-Nearest Neighbors':
    params['n_neighbors'] = st.sidebar.slider('Number of Neighbors', 1, 20, 5)
elif selected_model == 'Logistic Regression':
    params['C'] = st.sidebar.slider('C', 0.01, 10.0, 1.0)
elif selected_model == 'Decision Tree':
    params['max_depth'] = st.sidebar.slider('Max Depth', 1, 20, 10)
elif selected_model == 'Naive Bayes':
    params['var_smoothing'] = st.sidebar.slider('Var Smoothing', 1e-10, 1e-9, 1e-9)

# Train and display classification report of the selected model
model = models[selected_model].set_params(**params)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

st.subheader('Classification Report')
st.text(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_labels = [str(i) for i in range(len(cm))]

# Create the plotly figure
fig = ff.create_annotated_heatmap(z=cm, x=cm_labels, y=cm_labels, 
                                  colorscale='Blues', showscale=True)

# Update layout for better readability
fig.update_layout(
    title="Confusion Matrix",
    xaxis_title="Predicted Label",
    yaxis_title="True Label"
)

# Display the confusion matrix
st.subheader('Confusion Matrix')
st.plotly_chart(fig)
