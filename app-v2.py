import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_player import st_player
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def create_df():
    heart_disease = pd.read_csv("heart.csv")
    return heart_disease

def histogram(heart_disease):
    histogram = px.histogram(heart_disease, x="Age", color="HeartDisease").update_layout(yaxis_title = "Number of People")
    return histogram

def bubble_chart(heart_disease):
    data_frame = heart_disease[heart_disease["HeartDisease"] == 1]
    fig = px.scatter(data_frame, x="Age", y="MaxHR",
                 size="Cholesterol", color="Sex",
                     size_max=30)
    return fig

def histogram_2D_haveHeartDisease(heart_disease):
    df = heart_disease[heart_disease["HeartDisease"] == 1]
    fig = px.density_heatmap(df, x="Cholesterol", y="MaxHR", marginal_x="histogram", marginal_y="histogram")
    return fig

def histogram_2D_noHeartDisease(heart_disease):
    df = heart_disease[heart_disease["HeartDisease"] == 0]
    fig = px.density_heatmap(df, x="Cholesterol", y="MaxHR", marginal_x="histogram", marginal_y="histogram")
    return fig

def pie(heart_disease):
    has_heart_disease = heart_disease[heart_disease["HeartDisease"] == 1]
    counts_df = has_heart_disease.ST_Slope.value_counts()
    dict = {"Flat":460, "Up":395, "Down":63}
    fig = px.pie(counts_df, values= dict, names= dict.keys(), color_discrete_sequence=px.colors.sequential.Plasma)
    return fig

def pietwo(heart_disease):
    has_heart_disease = heart_disease[heart_disease["HeartDisease"] == 1]
    counts_df = has_heart_disease.ChestPainType.value_counts()
    dict = {"ASY":392, "NAP":72, "ATA":24, "TA":20}
    fig = px.pie(counts_df, values= dict, names= dict.keys(), color_discrete_sequence=px.colors.sequential.Plasma)
    return fig

def histogramfour(heart_disease):
    histogram = px.histogram(heart_disease, x='ExerciseAngina', y='HeartDisease').update_layout(xaxis_title = "Exercise Angina", yaxis_title = "Number of Heart Disease Patients")
    return histogram

def bar_graph(heart_disease):
    df = heart_disease[heart_disease["HeartDisease"] == 1]
    fig = px.histogram(df, x="ChestPainType", y="HeartDisease",
                 color='Sex', barmode='group',
                 title="Compare Male and Female Patients with Heart Disease to the Type of Chest Pain They Feel")
    return fig

def scatter_plot(heart_disease):
    df = heart_disease[heart_disease["HeartDisease"] == 1]
    fig = px.scatter(df, x="RestingBP", y="Age")
    return fig

def box_plot(heart_disease):
    df = heart_disease[heart_disease["HeartDisease"] == 1]
    fig = px.box(df, x="FastingBS", y="Age")
    return fig

# predict if they are prone to heart disease
def train_model(heart_disease_df):
    heart_disease = heart_disease_df.copy()
    heart_disease['Sex'] = (heart_disease['Sex'] == "M").astype('int')
    heart_disease['ExerciseAngina'] = (heart_disease['ExerciseAngina'] == "Y").astype('int')

    heart_disease["ChestPainType"].unique()
    cpt_nums = {'ATA':0, 'NAP':1, 'ASY':2, 'TA':3}

    words = heart_disease.ChestPainType.tolist()
    nums = [cpt_nums[k] for k in words]
    heart_disease['ChestPainType'] = nums

    heart_disease["RestingECG"].unique()
    ecg_nums = {'Normal':0, 'ST':1, 'LVH':2}

    ecg_words = heart_disease.RestingECG.tolist()
    ecg_nums = [ecg_nums[k] for k in ecg_words]
    heart_disease['RestingECG'] = ecg_nums

    heart_disease["ST_Slope"].unique()
    sts_nums = {'Up':0, 'Flat':1, 'Down':2}

    sts_words = heart_disease.ST_Slope.tolist()
    sts_nums = [sts_nums[k] for k in sts_words]
    heart_disease['ST_Slope'] = sts_nums

    heart_X = heart_disease.drop(['HeartDisease'], axis=1)
    heart_Y = heart_disease['HeartDisease']

    X_heart_train, X_heart_test, Y_heart_train, Y_heart_test = train_test_split(heart_X, heart_Y, test_size = 0.15)
    model = LogisticRegression()
    model.fit(X_heart_train, Y_heart_train)
    
    return model

heart_disease = create_df()

# streamlit layout
st.set_page_config(layout="wide")
st.title("Predicting Heart Disease")

st.write("If we want to predict heart disease we first have to know what it is. Here is a video from Mayo Clinic that quickly explains what heart disease is: ")

# embed a video
st_player("https://www.youtube.com/watch?v=Oqt9TgWcrxI")

st.header("Taking data from UC Irvine's medical research, we analyzed the factors relating to heart disease. The goal of this project is to find possible causes and symptoms of heart disease as preventative measures and maybe even a cure through data visualizations.")

st.plotly_chart(histogram(heart_disease))
st.caption("This histogram compares age and number of people afflicted with heart disease. This data set has many more people with heart disease versus those without. According to this chart, heart disease is most common in people who are in their mid 60s.")

# put the two histograms into columns
c1, c2 = st.columns((1,1))

with c1:
    c1.header("People with heart disease")
    st.plotly_chart(histogram_2D_haveHeartDisease(heart_disease))
#     with_heart_disease = st.plotly_chart(histogram_2D_haveHeartDisease(heart_disease))
#     c1.with_heart_disease
    c1.caption("This is a 2D histograms. In this case, cholestrol and maximum heart rate are being compared to one another. The color scheme on the left shows that the more yellow a block is, the higher the number of cases of heart disease are within those parameters of cholestrol and maximum heart rate. The more blue the block is, the opposite.")
    # st.plotly_chart(fig,height=800)

with c2:
    c2.header("People without heart disease")
    c2.plotly_chart(histogram_2D_noHeartDisease(heart_disease))
    c2.caption("Like the other 2D histogram, this chart displays similar information with one sole differentiating factor: that this graph shows information relating to people without heart disease.")


# bubble chart
st.plotly_chart(bubble_chart(heart_disease))
st.caption("Bubble charts display three variables, and in this case: age, maximum heart rate, and cholestrol levels. As a bonus, the bubbles are color coded to the sex of the patients. This chart shows that men are more likely to develop heart disease compared to women. Additionally, maximum heart rate steadily lowers in a linear fashion as age increases; this negative correlation indicates that older people's hearts cannot beat as quickly as when they were younger. Lower maximum heart rates have been linked to increased chances of heart disease, supporting the claim that the elderly are more likely to suffer from heart disease. The size of the bubbles refers to the cholestrol level of patients, and higher cholestrol levels increase the chance of heart disease.")

# put the pie charts into columns
c3, c4 = st.columns(2)
with c3:
    # pie charts
    c3.header("ST Slopes in Heart Disease Patients")
    c3.plotly_chart(pie(heart_disease))
    c3.caption("This pie chart shows the prevalance of different ST slopes in heart disease patients. The most prevalent type of ST slope is a flat slope, indicating that people with a flat ST slope could be at the most risk for heart disease.")

with c4:
    c4.header("Chest Pain in Heart Disease Patients")
    c4.plotly_chart(pietwo(heart_disease))
    c4.caption("This pie chart shows the prevalance of different types of chest pain in heart disease patients. Patients who are asymptomatic, meaning they experience no chest pain, make up the largest proportion.")

st.header("Exercise Angina in Heart Disease Patients")
st.plotly_chart(histogramfour(heart_disease))
st.caption("This histogram shows the proportion of heart disease patients who experience exercise angina. It is clear that the majority of people in this data set with heart disease experience exercise angina, meaning that it could be an early indicator of heart disease.")

st.header("Resting BP vs. Age in Heart Disease Patients")
st.plotly_chart(scatter_plot(heart_disease))
st.caption("This is a scatter plot that compares resting blood pressure and age in heart disease patients. High resting blood pressure has been linked to increased chances of heart disease, meaning resting blood pressure and heart disease have a positive correlation, much like age and heart disease. This chart shows that heart disease is most common in patients with resting blood pressures over 10 and in patients between the ages of 45 to 65.")

st.header("Age and Fasting Blood Sugar")
st.plotly_chart(box_plot(heart_disease))
st.caption("This is a box plot comparing age and fasting blood sugar. A fasting blood sugar of 1 is more likely to indicate heart disease, and in this case, patients between 53 and 62 are more likely to have heart disease.")

# bar graph
st.header("Bar Graph")
st.plotly_chart(bar_graph(heart_disease))
st.caption("This is a bar graph comparing male and female patients with heart disease to the type of chest pains they feel. Overall, the majority of the patients are asymptomatic and experience no chest pain.")

st.write("In conclusion, these charts allow for easy interpretation of data and provide a way to observe some of the symptoms and side effects of heart disease.")

model = train_model(heart_disease)
st.form("myform", clear_on_submit=False)
with st.form("Do you have heart disease?"):
    st.write("Fill out this form to find out if you are at risk for heart disease. ")
    age = st.number_input("Enter your age: ")
    sex_nums = {"Male":1, "Female":0}
    sex = st.selectbox(
     'Select your Sex:',
     ('Male', 'Female'))
    sex = sex_nums[sex]
    cpt_nums = {'ATA':0, 'NAP':1, 'ASY':2, 'TA':3}
    chest_pain_type = st.selectbox("Select your chest pain type: ", ("ASY", "NAP", "ATA", "TA"))
    chest_pain_type = cpt_nums[chest_pain_type]
    restingBP = st.number_input("Enter your resting blood pressure: ")
    cholesterol = st.number_input("Enter your cholesterol: ")
    fastingBS = st.selectbox('Select your fasting blood sugar:',('0', '1'))
    ecg_nums = {'Normal':0, 'ST':1, 'LVH':2}
    restingECG = st.selectbox('Select your resting ECG:',('Normal', 'ST', 'LVH'))
    restingECG = ecg_nums[restingECG]
    maxHR = st.number_input("Enter your maximum heart rate: ")
    old_peak = st.number_input("Enter your old peak: ")
    exercise_nums = {'Yes':0, 'No':1}
    exercise_angina = st.selectbox("Do you have exercise angina?", ("Yes", "No"))
    exercise_angina = exercise_nums[exercise_angina]
    sts_nums = {'Up':0, 'Flat':1, 'Down':2}
    stSlope = st.selectbox('Select your ST Slope: ', ("Up", "Flat", "Down"))
    stSlope = sts_nums[stSlope]
    submitted = st.form_submit_button("Submit")
    if submitted:
        prediction = model.predict([[age, sex, chest_pain_type, restingBP, cholesterol, fastingBS, restingECG, maxHR, old_peak, exercise_angina, stSlope]])
        if prediction[0] == 0:
            st.write("You are not likely to get heart disease!")
        elif prediction[0] == 1:
            st.write("You might get heart disease.")



