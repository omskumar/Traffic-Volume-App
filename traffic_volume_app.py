# Import libraries
import streamlit as st
import pandas as pd
import pickle

#title
st.title('Traffic Volume Prediction: A Machine Learning App') 
# Display the image
st.image('traffic_image.gif', width=700)
st.subheader("Utilize my advanced Machine Learning application to predict traffic volume") 
st.write("Use the following form to get started")

# load decision tree pickle file
dt_pickle = open('dt_traffic_volume.pickle', 'rb') 
dt_model = pickle.load(dt_pickle) 
dt_pickle.close()

# load random forest pickle file
rf_pickle = open('rf_traffic_volume.pickle', 'rb') 
rf_model = pickle.load(rf_pickle) 
rf_pickle.close()

# load AdaBoost pickle file
ab_pickle = open('ab_traffic_volume.pickle', 'rb') 
ab_model = pickle.load(ab_pickle) 
ab_pickle.close()

# load XGBoost pickle file
xgb_pickle = open('xgb_traffic_volume.pickle', 'rb') 
xgb_model = pickle.load(xgb_pickle) 
xgb_pickle.close()

#pre-processing for user form
df1 = pd.read_csv('Traffic_Volume.csv')
df1['month'] = pd.to_datetime(df1['date_time']).dt.month_name()
df1['weekday'] = pd.to_datetime(df1['date_time']).dt.day_name()
df1['hour'] = pd.to_datetime(df1['date_time']).dt.hour
df1['holiday'] = df1['holiday'].fillna("None")
df2 = df1.drop(columns = ['traffic_volume','date_time','weather_description'])
#user form
with st.form('user_inputs'): 
  holiday = st.selectbox('Choose whether today is a designated holiday or not', options= df2['holiday'].unique())
  temp = st.number_input('Average temperature in Kelvin') 
  rain_1h = st.number_input('Amount in mm of rain that occured in the hour')
  snow_1h = st.number_input('Amount in mm of snow that occured in the hour')
  clouds_all = st.number_input('Percentage of cloud cover')
  weather_main = st.selectbox('Choose the current weather', options= df2['weather_main'].unique())
  month = st.selectbox('Choose month', options= ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
  weekday = st.selectbox('Choose day of the week', options= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
  hour = st.selectbox('Choose hour', options=range(0,24))
  ml_model = st.selectbox('Select Machine Learning Model for Prediction', options = ['Decision Tree', 'Random Forest', 'AdaBoost', 'XGBoost'],
                          placeholder = 'Choose an option') 
  #call dataframe that will be color coded
  ml_analysis_df = pd.read_csv('ml_analysis.csv')
  ml_analysis_df = ml_analysis_df[['ML Model', 'R2', 'RMSE']]
  #color coding
  def color_coding(row):
    if row['ML Model'] == "XGBoost":
        return ['background-color: lime'] * len(row)
    elif row['ML Model'] == "AdaBoost":
        return ['background-color: orange'] * len(row)
    else:
        return ['background-color:'] * len(row)
  st.write("These ML models exhibited the following predictive performance on the test dataset.")
  st.dataframe(ml_analysis_df.style.apply(color_coding, axis=1))
  st.form_submit_button() 

#get user inputted data into the correct form for the model
#pre-process df
df = pd.read_csv('Traffic_Volume.csv')
df['month'] = pd.to_datetime(df['date_time']).dt.month_name()
df['weekday'] = pd.to_datetime(df['date_time']).dt.day_name()
df['hour'] = pd.to_datetime(df['date_time']).dt.hour
df['holiday'] = df['holiday'].fillna("None")
encode_df = df.copy()
#keep only necessary columns
encode_df = encode_df.drop(columns = ['traffic_volume','date_time','weather_description'])
# add the user inputs onto the training data
encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]
# get dummies for cateforigcal columns
cat_var = ['holiday', 'weather_main', 'hour', 'month', 'weekday']
encode_dummy_df = pd.get_dummies(encode_df, columns = cat_var)
# keep only the user inputed data
user_encoded_df = encode_dummy_df.tail(1)

#run user specified models
if ml_model == 'Decision Tree':
    #decision tree prediction
    new_prediction_dt = dt_model.predict(user_encoded_df)
    new_prediction_dt = new_prediction_dt.astype(int)
    #prediction
    st.write("Decision Tree Traffic Prediction: {}".format(*new_prediction_dt))
    st.subheader("Plot of Decision Tree Feature Importance:")
    #feature importance
    st.image('dt_feature_imp.svg')
elif ml_model == 'Random Forest':
    #random forest prediction
    new_prediction_rf = rf_model.predict(user_encoded_df)
    new_prediction_rf = new_prediction_rf.astype(int)
    st.write("Random Forest Traffic Prediction: {}".format(*new_prediction_rf))
    st.subheader("Plot of Random Forest Feature Importance:")
    #feature importance
    st.image('rf_feature_imp.svg')
elif ml_model == 'AdaBoost':
    #adaboost prediction
    new_prediction_ab = ab_model.predict(user_encoded_df)
    new_prediction_ab = new_prediction_ab.astype(int)
    st.write("AdaBoost Traffic Prediction: {}".format(*new_prediction_ab))
    st.subheader("Plot of AdaBoost Feature Importance:")
    #feature importance
    st.image('ab_feature_imp.svg')
elif ml_model == 'XGBoost':
    #XGBoost prediction
    new_prediction_xgb = xgb_model.predict(user_encoded_df)
    new_prediction_xgb = new_prediction_xgb.astype(int)
    st.write("XGBoost Traffic Prediction: {}".format(*new_prediction_xgb))
    st.subheader("Plot of XGBoost Feature Importance:")
    #feature importance
    st.image('xgb_feature_imp.svg')



