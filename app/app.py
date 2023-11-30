import streamlit as st
import pickle
import pandas as pd
import os


directorio_actual = os.getcwd()

#Extraer datos

with open(os.path.join(directorio_actual, '..', 'Models','finished_model.pkl'), 'rb') as archivo_entrada:
     modelo = pickle.load(archivo_entrada)

#funcion para clasificar las reservas

def classify(num):
     if num == 0:
          return 'Not Canceled'
     else:
          return 'Canceled'

def main():
     #pongo titulo
     st.title('Predicción de Cancelaciones Hoteleras con Machine Learning')

     #pongo titulo a la barra lateral
     st.sidebar.header('User Input Parameters')

     #introduzco variables para trastear de la barra lateral
     def user_input_parameters():
          no_of_adults = st.sidebar.slider('no_of_adults', 0, 4)
          no_of_children = st.sidebar.slider('no_of_children', 0, 10) 
          no_of_weekend_nights = st.sidebar.slider('no_of_weekend_nights', 0, 6) 
          no_of_week_nights = st.sidebar.slider('no_of_week_nights', 0, 16) 
          type_of_meal_plan = st.sidebar.slider('type_of_meal_plan', 0, 2) 
          required_car_parking_space = st.sidebar.slider('required_car_parking_space', 0, 1) 
          room_type_reserved = st.sidebar.slider('room_type_reserved', 1, 7) 
          lead_time = st.sidebar.slider('lead_time', 0, 443) 
          arrival_month = st.sidebar.slider('arrival_month', 1, 12) 
          market_segment_type = st.sidebar.slider('market_segment_type', 0, 4) 
          repeated_guest = st.sidebar.slider('repeated_guest', 0, 1) 
          no_of_previous_cancellations = st.sidebar.slider('no_of_previous_cancellations', 1, 11) 
          no_of_previous_bookings_not_canceled = st.sidebar.slider('no_of_previous_bookings_not_canceled', 0, 55)
          avg_price_per_room = st.sidebar.slider('avg_price_per_room', 0, 375)
          no_of_special_requests = st.sidebar.slider('no_of_special_requests', 0, 5)
          data = {'no_of_adults': no_of_adults,
                  'no_of_children': no_of_children,
                  'no_of_weekend_nights': no_of_weekend_nights,
                  'no_of_week_nights': no_of_week_nights,
                  'type_of_meal_plan': type_of_meal_plan,
                  'required_car_parking_space': required_car_parking_space,
                  'room_type_reserved': room_type_reserved,
                  'lead_time': lead_time,
                  'arrival_month': arrival_month,
                  'market_segment_type': market_segment_type,
                  'repeated_guest': repeated_guest,
                  'no_of_previous_cancellations': no_of_previous_cancellations,
                  'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
                  'avg_price_per_room': avg_price_per_room,
                  'no_of_special_requests': no_of_special_requests,
                  }
          features = pd.DataFrame(data, index= [0])
          return features
     
     df= user_input_parameters()

     #introduzco en la barra lateral el tipo de modelo
     option = ['Finished Model Gradient Boosting']
     model = st.sidebar.selectbox('Modelo Final', option)        

     #introduzco el dataframe para que se vea la info que elijo en los datos de la barra lateral
     #y el modelo para evaluar
     st.subheader('User Input Parameters')
     st.subheader(model)
     st.write(df)
     
     #uso boton run para que me devuelva el tipo de clasificacion
     if st.button('RUN'):
          if model == 'Finished Model Gradient Boosting':
               st.success(classify(modelo.predict(df)))
               

if __name__ == '__main__':
     main()
 











