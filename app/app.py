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
     st.title('Predicción de Cancelaciones Hoteleras con Machine Learning')

     st.sidebar.header('User Input Parameters')

    
def user_input_parameters():
    pass

df= user_input_parameters()


option = ['Finished Model']
#model = st.sidebar.selectbox('Modelo Final')

st.subheader('User Input Parameters')
#st.subheader(model)
st.write(df)
          

     
if __name__ == '__main__':
     main()
 











