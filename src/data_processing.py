import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

hotel = pd.read_csv('../data/raw/Hotel_reservations.csv', index_col='Booking_ID')
hotel

hotel.info()

hotel.isnull().sum()

plt.figure(figsize = (20,25))

plt.subplot(4,2,1)
plt.title('Numero de adultos')
sns.countplot(x = 'no_of_adults', palette = 'tab10', data = hotel)
plt.xlabel('Number of adults')
plt.ylabel('Counting values')

plt.subplot(4,2,2)
plt.title('Mes de llegada')
sns.countplot(x = 'arrival_month', palette = 'tab10', data = hotel)
plt.xlabel('Arrival month')
plt.ylabel('Counting values')

plt.subplot(4,2,3)
plt.title('Numero de noches en fin de semana')
sns.countplot(x = 'no_of_weekend_nights', palette = 'tab10', data = hotel)
plt.xlabel('Number of nights on weekends')
plt.ylabel('Counting values')

plt.subplot(4,2,4)
plt.title('Numero de noches ebtre semana')
sns.countplot(x = 'no_of_week_nights', palette = 'tab10', data = hotel)
plt.xlabel('Number of weekly nights')
plt.ylabel('Counting values')

plt.subplot(4,2,5)
plt.title('Plan de comidas')
sns.countplot(x = 'type_of_meal_plan', palette = 'tab10', data = hotel)
plt.xlabel('Meal plan type')
plt.ylabel('Counting values')

plt.subplot(4,2,6)
plt.title('Tipo de habitacion')
sns.countplot(x = 'room_type_reserved', palette = 'tab10', data = hotel)
plt.xlabel('Type of room booked')
plt.ylabel('Counting values')

plt.subplot(4,2,7)
plt.title('Tipo de segmento de mercado')
sns.countplot(x = 'market_segment_type', palette = 'tab10', data = hotel)
plt.xlabel('Market Segment Type')
plt.ylabel('Counting values')

plt.subplot(4,2,8)
plt.title('Estado de la reserva')
sns.countplot(x = 'booking_status', palette = 'tab10', data = hotel)
plt.xlabel('Booking Status')
plt.ylabel('Counting values')

hotel['type_of_meal_plan'].unique()

meal_plan_mapping = {'Not Selected': 0,'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3}
hotel['type_of_meal_plan'] = hotel['type_of_meal_plan'].map(meal_plan_mapping)

hotel['room_type_reserved'].unique()

room_type_mapping = {'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7}
hotel['room_type_reserved'] = hotel['room_type_reserved'].map(room_type_mapping)

hotel['market_segment_type'].unique()

market_mapping = {'Offline': 0,'Online': 1,'Corporate': 2,'Aviation': 3,'Complementary': 4,}
hotel['market_segment_type'] = hotel['market_segment_type'].map(market_mapping)

hotel['booking_status'].unique()

booking_mapping = {'Not_Canceled': 0, 'Canceled': 1}
hotel['booking_status'] = hotel['booking_status'].map(booking_mapping)

hotel.head()

hotel.isnull().sum()

valor = hotel["booking_status"].value_counts()
suma = valor.sum()
porcentaje = valor*100/suma
print(porcentaje)

#hitograma tiempo antelacion reserva
sns.histplot(hotel['lead_time'], bins=30, kde=True, color='skyblue')
plt.title('Distribución de Lead Time (Tiempo de Antelación)')
plt.xlabel('Lead Time (Días)')
plt.ylabel('Frecuencia')
plt.show()

sns.histplot(hotel['avg_price_per_room'], bins=30, kde=True, color='salmon')
plt.title('Distribución de Precio Promedio por Habitación')
plt.xlabel('Precio Promedio por Habitación')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='lead_time', y='avg_price_per_room', data=hotel, hue='booking_status', palette='viridis', s=80)
plt.title('Relación entre Lead Time y Precio Promedio por Habitación')
plt.xlabel('Lead Time (Días)')
plt.ylabel('Precio Promedio por Habitación')
plt.legend(title='Booking Status', loc='upper right')
plt.show()

fig, ax = plt.subplots(figsize=(24, 10))
sns.heatmap(hotel.corr(), annot = True, cmap = 'magma',fmt='.2f')
plt.title('Correlacion entre variables')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='booking_status', y='avg_price_per_room', data=hotel)
plt.title('Relación entre avg_price_per_room y booking_status')
plt.show()

hotel = hotel.drop(['arrival_year','arrival_date'], axis=1)
hotel

nombre_archivo = 'hotel_dataset_modificado.csv'

hotel.to_csv(nombre_archivo, index=False)
