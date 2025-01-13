import streamlit as st
import torch
from linear_regression_model import LinearRegressionModel

# Załadowanie modelu tylko raz
if 'model' not in st.session_state:
    st.session_state.model = LinearRegressionModel()
    st.session_state.model.load_state_dict(torch.load(f='model.pth', map_location='cpu', weights_only=True))
    st.session_state.model.eval()

# Interfejs użytkownika
st.title('Przewidywanie Cen domów 🏠')
area = st.slider(label='Powierzchnia w m2', min_value=25, max_value=300, step=1)

# Przygotowanie tensora
area_tensor = torch.tensor(area, dtype=torch.float32).unsqueeze(dim=0)

# Predykcja modelu
with torch.inference_mode():
    price = st.session_state.model(area_tensor)

# Wyświetlenie wyniku
st.metric(label='Cena mieszkania', value=f'${int(price):,.2f}')
    
