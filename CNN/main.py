import streamlit as st
import torch
from cnn import CNN, load_image_to_eval, model_eval

# Załadowanie modelu tylko raz
if 'model' not in st.session_state:
    st.session_state.model = CNN(input_shape=3, hidden_units=64, output_shape=15)
    st.session_state.model.load_state_dict(torch.load(f='cnn_model.pth', map_location='cpu', weights_only=True))

# Interfejs użytkownika
st.title('Rozpoznawanie warzyw')
uploaded_image = st.file_uploader(label='Zaladuj zdjecie warzywa', type=['png', 'jpg', 'jpeg'])


if uploaded_image:
    st.image(uploaded_image)
    image = load_image_to_eval(image_file=uploaded_image)
    vegetable = model_eval(model=st.session_state.model, image=image)

    st.write(f'{vegetable}')

    