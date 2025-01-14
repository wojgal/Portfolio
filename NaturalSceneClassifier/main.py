import streamlit as st
import torch
from cnn import CNN, model_eval

# Załadowanie modelu
if 'model' not in st.session_state:
    st.session_state.model = CNN(input_shape=3, hidden_units=64, output_shape=6)
    st.session_state.model.load_state_dict(torch.load(f='model.pth', map_location='cpu', weights_only=True))
    st.session_state.model.eval()

# Interfejs użytkownika
st.title('Natural Scene Classifier 📝')

st.markdown('---')

# Krótki opis aplikacji
with st.expander("O aplikacji"):
    st.write("""
        Aplikacja pozwala na rozpoznawanie naturalnych scen na podstawie przesłanego obrazu.\n
        Rozpoznawane kategorie to:\n 
        * Ulica 🛣\n 
        * Budynek 🏢\n 
        * Morze 🌊\n
        * Las 🌳\n
        * Lodowiec ❄️\n
        * Góra 🏔\n
        Wgraj obraz w formacie PNG, JPG lub JPEG, a model wyświetli przewidywaną klasę.
    """)

st.markdown('---')

# Załadowanie obrazu
uploaded_image = st.file_uploader(label='Zaladuj zdjecie krajobrazu, a model rozpozna jego klasę!', type=['png', 'jpg', 'jpeg'])

# Wyświetlanie obrazu i jego ewaluacja
if uploaded_image:
    try:
        st.image(uploaded_image, caption='Wgrane zdjęcie', use_column_width=True)

        if st.button('Rozpoznaj obraz'):
            label = model_eval(model=st.session_state.model, image_file=uploaded_image)
            st.success(f'Na obrazie znajduje się: **{label}**')

    except Exception as e:
        st.error('Wystąpił błąd podczas przetwarzania obrazu. Upewnij się, że wgrałeś poprawny plik w formacie PNG, JPG lub JPEG.')