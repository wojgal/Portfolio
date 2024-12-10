# Model Regresji Liniowej: Przewidywanie Cen Domów.

## 1. Wprowadzenie
Projekt przedstawia stworzenie **modelu regresji liniowej** do **przewidywania cen domów** na podstawie ich **powierzchni**. Zadanie ilustruje prostą, ale skuteczną metodologię **wnioskowania maszynowego**. Całość osadzona jest w intuicyjnej **aplikacji webowej**, umożliwiającej użytkownikowi wygodne korzystanie z modelu i eksplorację jego wyników w czasie rzeczywistym.

## 2. Język i Biblioteki
Projekt został zaimplementowany w jeżyku programowania **Python**, przy użyciu następujących bibliotek:
* PyTorch
* NumPy
* Pandas
* Streamlit
* Matplotlib

## 3. Zbiór Danych
Wykorzystany zbiór danych to [House Price Prediction Treated Dataset](https://www.kaggle.com/datasets/aravinii/house-price-prediction-treated-dataset?resource=download) zawierający informacje o sprzedaży domów w King Country w USA. Całość podzielona jest na dwa osobne pliki: `train` i `test`, które zawierają po 14 kolumn.

## 4. Proces Tworzenia
1. Analiza i przetworzenie danych - dane są ładowane do projektu i przechodzą wstępna selekcję, po której zostają tylko kolumny `living_in_m2` (powierzchnia) i `price` (cena). Następnie są zamieniane na tensory.
2. Budowa modelu regresji liniowej - zaimplementowana zostaje główna klasa `LinearRegressionModel`, która dziedziczy po `nn.Module`. W środku zostaje użyta klasa `nn.Linear` odpowiadająca modelowi regresji liniowej. Ustawiona zostaje na 1 argument wejściowy (powierzchnię) i 1 argument wyjściowy (cenę).
3. [..]
