# Model Regresji Liniowej: Przewidywanie Cen Domów.

## 1. Wprowadzenie
Projekt przedstawia stworzenie **modelu regresji liniowej** do **przewidywania cen domów** na podstawie ich **powierzchni**. Zadanie ilustruje prostą, ale skuteczną metodologię **wnioskowania maszynowego**. Całość osadzona jest w intuicyjnej **aplikacji webowej**, umożliwiającej użytkownikowi wygodne korzystanie z modelu i eksplorację jego wyników w czasie rzeczywistym.

## 2. Język i Biblioteki
Projekt został zrealizowany w języku **Python**, przy użyciu następujących bibliotek:
* PyTorch
* NumPy
* Pandas
* Streamlit
* Matplotlib

## 3. Zbiór Danych
Wykorzystany zbiór danych to [House Price Prediction Treated Dataset](https://www.kaggle.com/datasets/aravinii/house-price-prediction-treated-dataset?resource=download) zawierający informacje o sprzedaży domów w King Country w USA. Całość podzielona jest na dwa osobne pliki: `train` i `test`, które zawierają po 14 kolumn. Ostatecznie zostały wybrane tylko 2 kolumny `prize` i `living_in_m2`, które będą wykorzystane przy treningu modelu.

## 4. Model Regresji Liniowej
Zaimplementowana została klasa główna `LinearRegressionModel`, która dziedziczy po `nn.Module`. W jej wnętrzu jako **warsta liniowa** zostaje użyta klasa `nn.Linear`, która posiada jeden parametr wejściowy (powierzchnia) i 1 wyjściowy (cena).

## 5. Funkcja straty, optymalizator i pętla treningowa
Funkcja straty obliczana jest za pomocą klasy `nn.L1Loss` liczącej **średni błąd bezwzględny (MAE)**. 
Jako optymalizatora użyjemy `optim.SGD` metody **stochastycznego spadku gradientu**. 
Szybkość uczenia wynośi `0.05`, a ilość epok w treningu to `501`. 

## 6. Wizualizacja wyników i wartość funkcji straty
### 6.1 Wyniki modelu przed treningiem
![Predictions_before_train](https://github.com/user-attachments/assets/6519e46d-e5ec-493f-ad6a-eb0456fff902)

### 6.2 Wyniki modelu po treningu
![Predictions_after_train](https://github.com/user-attachments/assets/3640bc5d-c2ff-404a-a938-7367e87111de)

Po porównaniu dwóch wykresów można zauważyć, iż zielona linia przedstawiająca predykcje modelu dopasowała się do trendu danych.

### 6.3 Wartość funkcji straty
![Loss_values](https://github.com/user-attachments/assets/02a8bf37-86d1-4e4c-bfe2-a7cfd8b14cdf)

Wartość funkcji straty wraz z kolejnymi epokami drastycznie **spada** co jest dobrą oznaką. Model po treningu osiągnął swoje możliwości dla aktualnej konfiguracji.

## 7. Aplikacja webowa z wykorzystaniem modelu
Wartości parametrów wytrenowanego modelu zostały zapisane do pliku `model.pth`, który został wykorzystany przy implementacji aplikacji webowej. W prostym interfejsie użytkownik może korzystać z możliwości modelu do własnych predykcji cen mieszkań na podstawie ich powierzchni. 

![app_ui](https://github.com/user-attachments/assets/97c1fbe7-fe06-41c8-a981-28f413c20236)





