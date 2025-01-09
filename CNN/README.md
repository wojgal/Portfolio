# Model CNN: Klasyfikacja obrazów krajobrazu

## 1. Wprowadzenie
Projekt przedstawia stworzenie **modelu klasyfikacji obrazów scen naturalnych** przy użyciu **konwolucyjnych sieci nueronowcyh (CNN)**. Celem było zbudowanie i ocena modelu, który z wysoką dokładnością klasyfikuje zdjęcia do odpowiednich kategorii scen naturalnych na podstawie dostarczonego zbioru danych. Wyniki przedstawione są zarówno w formie wizualnej, za pomocą wykresów i metryk, jak i w formie funkcjonalnej - w postaci aplikacji webowej umożliwiającej praktyczne użycie modelu.

## 2. Języki i biblioteki
Projekt został zrealizowany w języku **Python**, przy użyciu następujących bibliotek:
* PyTorch
* PyTorchVision
* PyTorchMetrics
* Pandas
* Matplotlib
* Streamlit

## 3. Zbiór danych
Wykorzystany zbiór danych to [**Intel Image Classification**](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). Zbiór ten pierwotnie służył do konkursu organizowanego przez **Analytics Vidhya** we współpracy z firmą **Intel**, mającego na celu klasyfikację scen naturalnych na różne kategorie. 

Rozmiar zbioru danych:
* Łączna liczba obrazów: `25 000`
* Rozmiar obrazów: `150x150` pikseli

Podział zbioru danych:
* Zbiór treningowy: `11 200`
* Zbiór walidacyjny: `2 800`
* Zbiór testowy: `3 000`

Każdy z obrazów został przypisany do jednej z sześciu kategorii:
* **Buildings** (budynki)
* **Forest** (las)
* **Glacier** (lodowiec)
* **Mountain** (góra)
* **Sea** (morze)
* **Street** (ulica)

W celu efektywnego zarządzania i przetwarzania danych, zaimplementowana została klasa `CustomDataset`, która umożliwia:
* **Dynamiczne ładowanie obrazów** w czasie treningu i ewaluacji modelu
* **Transformacje danych**, takie jak normalizacja i augmentacja, dostosowane do specyfiki zbioru
* **Etykietowanie** odpowiednich labeli do obrazów na podstawie struktury katalogów

## 4. Architektura Konwolucyjnej Sieci Neuronowej (CNN)
...

## 5. Funkcja straty, optymalizator, oraz metryki oceny jakości
...

## 6. Wizualizacja wyników
...

## 7. Aplikacja webowa z wykorzystaniem modelu
...
