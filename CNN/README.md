# Natural Scene Classifier

## 1. Wprowadzenie
Projekt przedstawia stworzenie **modelu klasyfikacji obrazów scen naturalnych** przy użyciu **Konwolucyjnych Sieci Neuronowcyh (CNN)**. Celem było zbudowanie i ocena modelu, który z wysoką dokładnością klasyfikuje zdjęcia do odpowiednich kategorii scen naturalnych na podstawie dostarczonego zbioru danych. Wyniki przedstawione są zarówno w formie wizualnej, za pomocą wykresów i metryk, jak i w formie funkcjonalnej - w postaci aplikacji webowej umożliwiającej praktyczne użycie modelu.

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

W zbiorze występują sporadyczne nieścisłości w etykietowaniu np. obraz **samochodu** znajduje się w klasie `lodowiec`. Anomalie te mogą wynikać z błędów etykietowania lub umyślnego wprowadzenia trudniejszych przypadków w celu dorzucenia kolejnych wyzwań podczas konkursu. Tego typu nieprawidłowości **zwiększają złożoność problemu** i wymagają od modelu większej zdolności do **generalizacji**.

W celu efektywnego zarządzania i przetwarzania danych, zaimplementowana została klasa `CustomDataset`, która umożliwia:
* **Dynamiczne ładowanie obrazów** w czasie treningu i ewaluacji modelu
* **Transformacje danych**, takie jak normalizacja i augmentacja, dostosowane do specyfiki zbioru
* **Etykietowanie** odpowiednich labeli do obrazów na podstawie struktury katalogów

## 4. Architektura Konwolucyjnej Sieci Neuronowej (CNN)
Model opiera się na uproszczonej architekturze **Tiny VGG**, dostosowanej do klasyfikacji obrazów scen naturalnych. Został zaprojektowany z myślą o wysokiej wydajności i możliwości trenowania na sprzęcie o ograniczej mocy obliczeniowej, zachowując jedocześnie zdolność do dokładnego rozpoznawania cech wizualnych.
WSTĘP TRZEBA JESZCZE DOPRACOWAĆ

**Struktura architektury**
1. Bloki konwolucyjne
   * Model składa się z **trzech bloków konwolucyjnych**, z których każdy zawiera:
       * Dwie warstwy **konwolucyjne** (`Conv2d`) z funkcją aktywacji `ReLU`, które wyodrębniają cechy wizualne z obrazów
       * Warstwę **max pooling** (`MaxPool2d`), która redukuje wymiary przestrzenne danych, zwiększając efektywność obliczeniową
   * Liczba kanałów (filtrów) **wzrastwa z każdym blokiem**, co umożliwia przechwycenie bardziej złożonych wzorców w głębszych warstwacH:
       * Blok 1: **64** filtry (wartość `hidden_units`)
       * Blok 2: **128** filrtów
       * Blok 3: **256** filtrów
2. Częśc klasyfikacyjna:
   * Po przejściu przez bloki konwolucyjne, dane są spłaszczane za pomocą warstwy **Flatten**
   * Następnie dane przechodzą przez warstwę w pełni połączoną (`Linear`), która dokonuje klasyfikacji na 6 kategorii
3. Funkcja aktywacji na wyjściu:
   * Model zwraca surowe wartości (logity), które można przekształcić w prawdopodobieństwa klas za pomocą funkcji **Softmax**
## 5. Funkcja straty, optymalizator, oraz metryki oceny jakości
...

## 6. Wizualizacja wyników
...

## 7. Aplikacja webowa z wykorzystaniem modelu
...
