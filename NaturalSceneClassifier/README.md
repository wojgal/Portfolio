# Natural Scene Classifier

## 1. Wprowadzenie
Projekt przedstawia stworzenie **modelu klasyfikacji obrazów scen naturalnych** przy użyciu **Konwolucyjnych Sieci Neuronowcyh (CNN)**. Celem było zbudowanie i ocena modelu, który z wysoką dokładnością klasyfikuje zdjęcia do odpowiednich kategorii scen naturalnych na podstawie dostarczonego zbioru danych. Wyniki przedstawione są zarówno w formie wizualnej, za pomocą wykresów i metryk, jak i w formie funkcjonalnej - w postaci aplikacji webowej, dostępnej online pod adresem: [Natural Scene Classifier](https://naturalsceneclassifier.streamlit.app/). Dzięki czemu każdy może samodzielnie przetestować działanie modelu na własnych obrazach.

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
W projekcie zastosowano **Konwolucyjną Sieć Neuronową (CNN)**, zaprojektowaną w celu skutecznej **klasyfikacji obrazów scen nautralnych na sześć kategorii**. Model został skonstruowany w oparciu o architekturę **Tiny VGG**, która łączy wysoką efektynwość z umiarkowaną złożonością obliczeniową. Struktura sieci składa się z kilku bloków konwolucyjnych, które wyodrębniają kluczowe cechy wizualne z obrazów oraz części klasyfikacyjnej, która przekształca te cechy w ostateczne predykcje.

**Struktura architektury**
1. **Bloki konwolucyjne**
   * Model składa się z **trzech bloków konwolucyjnych**, z których każdy zawiera:
       * Dwie warstwy **konwolucyjne** (`Conv2d`) z funkcją aktywacji **ReLU**, które wyodrębniają cechy wizualne z obrazów
       * Warstwę **max pooling** (`MaxPool2d`), która redukuje wymiary przestrzenne danych, zwiększając efektywność obliczeniową
   * Liczba kanałów (filtrów) **wzrastwa z każdym blokiem**, co umożliwia przechwycenie bardziej złożonych wzorców w głębszych warstwach:
       * Blok 1: **64** filtry (wartość `hidden_units`)
       * Blok 2: **128** filrtów
       * Blok 3: **256** filtrów
2. **Częśc klasyfikacyjna**:
   * Po przejściu przez bloki konwolucyjne, dane są spłaszczane za pomocą warstwy `Flatten`
   * Następnie dane przechodzą przez warstwę w pełni połączoną (`Linear`), która dokonuje klasyfikacji na 6 kategorii
3. **Funkcja aktywacji na wyjściu**:
   * Model zwraca surowe wartości (logity), które można przekształcić w prawdopodobieństwa klas za pomocą funkcji **Softmax**

![image](https://github.com/user-attachments/assets/a8f1d51c-f117-491c-96fc-6498600827a7)

## 5. Funkcja straty, optymalizator oraz metryki oceny jakości
W projekcie wykorzystano `CorssEntropyLoss` jako funkcję straty, odpowiednią dla problemów wielkoklasowej klasyfikacji. Do optymalizacji wybrany został `Adam`, który zapewnia szybkie i stabilne uczenie się dzięki adaptacyjnym współczynnikom uczenia, przy `lr=0.001`.

Model oceniony został za pomocą:
1. Dokładności (`Accuracy`), monitorującej ogólną skuteczność klasyfikacji
2. Macierzy pomyłek (`Confusion Matrix`), umożliwiającej analizę błędów i identyfikację najczęściej mylonych kategorii

Takie podejście pozwoliło na efektywne trenowanie i ocenę modelu, zapewniając jednocześnie możliwość szczegółowej analizy jego wyników.

## 6. Wizualizacja wyników
Wyniki treningu i ewaluacji modelu CNN zostały przedstawione w formie trzech wykresów, które ilustrują postęp w procesie uczenia się oraz efektywności klasyfikacji.

### 6.1 Wykres dokładności (`Accuracy`)<br/>
Wykres przedstawia zmianę dokładnosći w trakcie treningu (zbiór treningowy i walidacyjny) oraz ostateczną dokładność modelu na zbiorze testowym. Model stopniowo przez `15` **epok** poprawiał swoje wyniki, osiągając końcowo dokładność **84,24%**.

![accuracy](https://github.com/user-attachments/assets/ee318eb7-850e-45af-a429-e55ef17156f9)

### 6.2 Wykres funkcji straty (`Loss Function`)<br/>
Wykres obrazuje zmiany wartości funkcji straty podczas treningu i walidacji oraz ostateczną stratę na zbiorze testowym. Obserwowany trend spadkowy potwierdza efektywność procesu uczenia się, choć pod koniec można zauważyć lekki wzrost straty walidacyjnej, co może być oznaką początków przeuczenia.

![loss](https://github.com/user-attachments/assets/f8504bd7-04e1-45a2-992b-b308407890c0)

### 6.3 Macierz pomyłek (`Confusion Matrix`)<br/>
Macierz pomyłek ilustruje, jak często model klasyfikował obrazy poprawnie oraz które klasy były najczęściej mylone. W naszym przypadku szczególnie zauważalne są pomyłki między kategoriami:
* Budynki (`buildings`) i ulica (`street`) - co prawdopodobnie wynika z podobnych elementów architektonicznych widocznych na obu klasach
* Góra (`mountain`) i lodowiec (`glacier`) - co jest zrozumiałe, biorać pod uwagę wizualne podobieńśtwo między tymi krajobrazami

Analiza macierzy pomyłek pozwala lepiej zrozumieć trudności modelu i wskazuje potencjalne możliwości ulepszenia poprzez modyfikację zbioru danych lub architektury.

![confusion_matrix](https://github.com/user-attachments/assets/1e75e00a-89a2-4245-a8b6-cb5404ad1460)


## 7. Aplikacja webowa z wykorzystaniem modelu
Model został wdrożony w postaci prostej i intuicyjnej aplikacji webowej stworzonej przy użyciu **Streamlit**, która umożliwia użytkownikom interakcję z modelem i testowania jego działania w praktyce.

Aplikacja dostępna jest online pod adresem [www.naturalsceneclassifier.streamlit.app](https://naturalsceneclassifier.streamlit.app/).

**Funkcjonalności:**
* **Wgrywanie obrazów**: użytkownik przesyła dowolny obraz krajobrazu
* **Przetwarzanie**: obraz jest automatycznie transformowany, a model dokonuje predykcji
* **Wyniki**: aplikacja wyświetla przewidywaną kategorię

![image](https://github.com/user-attachments/assets/820701a6-b842-48e9-961c-0fd6eeb36b3e)

![image](https://github.com/user-attachments/assets/652ad644-29a2-480c-a46e-4d89590bcfa2)


