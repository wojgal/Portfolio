# Gra Okienkowa: TicTacToe + AI

## 1. Wprowadzenie
Projekt przedstawią grę **Kółko i Krzyżyk** z wbudowaną **sztuczną inteligencją** opartną na algorytmie **MinMax**. Użytkownik ma możliwość gry w dwóch trybach:
1. **Gracz vs Gracz** - klasyczna rozgrywka dla dwóch osób.
2. **Gracz vs AI** - rozgrywka z komputerem, który analizuje najlepszy możliwy ruch.

Gra została stworzona w **PyGame** i oferuje przyjazny interfejs użytkownika z dodatkowymi elementami wizualnymi i dźwiękowymi.

## 2. Jęzki i Biblioteki
Projekt został zrealizowany w jęzku **Python** z użyciem następujących bibliotek:
* PyGame

## 3. Struktura projektu
Gra została zaimplementowana w modularnej strukturze, co ułatwia jej rozwijanie. Najważniejsze elementy projektu to:
* Klasa `TicTacToe` - odpowiada za główną logikę gry, obsługę trybów, zarządzanie planszą, obsługę zdarzeń oraz interakcję gracza z AI
* Klasa `Field` - reprezentuje pojedyczne pole planszy, umożliwiając przypisywanie symboli, rysowanie ich na ekranie oraz sprawdzanie kliknięć gracza.
* Klasa `Button` - implementuje interaktywne przyciski z funkcją aktywacji i obsługą kliknięć.
* **Algorytm MiniMax** - wykorzystywany przez AI do podejmowania optymalnych decyzji w grze.

## 4. Funkcje gry
* **Tryb rozgrywki** - możliwość gry w trybie 1v1 lub przeciwko AI.
* **Algorytm MiniMax** - AI analizuje możliwe ruchy i wybiera najlepszy, zapewniając trudną, ale sprawiedliwą rozgrywkę.
* **Interfejs użytkownika** - intuicyjne przyciski do wyboru trybu gry, restartu gry, czy zmiany symbolu gracza.
* ** Kolizje** - mechanizmy obsługi kliknięć na planszy i przyciskach.
* **Punktacja i wyniki** - automatyczne wykrywanie zwycięzcy, remisu oraz aktualizacja wyników na ekranie.

## 5. Algorytm MinMax
Sztuczna inteligencja wykorzystuje algorytm **MinMax**, który analizuje wszystkie możliwe ruchy i ich konsekwencje, aby wybrać najlepszą strategię. Algorytm działa rekurencyjnie, przeszukując drzewo stanów gry, gdzie:
1. **Max** - AI maksymalizuje swój wynik, wybierając korzystne ruchy.
2. **Min** - AI zakłada, że przeciwnik minimalizuje jej wyniki, więc stara się ograniczyć ryzyko przegranej

Sztuczna inteligencja przewiduje wszystkie możliwe scenariusze, oceniając każdy ruch, aż do zwycięstwa, remisu lub osiągnięcia maksymalnej głębokości analizy. Dzięki temu algorytm zawsze podejmuje optymalne decyzje, oferując wymagające wyzwanie dla gracza.

## 6. Elementy wizaulne i dźwiękowe
* **Grafiki** - plansza i symbole (X oraz O) zostały stworzone jako osobne grafiki, co umożliwia ich łatwe dostosowywanie.
* **Przyciski** - w pełni funkcjonalne przyciski umożliwiające interakcję z grą, w tym wybór trybu i restart.
* **Oprawa dźwiękowa** - gra zawiera efekty dźwiękowe takie jak kliknięcia, dźwięki zakończenia gry, co wzbogaca wrażenia z rozgrywki.

## 7. Przykładowy zrzut ekranu
![TicTacToe](https://github.com/user-attachments/assets/a96eda58-32b0-4e20-9247-e7a27add578f)
