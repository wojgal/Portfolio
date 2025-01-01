# Gra Okienkowa: Tetris

## 1. Wprowadzenie
Projekt przedstawia klasyczną grę **Tetris** w wersji okienkowej, stworzoną w **PyGame**. Gra została wzbogacona o estetyczne **grafiki klocków i planszy** oraz **efekty dźwiękowe**, które dodają dynamiczności i imersji rozgrywce. Aplikacja zawiera zaimplementowaną logikę gry, obsługę punktacji oraz możliwość rotacji i przesuwania klocków, dzięki czemu wiernie odtwarza mechanikę oryginalnego Tetrisa.

## 2. Języki i biblioteki
Projekt został zrealizowany w jęzku Python, przy użyciu biblioteki:
* PyGame

## 3. Struktura projektu
Gra została zaimplementowana w sposób modularny, co zwiększa jej przejrzystość i możliwości dalszego rozwijania. Najważniejsze elementy projektu to:
* Klasa `Tetris` - odpowiedzialna za główną logikę gry, w tym zarządzanie planszą, kolizjami, punktacją i przebiegiem rozgrywki.
* Klasa `Block` - reprezentuje pojedyczny klocek Tetrisa, zapewniając funkcje takie jak przesuwanie i rotacja.

## 4. Funkcje gry
* Ruch klocków - gracz może przesuwać klocki w lewo, prawo oraz przyspieszać ich opadanie.
* Rotacja klocków - klocki mogą być obracane w celu ich lepszego dopasowania na planszy.
* Punktacja - gra automatycznie przyznaje punkty za usuwanie pełnych wierszy. Mechanika punktacji opiera się na liczbie usuniętych wierszy jednocześnie.
* Kolizje - system kolizji zapobiega przenikaniu klocków przez siebie oraz przez granice planszy.

## 5. Elementy wizualne i dźwiękowe
* Grafiki - wszystkie elementy wizualne, takie jak klocki i plansza, są zapisane w formie oddzielnych plików graficznych `.png`.
* Dźwięki - gra oferuje efekty dźwiękowe takie jak dźwięki usuwania wierszy, czy końca gry. Całość dopełnia soundtrack w tle, który buduje atmosferę i wprowadza gracza w klimat klasycznego Tetrisa.

## 6. Przykładowe zrzuty ekranu z rozgrywki
![Tetris1](https://github.com/user-attachments/assets/b23c8403-7740-4e49-b6ae-1871ca051a80)
![Tetris2](https://github.com/user-attachments/assets/e4b4adf7-8bc8-4e7f-af7f-2fc099eaa351)


