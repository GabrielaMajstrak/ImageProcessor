## Opis Projektu
Jest to aplikacja desktopowa umożliwiająca wykonywanie podstawowych operacji na obrazach, takich jak:
- Konwersja do skali szarości
- Regulacja jasności
- Regulacja kontrastu
- Negatyw obrazu
- Dodawanie efektu winiety
- Binaryzacja obrazu
- Nakładanie filtrów: uśredniającego, gaussowskiego, wyostrzającego 
- Dodawanie szumu
- Detekcja krawędzi za pomocą krzyża Robertsa oraz operatora Sobela

Aplikacja została napisana w języku Python z wykorzystaniem bibliotek **Tkinter** oraz **CustomTkinter** do interfejsu graficznego. Operacje na obrazach zostały zaimplementowane ręcznie, bez użycia bibliotek takich jak OpenCV.

**Projekt został stworzony na potrzeby przedmiotu Biometria.**

## Wymagania
Aby uruchomić aplikację, wymagane jest posiadanie:
- Python 3.x
- Bibliotek:
  - `numpy`
  - `matplotlib`
  - `pillow`
  - `tkinter` (wbudowany w Pythona)
  - `customtkinter`

Instalacja wymaganych bibliotek:
```sh
pip install numpy matplotlib pillow customtkinter

## Autorzy
- Jan Opala
- Gabriela Majstrak
