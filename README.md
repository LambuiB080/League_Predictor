**1. Opis rzeczywistego problemu.**

Projekt League_Predictor skupia się na przewidywaniu meczy ligi angielskiej Premier League. W projekcie utworzony został model przewidujący mecze na podstawie wcześniejszych
statystyk poszczególnych zespołów takich jak forma drużyny, średnie gole, średnie żółte kartki, czy celne strzały na bramke. Wynikiem jest prawdopodobieństwo rezultatu danego meczu. 
Projekt może posłużyć do pomocy w obstawianiu zakładów bukmacherskich.

**2. State of Art**

W koncepcji przewidywania meczy należy rozważyć 3 koncepcje:

_Regresja logistyczna_ - Regresja logistyczna przekształca problem przewidywania wyniku w prawdopodobieństwo, wykorzystując funkcję logistyczną (sigmoidę).
Model regresji logistycznej jest uczony na danych historycznych. Proces ten polega na:

+ przypisywaniu wag: Na początku wagi są losowe, ale podczas uczenia algorytm dostosowuje je, aby minimalizować błąd.

+ określeniu funkcji kosztu: Regresja logistyczna korzysta z funkcji entropii krzyżowej, aby ocenić różnicę między przewidywanymi a rzeczywistymi wynikami.

+ optymalizacji: Algorytm iteracyjnie aktualizuje wagi, aby minimalizować funkcję kosztu.

    _Zalety:_ 
    + Prosty do zrozumienia i interpretacji.
    + Działa dobrze przy ograniczonej liczbie cech.
    + Szybki w implementacji.

    _Wady:_
    + Zakłada liniową zależność między cechami a logarytmem prawdopodobieństwa, co nie zawsze jest prawdziwe.
    + Wrażliwy na brakujące dane i silnie skorelowane cechy.

_Gradient Boosting_ - Gradient boosting polega na budowaniu kolejnych modeli iteracyjnie, które poprawiają błędy poprzednich modeli.
W każdym kroku obliczany jest błąd aktualnego modelu (czyli różnica między prawdziwą wartością a wartością przewidywaną).
W każdym kroku tworzony jest też nowy model, który próbuje przewidzieć pozostały błąd poprzedniego modelu.
Nowe modele uczą się, jak "przesuwać" przewidywania w kierunku rzeczywistych wartości.
Wyniki nowego modelu są dodawane do poprzedniego modelu, ale z pewnym "współczynnikiem uczenia" (η), który kontroluje, jak duży wpływ mają nowe modele na końcowy wynik.
Proces ten jest powtarzany wielokrotnie, a każdy kolejny model stara się poprawić błędy poprzedniego.
Ostateczny wynik to suma wkładów wszystkich modeli.


    _Zalety:_ 
    + Prosty do zrozumienia i interpretacji.
    + Działa dobrze przy ograniczonej liczbie cech.
    + Szybki w implementacji.

    _Wady:_
    + Zakłada liniową zależność między cechami a logarytmem prawdopodobieństwa, co nie zawsze jest prawdziwe.
    + Wrażliwy na brakujące dane i silnie skorelowane cechy.




_Random Tree Forest_ - Został wybrany jako rozwiązanie tego problemu.
Wybrana koncepcja nosi nazwę Random Forestx (Lasy Losowe). Metoda polega na budowaniu drzew decyzyjnych. Drzewa decyzyjne reprezentują schematycznie proces podejmowania decyzji
pod pewnymi warunkami. Algorytm dąży do tego aby wyodrębnić poszczególne poprzez przechodzenie przez konkretne warunki. Idealne drzewo decyzyjne po danej sekwencji warunków
zawiera tylko jedną klase wyboru w "liściu". Algorytm Random Forests (RFS) polega na stworzeniu n drzew decyzyjnych. Dla każdego drzewa wybieramy losowo X punktów danych
ze zbioru uczącego i Y cech. Dla takiego drzewa tworzone jest niezależne drzewo decyzyjne. Następnie każde drzewo dostaje losową liczbę obserwacji ze zwracaniem. 
Każde ma także ten sam zbiór wejściowych cech, ale ostatecznie wybierany jest inny wylosowany podzbiór cech. Na końcu dokonujmy prognozowania dla każdego drzewa zbudowanego
w pierwszym etapie, a ostateczny wynik (w przypadku klasyfikacji) jest rozpatrywany na podstawie głosowania większościowego. W przypadku regresji możemy wziąć na przykład
przewidywaną średnią wartość ze wszystkich drzew.

      _Zalety RFS_:

    + Lasy losowe są bardzo skuteczne w wielu zadaniach, takich jak klasyfikacja i regresja, zwłaszcza gdy dane mają skomplikowane zależności i dużo szumu.

    + Dzięki losowemu wybieraniu próbek danych i cech, las losowy zapobiega przeuczeniu, które często występuje w pojedynczych drzewach decyzyjnych.

    + Lasy losowe mogą dobrze działać nawet wtedy, gdy zestaw danych zawiera bardzo dużo zmiennych.

    + Las losowy automatycznie oblicza, jak ważna jest każda cecha w przewidywaniu wyniku, co pomaga w analizie i interpretacji danych.

    + Koncepcja nie zakłada liniowej zależności między cechami a wynikami, co czyni ją uniwersalną w przypadku danych nieliniowych.

      _Wady RFS:_

    + Trudno jest interpretować wyniki lasów losowych, ponieważ są one wynikiem złożonej agregacji wielu drzew decyzyjnych. Nie można wyciągać prostych wniosków, jak w przypadku pojedynczego drzewa decyzyjnego.

    + Trening lasów losowych może być czasochłonny i wymaga dużo pamięci, zwłaszcza gdy liczba drzew i cech jest bardzo duża.


**3.Opis wybranej koncepcji**

Wybrany został algorytm RFS ponieważ wady i zalety zestawionych metod przedstawiały RFS w najlepszym świetle,
biorąc pod uwagę nieliniowość, ryzyko przeuczenia, oraz duży zbiór cech połączony z ograniczoną liczbą danych do nauki modelu.
Wpływ miało również przeprowadzone badanie, z którego wynikła wyższość modelu RFS nad Gradient Boostingiem i C5.0. 
link do artykułu: https://www.mecs-press.org/ijisa/ijisa-v11-n7/IJISA-V11-N7-3.pdf


Do Przeprowadzenia predykcji meczu, potrzebne są dane wejściowe:

'HomeTeam' - Drużyna grająca u siebie

'AwayTeam' - Drużyna grająca na wyjeździe

'PH5H' - Forma drużyny, która gra u siebie z 5 ostatnich meczy u siebie

'PA5A' - Forma drużyny, która gra na wyjeździe z 5 ostatnich meczy na wyjeździe

'AAvgST' - Średnie ilość celnych strzałów drużyny grającej na wyjeździe

'HAvgST' - Średnie ilość celnych strzałów drużyny grającej u siebie

'PA5' - Forma drużyny grającej na wyjeździe z 5 ostatnich meczy

'PH5' - Forma drużyny grającej u siebie z 5 ostatnich meczy

'AvgHTHG' - Średnia liczba goli strzelonych grając u siebie drużyny, która gra u siebie

'AvgATAG' - Średnia liczba goli strzelonych grając na wyjeździe drużyny, która gra na wyjeździe

'Avg5HTG' - Średnia goli z ostatnich 5 meczy drużyny grającej u siebie

'Avg5ATG' - Średnia goli z ostatnich 5 meczy drużyny grającej na wyjeździe

'HAvgY' - Średnia liczba żółtych kartek, drużyny grającej u siebie

'AAvgY' - Średnia liczba żółtych kartek, drużyny grającej na wyjeździe

'HAvgS' - Średnia liczba oddanych strzałów, drużyny grającej u siebie

'AAvgS' - Średnia liczba oddanych strzałów, drużyny grającej na wyjeździe

Cechy zostały dobrane intuicyjnie oraz ze względu na dużą wagę (powinny mieć największy wpływ na rezultat)
Napisany program samodzielnie pobiera bazę danych oblicza wartości cech dla danych drużyn, tworzy model i na jego podstawie przewiduje
prawdopodobieństwo wygranej drużyny grającej u siebie, drużyny grającej na wyjeździe lub remisu.

Program zawiera:
model o skuteczności 60%, którego można użyć do predykcji
możliwość wytrenowania modelu na podstawie wszystkich meczy dotychczas
możliwość predykcji rezultatu meczu za pomocą wytrenowanego wcześniej modelu
możliwość sprawdzenia parametrów modelu (accuracy, confusion matrix)
intuicyjne menu które wyraźnie wskazuję co się dzieje w programie

Program zapobiega błędom, jednak aby działał poprawnie potrzebne jest połączenie internetowe (do pobrania bazy danych)

