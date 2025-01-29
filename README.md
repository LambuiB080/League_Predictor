**1. Opis rzeczywistego problemu.**

Projekt League_Predictor skupia się na przewidywaniu meczy ligi angielskiej Premier League. W projekcie utworzony został model przewidujący mecze na podstawie wcześniejszych
statystyk poszczególnych zespołów takich jak forma drużyny, średnie gole, średnie żółte kartki, czy celne strzały na bramke. Wynikiem jest prawdopodobieństwo rezultatu danego meczu. 
Projekt może posłużyć do pomocy w obstawianiu zakładów bukmacherskich.

**2. State of Art**

W koncepcji przewidywania meczy należy rozważyć 3 koncepcje:

_Regresja logistyczna_ - algorytm uczenia maszynowego używany do zadań klasyfikacyjnych. Celem jest określenie prawdopodobieństwa, że dane zdarzenie (u nas - wynik meczu) należy do danej klasy (zwycięstwo, remis, porażka). W tym celu wykorzystywana jest tzw. funkcja sigmoidalna - przyjmuje ona dane wejściowe jako zmienne niezależne a wynikiem jej działania jest wartość prawdopodobieństwa (zmienna zależna) między 0 a 1. Stosując ten algorytm, konieczne jest spełnienie kilku założeń takich jak: niezależne obserwacje, duża próbka danych, brak wartości "odstających" od reszty i liniowa zależność między zmiennymi niezależnymi i funkcją logitową. Algorytm realizuje się w trzech krokach:
1. Wyznaczenie funkcji liniowej z zależnej od zbioru cech wejściowych X za pomocą regresji liniowej. Kluczowe są tu wagi przypisywane do każdej cechy.
2. Przekształcenie funkcji z z użyciem funkcji sigmoidalnej, która przekształca funkcję ciągłą na wartość prawdopodobieństwa.
3. Obliczenie funkcji kosztu.
4. Optymalizacja wag poprzez minimalizację funkcji kosztu.
5. Przewidywanie wyniku.

Zalety:
- Prostowa i łatwa interpretacja
- Dobrze działa na małych i średnich zbiorach danych
- Możliwość skalowania do większych problemów (np. Softmax regression)

Wady: 
- Wrażliwość na wartości odstające
- Gorsze działanie dla silnie skorelowanych cech
- Gorsze radzenie sobie z dużą liczbą cech.

_Gradient Boosting_ - algorytm trenuje model sekwencyjnie (nazywamy to boostingiem) - każdy nowy model próbuje poprawić poprzedni m.in. zminimalizować funkcję strat. W pierwszym kroku obliczany jest błąd, czyli różnica między wartościami rzeczywistymi a przewidywaniami modelu. Następnie nowy model jest trenowany - przewiduje przy tym przyszły błąd i tym samym uczy się poprawiać błędy poprzedniego modelu. Prognozy z każdej iteracji są dodawane do dotychczasowego modelu. Proces ten trwa do momentu, gdy błąd przestanie się zmniejszać lub zostanie osiągnięta zadana liczba iteracji.

ZALETY:
- wysoka dokładność - wyniki są lepsze niż dla regresji logistycznej,
- obsługa różnych typów danych,
- dobra odporność na przeuczenie.

WADY:
- wysoka złożoność obliczeniowa,
- Wrażliwość na wartości odstające
- Wymaga dobrego dostrojenia parametrów.

_Random Forest_ - Został wybrany jako rozwiązanie tego problemu.
Wybrana koncepcja nosi nazwę Random Forestx (Lasy Losowe). Metoda polega na budowaniu drzew decyzyjnych. Drzewa decyzyjne reprezentują schematycznie proces podejmowania decyzji
pod pewnymi warunkami. Algorytm dąży do tego, aby wyodrębnić poszczególne klasy poprzez przechodzenie przez konkretne warunki. Idealne drzewo decyzyjne po danej sekwencji warunków
zawiera tylko jedną klase wyboru w "liściu". Algorytm Random Forests (RFS) polega na stworzeniu n drzew decyzyjnych. Dla każdego drzewa wybieramy losowo X punktów danych
ze zbioru uczącego i Y cech. Dla takiego drzewa tworzone jest niezależne drzewo decyzyjne. Następnie każde drzewo dostaje losową liczbę obserwacji ze zwracaniem. 
Każde ma także ten sam zbiór wejściowych cech, ale ostatecznie wybierany jest inny wylosowany podzbiór cech. Na końcu dokonujmy prognozowania dla każdego drzewa zbudowanego
w pierwszym etapie, a ostateczny wynik (w przypadku klasyfikacji) jest rozpatrywany na podstawie głosowania większościowego. W przypadku regresji możemy wziąć na przykład
przewidywaną średnią wartość ze wszystkich drzew.

#### Zalety RFS:
- Lasy losowe są bardzo skuteczne w wielu zadaniach, takich jak klasyfikacja i regresja, zwłaszcza gdy dane mają skomplikowane zależności i dużo szumu.
- Dzięki losowemu wybieraniu próbek danych i cech, las losowy zapobiega przeuczeniu, które często występuje w pojedynczych drzewach decyzyjnych.
- Lasy losowe mogą dobrze działać nawet wtedy, gdy zestaw danych zawiera bardzo dużo zmiennych.
- Las losowy automatycznie oblicza, jak ważna jest każda cecha w przewidywaniu wyniku, co pomaga w analizie i interpretacji danych.
- Koncepcja nie zakłada liniowej zależności między cechami a wynikami, co czyni ją uniwersalną w przypadku danych nieliniowych.

#### Wady RFS:
- Trudno jest interpretować wyniki lasów losowych, ponieważ są one wynikiem złożonej agregacji wielu drzew decyzyjnych. Nie można wyciągać prostych wniosków, jak w przypadku pojedynczego drzewa decyzyjnego.
- Trening lasów losowych może być czasochłonny i wymaga dużo pamięci, zwłaszcza gdy liczba drzew i cech jest bardzo duża.


**3.Opis wybranej koncepcji**

Wybrany został algorytm RFS ponieważ wady i zalety zestawionych metod przedstawiały RFS w najlepszym świetle,
biorąc pod uwagę nieliniowość, ryzyko przeuczenia, oraz duży zbiór cech połączony z ograniczoną liczbą danych do nauki modelu.
Wpływ miało również przeprowadzone badanie, z którego wynika wyższość modelu RFS nad Gradient Boostingiem i C5.0. 
Link do artykułu: https://www.mecs-press.org/ijisa/ijisa-v11-n7/IJISA-V11-N7-3.pdf


Do przeprowadzenia predykcji meczu, potrzebne są dane wejściowe:

+ 'HomeTeam' - drużyna grająca u siebie

+ 'AwayTeam' - drużyna grająca na wyjeździe

+ 'PH5H' - forma drużyny, która gra u siebie z 5 ostatnich meczy u siebie

+ 'PA5A' - forma drużyny, która gra na wyjeździe z 5 ostatnich meczy na wyjeździe

+ 'AAvgST' - średnia ilość celnych strzałów drużyny grającej na wyjeździe

+ 'HAvgST' - średnia ilość celnych strzałów drużyny grającej u siebie

+ 'PA5' - forma drużyny grającej na wyjeździe z 5 ostatnich meczy

+ 'PH5' - forma drużyny grającej u siebie z 5 ostatnich meczy

+ 'AvgHTHG' - średnia liczba goli strzelonych drużyny, która gra u siebie

+ 'AvgATAG' - śąrednia liczba goli strzelonych drużyny, która gra na wyjeździe

+ 'Avg5HTG' - średnia goli z ostatnich 5 meczy drużyny grającej u siebie

+ 'Avg5ATG' - średnia goli z ostatnich 5 meczy drużyny grającej na wyjeździe

+ 'HAvgY' - średnia liczba żółtych kartek drużyny grającej u siebie

+ 'AAvgY' - średnia liczba żółtych kartek drużyny grającej na wyjeździe

+ 'HAvgS' - średnia liczba oddanych strzałów drużyny grającej u siebie

+ 'AAvgS' - średnia liczba oddanych strzałów drużyny grającej na wyjeździe

Cechy zostały dobrane intuicyjnie oraz ze względu na dużą wagę (duża waga powinna mieć największy wpływ na rezultat).
Napisany program samodzielnie pobiera bazę danych, oblicza wartości cech dla danych drużyn, tworzy model i na jego podstawie przewiduje
prawdopodobieństwo wygranej drużyny grającej u siebie, drużyny grającej na wyjeździe lub remisu.

Program zawiera:
+ model o skuteczności 60%, którego można użyć do predykcji
+ możliwość wytrenowania modelu na podstawie wszystkich meczy dotychczas rozegranych
+ możliwość predykcji rezultatu meczu za pomocą wytrenowanego wcześniej modelu
+ możliwość sprawdzenia parametrów modelu (accuracy, confusion matrix)
+ intuicyjne menu które wyraźnie wskazuje, co się dzieje w programie.

Program zapobiega błędom, jednak aby działał poprawnie potrzebne jest połączenie internetowe (do pobrania bazy danych).

ŹRÓDŁA:
- https://www.geeksforgeeks.org/understanding-logistic-regression/
- https://www.geeksforgeeks.org/ml-gradient-boosting/
- https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/
