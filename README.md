# Implementacja agenta SARSA — Marek Borkowski

Autorska implementacja agenta **SARSA** przeznaczona do współpracy z biblioteką
[reinforced-lib](https://reinforced-lib.readthedocs.io/en/latest/index.html) w scenariuszach `ns-3`.
Projekt zawiera minimalny, ale kompletny zestaw plików do uruchomienia eksperymentów oraz łatwej
podmiany integracji w przykładowym projekcie `reinforced-lib`.

---

## Zawartość repozytorium

```
├─ sarsa.py # Implementacja agenta SARSA (on-policy, Q-tablica)
├─ main.py # Skrypt uruchamiający symulacje/eksperymenty
└─ ext.py # Plik do podmiany w reinforced-lib/examples/ns-3-ra
```

---

## Wymagania

- Python **3.9+** (testowane z CPython 3.9–3.12)
- **ns-3** (wersja zgodna z integracją reinforced-lib)
- **reinforced-lib** – instalacja wg dokumentacji:
  https://reinforced-lib.readthedocs.io/en/latest/index.html

> **Wskazówka:** użyj wirtualnego środowiska (venv/conda) i trzymaj spójne wersje zależności
z przykładami `reinforced-lib`.

---

## Instalacja (skrót)

1. Zainstaluj `reinforced-lib` zgodnie z oficjalną dokumentacją.
2. Zbuduj/udostępnij `ns-3` i zapamiętaj jego ścieżkę (parametr `--ns3Path`).
3. (Opcjonalnie) Podmień integrację:
   - skopiuj `ext.py` do:
     - Linux/macOS: `reinforced-lib/examples/ns-3-ra/ext.py`
     - Windows/WSL: `reinforced-lib\examples\ns-3-ra\ext.py`

---

## Szybki start

Uruchom przykładową symulację z poziomu przykładów `reinforced-lib`:

```bash
python $REINFORCED_LIB/examples/ns-3-ccod/main.py \
  --agent "SARSA" \
  --ns3Path "$YOUR_NS3_PATH"
```


Parametry uruchomienia (przykłady)

Najczęściej używane opcje (mogą różnić się zależnie od wersji przykładów w reinforced-lib):

--agent – nazwa agenta (SARSA)

--ns3Path – ścieżka do zbudowanego ns-3

--alpha – współczynnik uczenia (np. 0.1)

--gamma – współczynnik dyskontowania (np. 0.25)

--epsilon – eksploracja ε-greedy (np. 0.05)

--episodes, --steps – rozmiar eksperymentu

--outDir – katalog na logi/wyniki

Przykład:
python main.py \
  --agent SARSA \
  --ns3Path "/opt/ns-3-dev" \
  --alpha 0.1 --gamma 0.25 --epsilon 0.05 \
  --episodes 50 --steps 2000 \
  --outDir "./results/sarsa_1"
Wyniki i logi

Metryki (np. przepustowość, nagrody, (stan, akcja)) zapisywane są do plików w katalogu --outDir.

Dodatkowe logowanie (np. snapshot Q-tablicy co N kroków) możesz włączyć bezpośrednio w sarsa.py.

Wykresy wygodnie wygenerujesz w Pythonie/Matplotlib lub narzędziami z przykładów reinforced-lib.

Jak działa ta implementacja?

On-policy SARSA: aktualizacja Q wykorzystuje faktycznie wybraną akcję w kolejnym stanie (a′),
zgodnie z polityką ε-greedy (eksploracja/eksploatacja).

Parametry alpha, gamma, epsilon są łatwo strojalne; możesz dodać np. „decaying ε”
albo inne strategie eksploracji.

Interfejs agenta jest zgodny z warstwą eksperymentalną reinforced-lib, dzięki czemu integracja
z ns-3 jest prosta i przewidywalna.

Najczęstsze problemy (troubleshooting)

ns-3 nie znaleziony / błędna ścieżka
Upewnij się, że --ns3Path wskazuje na zbudowane ns-3 (np. ./waf build).

Konflikt wersji pakietów/Pythona
Użyj czystego venv/conda i zainstaluj reinforced-lib od nowa wg dokumentacji.

Brak wyników w outDir
Sprawdź prawa zapisu i logi w konsoli pod kątem wyjątków.

Wkład / rozwój

Pull requesty mile widziane: poprawki, rozszerzenia (soft updates, decaying ε, alternatywne
reprezentacje stanów), uspójnienia z aktualnym API reinforced-lib.

Licencja

Dodaj wybraną licencję (np. MIT/Apache-2.0).

Cytowanie

Jeśli użyjesz tej implementacji w pracy naukowej, rozważ dodanie odwołania do repozytorium
oraz dokumentacji reinforced-lib.
@misc{borkowski_sarsa_ns3,
  author       = {Borkowski, Marek},
  title        = {Implementacja agenta SARSA dla reinforced-lib i ns-3},
  year         = {2025},
  howpublished = {\url{https://reinforced-lib.readthedocs.io/en/latest/index.html}}
}
