# TSP Casablanca — Algorithme génétique (AG) et colonie de fourmis (ACO)

TSP fermé sur des points GPS réels de Sidi Maarouf / Aïn Chock / Bouskoura (Casablanca), départ et
retour au point #0 (Gare Casa Oasis), en minimisant la **distance routière réelle OSRM**.
Deux applications desktop (Python 3.13, Tkinter, carte Folium intégrée) :

| Lancement | Application |
|---|---|
| `python main.py` | **Algorithme génétique** (AG) : réglage manuel, Optuna, agents IA Gemini, comparaison |
| `python main_aco.py` | **Colonie de fourmis** (ACO, Ant System) : 40/60/80 points, réglage manuel, Optuna (BC2), KerasTuner (BC3), SMA à 3 agents (BC4) |

## Installation rapide (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env        # puis renseigner GEMINI_API_KEY dans .env (jamais versionné)
.venv\Scripts\python.exe main.py       # AG
.venv\Scripts\python.exe main_aco.py   # ACO
```

Optionnel pour BC3 KerasTuner (ACO) :
`.venv\Scripts\python.exe -m pip install keras-tuner six torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple`
Tests : `.venv\Scripts\python.exe -m pytest -q`. La matrice OSRM des 40 points est fournie en cache
(`data/cache/`) : l'appli démarre sans appel OSRM.

---

# Application AG — Algorithme génétique sur réseau routier réel

Application desktop (Python 3.13, Tkinter) qui résout un **TSP fermé** sur les 40 points GPS de
`data/input/points_tsp40.csv`. Le départ et le retour se font au point **#0 Gare Casa Oasis**.
L'objectif minimisé est la **distance routière OSRM** réelle, pas une distance euclidienne.

## Ce que contient la V1

| Quadrant | Contenu |
|---|---|
| 1 · Configuration | Réglage manuel : population 10/50/100, tournoi ou roue biaisée, mutation 0.01–0.20 (ou valeur libre), croisement 0.2/0.5/0.8, générations 100/500/1000, élitisme, graine. Boutons Lancer, Pause/Reprise, Pas à pas, Arrêter, barre de progression. |
| 2 · Performances | Courbe de convergence (meilleure distance globale), distance par génération (meilleur et moyenne), indicateurs : distance, temps, génération du meilleur, amélioration, fitness. |
| 3 · Plan cartésien | Projection locale en km (X = Est, Y = Nord) centrée sur le départ, parcours fléché, retour en pointillé, zoom et déplacement (barre Matplotlib), distances par étape en option. |
| 4 · Carte réelle | Carte Folium/Leaflet **intégrée** dans la fenêtre (Edge WebView2) : 40 marqueurs, puis trajet routier réel (OSRM Route) dans l'ordre de l'AG, avec le détail de chaque étape. |

**Bouton ② Réglage Optuna** (V2) : il ouvre une fenêtre d'optimisation multiobjectif, décrite plus bas.
**Bouton ③ Agents IA** : système de trois agents Gemini avec validation humaine, décrit plus bas.
**Bouton ⚖ Comparer** : 10 répétitions par mode, statistiques et boîtes à moustaches (décrit plus bas).
Le bouton **Exports** est présent mais désactivé (à venir).
Chaque simulation est enregistrée en JSON dans `data/results/`, avec son mode : `manuel`, `optuna`,
`agents` ou `agents (modifié)`.

## Réglage automatique Optuna (bouton ②)

- **Espace de recherche** : population {10, 50, 100}, sélection {tournoi, roulette}, mutation dans
  [0,01 ; 0,20] (échelle log, arrondie à 4 décimales), croisement {0,2 ; 0,5 ; 0,8},
  générations {100, 500, 1000}. L'élitisme est toujours actif.
- **Deux objectifs distincts** : distance (km) et temps d'exécution de l'AG (s), via
  `optuna.create_study(directions=["minimize", "minimize"])` et `TPESampler`. Avec plusieurs
  objectifs, `TPESampler` applique l'algorithme MOTPE (Optuna 5 n'a plus de `MOTPESampler` séparé).
  Les km et les secondes ne sont jamais additionnés.
- **Évaluation sur plusieurs graines** : chaque trial exécute l'AG sur **3 graines** (réglable de
  1 à 10). Optuna reçoit la **moyenne** de la distance et la **moyenne** du temps. Le tableau
  affiche aussi l'écart-type de la distance, et le détail par graine est enregistré. La raison est
  expliquée dans la section suivante.
- **Conditions comparables** : même matrice OSRM et **mêmes graines d'évaluation pour tous les
  trials** (graine de base, base+1, base+2 ; 42, 43, 44 par défaut). La graine du sampler est
  la graine de base.
- **Trials aléatoires initiaux** : `n_startup_trials = max(2, min(10, trials // 3))`. Avec la valeur
  par défaut d'Optuna (10), une étude de 10 trials serait entièrement aléatoire. Ici, 10 trials
  donnent 3 trials aléatoires puis 7 guidés par TPE.
- **Affichage en direct** : trial en cours, paramètres, distance, temps, barre de progression.
  Le tableau peut être trié et sert d'historique.
- **Front de Pareto** : les solutions non dominées sont marquées ★ et tracées sur le graphe
  distance/temps.
- **Configuration recommandée** : le point du front le plus proche du point idéal, après
  normalisation min-max des deux objectifs sur tous les trials (poids égaux). Le texte
  d'explication chiffre le compromis par rapport à la meilleure distance et à la solution
  la plus rapide.
- **Boutons d'application** :
  - « Appliquer la configuration et lancer la simulation » remplit le quadrant 1 avec la
    configuration recommandée et la graine de base, puis lance la simulation. La distance
    obtenue est exactement celle de cette graine dans le détail du trial, et non la moyenne
    affichée dans le tableau.
  - « Appliquer sans lancer » remplit seulement le quadrant 1.
  - « Appliquer le trial sélectionné » applique n'importe quel trial du tableau.
- Chaque étude est enregistrée dans `data/results/optuna_<date>.json` (tous les trials, front de
  Pareto, recommandation et explication).
- **Reproductibilité** :
  - avec une graine fixe, les trials aléatoires initiaux sont identiques, et une configuration
    donnée produit toujours exactement la même distance ;
  - les trials guidés par TPE peuvent varier légèrement d'une étude à l'autre, car ils dépendent
    du temps mesuré, qui est un temps réel.
- **Pendant une étude**, la simulation est bloquée, et inversement, pour ne pas fausser la mesure
  des temps. « Arrêter » interrompt le trial en cours, qui est alors exclu des résultats.
- **Temps mesurés** : ce sont ceux du moteur AG seul. Ils varient selon la charge de la machine.
  Une simulation affichée dans l'interface est un peu plus lente, à cause de la mise à jour des
  graphiques.

### Pourquoi évaluer chaque trial sur plusieurs graines

L'algorithme génétique est **stochastique** : la même configuration, lancée avec deux graines
différentes, ne donne pas la même distance. Sur ce problème, l'écart est important. Dans l'étude
à 3 graines, une même configuration varie de plusieurs kilomètres d'une graine à l'autre :

| Trial | Configuration | Distance par graine (42 / 43 / 44) |
|---|---|---|
| #29 | pop 10, tournoi, 1000 gén. | 100,8 / 115,5 / 99,1 km (16 km d'écart) |
| #18 | pop 100, tournoi, 500 gén. | 101,2 / 90,2 / 92,7 km (11 km d'écart) |
| #16 | pop 50, tournoi, 1000 gén. | 88,2 / 89,1 / 91,9 km (4 km d'écart) |

Cette variabilité est du même ordre que les écarts entre configurations. Avec **une seule
graine**, chaque trial est jugé sur un seul tirage : Optuna compare alors autant la chance que la
qualité, et la configuration retenue est en partie celle qui a eu le tirage le plus favorable.
C'est le **biais du gagnant** (*winner's curse*) : la performance annoncée est optimiste, et le
classement des configurations est peu fiable.

Faire la **moyenne sur 3 graines** :
- réduit le bruit, puisque l'écart-type de la moyenne est divisé par √3 ≈ 1,7 ;
- pénalise les configurations instables : une configuration qui n'est bonne que sur une graine
  voit sa moyenne remonter ;
- fournit un écart-type par trial, affiché dans le tableau.

Le coût est un temps d'étude multiplié par 3. Les graines d'évaluation (42, 43, 44) sont
**distinctes** de celles de la comparaison finale (1000 à 1009). La configuration est donc jugée
sur des tirages qui n'ont pas servi à la choisir.

**Ce qu'on a observé.** Les distances moyennes sont mesurées sur les graines 1000 à 1009, communes
aux trois modes :

| Étude Optuna | Recommandation | Valeur annoncée par l'étude | Moyenne sur 10 graines de test |
|---|---|---|---|
| 10 trials, **1 graine** (42) | pop 10, roulette, mut 0,1347, crois 0,8, 1000 gén. | 110,61 km | **112,22 ± 7,35 km** (pire 128,15) |
| 30 trials, **3 graines** (42–44) | pop 50, tournoi, mut 0,1801, crois 0,2, 500 gén. | 92,88 ± 1,53 km | **91,33 ± 3,78 km** (pire 99,71) |

Avec les mêmes graines de test, le mode Optuna passe de 112,22 km à 91,33 km de moyenne, avec une
dispersion divisée par deux. Il devient proche du mode multi-agents (90,27 km) tout en étant
nettement plus rapide.

*Honnêteté sur ce résultat* : la nouvelle étude change **deux** choses à la fois, 3 graines
**et** 30 trials au lieu de 10. Cette expérience ne permet donc pas d'attribuer tout le gain aux
graines multiples. Ce qu'elle montre : la valeur annoncée par l'étude multi-graines (92,88 km)
est **cohérente avec la moyenne sur 10 graines de test** (91,33 km), et le classement repose sur
des moyennes, pas sur un tirage unique.

## Installation (Windows)

```powershell
cd TSP_40
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prérequis : le runtime **Microsoft Edge WebView2** (déjà présent sur Windows 10/11 à jour) et
une connexion Internet au premier lancement, pour la matrice OSRM et les fonds de carte.

## Lancement

```powershell
.venv\Scripts\python.exe main.py
```

Attention : sur ce poste, la commande `python` seule lance Python 3.11. Utilisez donc le
Python du `.venv`, ou activez d'abord l'environnement (`.venv\Scripts\activate`).
Dans VS Code, choisissez l'interpréteur `.venv\Scripts\python.exe` (Ctrl+Maj+P, puis
« Python: Select Interpreter ») et lancez `main.py`.

Tests :

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Choix techniques

- **Matrice OSRM** : un seul appel OSRM Table (40×40, distances en mètres et durées). La matrice
  est **asymétrique** (A→B ≠ B→A) et mise en cache dans `data/cache/osrm_matrix_<empreinte>.json`.
  Au lancement suivant, l'application relit le cache sans aucun appel réseau. Si OSRM est
  indisponible, un message d'erreur s'affiche et aucune distance euclidienne n'est utilisée à la place.
- **OSRM Route** avec `continue_straight=false` : demi-tour autorisé aux points de passage, comme
  dans OSRM Table. Ainsi, la longueur du trajet tracé est identique à la distance optimisée.
  Sans cette option, un écart de l'ordre de +10 km apparaît.
- **AG** : un individu est une permutation des 39 points autres que le départ. Le départ ne fait
  donc jamais partie du génome et ne peut pas bouger. Croisement OX, mutation par inversion,
  sélection par tournoi (k=3) ou roue biaisée, élitisme (2 individus), fitness = 1/(1 + km).
  La population est évaluée de façon vectorisée avec NumPy, et la validité de la tournée finale
  est vérifiée.
- **IHM non bloquante** : l'AG tourne dans un thread et envoie ses messages par une `queue`,
  lue par la boucle Tk avec `after()`. Pause, pas à pas et arrêt passent par des `threading.Event`.
- **Carte intégrée sous Python 3.13** : `cefpython3` n'existe pas pour 3.13 et `tkinterweb`
  n'exécute pas le JavaScript de Leaflet. La solution retenue est la suivante :
  - un processus enfant `visualization/map_host.py` ouvre une fenêtre **pywebview / Edge WebView2**
    (Chromium) sans bordure ;
  - Tkinter l'incruste dans le quadrant 4 avec `SetParent` (Win32) et la redimensionne avec le panneau ;
  - les cartes sont servies par un petit serveur HTTP local (127.0.0.1), car les serveurs de tuiles
    OpenStreetMap refusent les pages `file://`.

## Agents IA Gemini (bouton ③)

**Configuration.** Copiez `.env.example` en `.env` et renseignez `GEMINI_API_KEY`. Le fichier `.env`
est exclu par `.gitignore`, et la clé n'apparaît ni dans le code, ni dans les journaux, ni dans les
résultats. Sans clé, le bouton ③ affiche un message expliquant quoi faire.

**SDK** : `google-genai`, le SDK officiel actuel. L'ancien `google-generativeai`, déprécié, n'est pas utilisé.

**Choix du modèle, vérifié et non supposé.**
- `models.list()` a été interrogé, puis des appels réels ont été faits avec cette clé le 16/11/2026.
- `gemini-2.5-*` sont listés mais refusés (404 « no longer available to new users »).
- `gemini-3.8-flash` et `3.7-flash` existent mais étaient souvent saturés (503/504).
- `gemini-3.6-flash`, `3.5-flash`, `3.5-flash-lite` et `3.1-flash-lite` répondent.
- Au lancement, un mini-appel (20 s maximum par modèle) choisit le premier modèle disponible dans
  `PREFERRED_MODELS` (`agents/gemini_client.py`). Un modèle qui échoue ensuite est écarté pour la
  session (repli automatique).
- `GEMINI_MODEL` dans `.env` force un modèle précis.
- Le modèle réellement utilisé et le résultat des tests de disponibilité sont affichés et enregistrés.

**Les trois agents** (`agents/`) échangent du JSON :
1. **Web Research** (`web_agent.py`, `web_search.py`) fait une **vraie recherche DuckDuckGo**
   avec la bibliothèque `ddgs`, sans clé API. Elle remplace Google Search grounding, qui est
   refusé (429) sur le palier gratuit de Gemini.
   1. Gemini rédige **3 à 5 requêtes**, validées par Python. S'il échoue, des requêtes par défaut
      sont utilisées, et c'est indiqué.
   2. Python exécute les requêtes : 5 résultats chacune, 2 tentatives par requête, une pause de 1 s
      entre les requêtes, puis dédoublonnage par URL. Il collecte **titre, URL et extrait**, numérotés
      `[S1]`, `[S2]`…
   3. Gemini reçoit **uniquement ces sources** et doit les citer par leur identifiant.
   4. **Python vérifie les citations** :
      - chaque identifiant cité existe ;
      - la justification contient au moins une citation ;
      - aucune URL n'apparaît hors de la liste.

      Une réponse non conforme, par exemple une source `[S9]` inventée, est renvoyée une fois au
      modèle avec le motif, puis rejetée. Le modèle doit écrire « non couvert par les sources »
      quand les extraits ne justifient pas un paramètre.
   5. **Fenêtre de validation** : la carte de l'agent a deux onglets, « Justification » et
      « Sources (N · K citées) ». Les sources citées sont marquées ★ et triées en tête. Au survol ou
      au clic, une info-bulle affiche le titre, l'URL, la requête d'origine et l'extrait transmis à
      Gemini. Un double-clic ouvre la page.
   6. Tout est enregistré dans la session (`agents_*.json`) : requêtes, journal de recherche, sources
      avec extraits et marquage « citée ».

   **Limite.** Python vérifie que les sources citées existent et ont bien été fournies, mais pas
   que l'extrait soutient réellement l'affirmation : ce contrôle reste humain, et l'info-bulle sert
   à le faire. L'arbitre reçoit les extraits des sources citées et écarte les citations hors sujet
   (lors du test réel, il a écarté un fil de forum sur un problème à 40 000 variables).

   **Si DuckDuckGo ne renvoie rien** (réseau coupé, limitation de débit), l'agent ne prétend pas
   avoir cherché :
   - il répond sans web, avec la consigne de ne citer aucune référence ;
   - il est marqué « RECHERCHE WEB NON EFFECTUÉE — sans sources vérifiées » ;
   - l'arbitre est informé que cette proposition n'est pas étayée.
2. **Scientifique** (`semantic_agent.py`) fait une analyse théorique : taille de l'espace, asymétrie,
   convergence prématurée, interactions entre hyperparamètres. Sa sortie est structurée par un schéma JSON.
3. **Arbitre** (`arbiter_agent.py`) reçoit les deux propositions, ainsi que les différences calculées
   par Python. Il produit :
   - les convergences ;
   - les divergences, avec une décision et une raison pour chacune ;
   - les arguments écartés ;
   - un consensus et la configuration finale.

Le **contexte** transmis aux agents est calculé à partir des vraies données : 40 points, statistiques
de la matrice OSRM (asymétrie, distances), opérateurs réellement implémentés et temps mesurés.

**Validation Python** (`agents/schemas.py`) : tous les champs sont obligatoires et doivent être dans
le domaine autorisé (population {10, 50, 100}, sélection, croisement {0,2 ; 0,5 ; 0,8}, générations
{100, 500, 1000}, booléen pour l'élitisme, justification non vide). Aucune correction n'est faite en
silence : une réponse invalide est renvoyée une fois au modèle avec le motif de l'erreur, puis, si
elle reste invalide, l'agent est marqué en erreur.

**Validation humaine.** L'AG n'est jamais lancé automatiquement. La fenêtre affiche :
- les trois propositions ;
- le tableau des divergences, avec la décision de l'arbitre ;
- les justifications ;
- la configuration finale.

Trois actions sont possibles :
- **Accepter** : la configuration est appliquée et la simulation lancée ;
- **Modifier** : les champs sont déverrouillés, la saisie est validée de la même façon et le mode
  devient `agents (modifié)` ;
- **Refuser** : rien n'est appliqué.

Chaque session est enregistrée dans `data/results/agents_<date>.json` (réponses brutes, sources,
décision humaine, paramètres appliqués). La simulation lancée y fait référence.

## Comparaison des trois modes (bouton ⚖ Comparer — sections 8 et 9)

**Sources des configurations (réelles uniquement).** Chaque mode utilise une configuration qui
existe vraiment :
- **Manuel** : le formulaire actuel du quadrant 1.
- **Optuna** : la recommandation d'une étude enregistrée (`optuna_*.json`), au choix.
- **Multi-agents** : la configuration finale de l'arbitre d'une session (`agents_*.json`), ou celle
  appliquée après modification humaine.

La source choisie est affichée et enregistrée. Un mode sans source (par exemple, aucune étude
Optuna lancée) est exclu et apparaît « — » dans les tableaux : rien n'est inventé.

**Protocole.**
- N répétitions par mode (10 par défaut).
- Graines `base, base+1, …`, **identiques pour les trois modes**.
- Même matrice OSRM et même moteur.
- Exécutions entrelacées (graine 1 : manuel, Optuna, agents ; graine 2 : …), pour que la charge de la
  machine pèse de la même façon sur les temps des trois modes.
- Exécution séquentielle, sans parallélisme, pour ne pas fausser les temps.
- Si la comparaison est arrêtée, seules les répétitions complètes, présentes pour tous les modes,
  sont conservées.

**Résultats affichés.**
- **Section 8 : exécution unique de référence** (première graine), avec la distance, le temps,
  le nombre de générations (et la génération du meilleur), la meilleure fitness et le taux
  d'amélioration.
- **Section 9 : statistiques sur N répétitions**, avec la meilleure, la moyenne et la pire distance,
  l'écart-type d'échantillon (n−1), le temps moyen ± écart-type, la génération moyenne du meilleur,
  l'amélioration moyenne et la meilleure fitness.
- **Boîtes à moustaches** de la distance et du temps : deux graphiques séparés, sans double axe,
  avec chaque exécution en point. Au survol, une info-bulle indique le mode, la répétition, la
  graine, la distance et le temps.
- Les couleurs des modes ont été validées pour le daltonisme et le contraste sur le fond sombre.

**Fichiers enregistrés.** `comparison_<date>.json` (configurations, sources, statistiques, référence,
toutes les exécutions) et `comparison_<date>_runs.csv` (une ligne par exécution, avec l'ordre de visite).

## Structure

```
main.py                     point d'entrée
config/settings.py          chemins, OSRM, choix de paramètres, thème
core/                       tsp_problem, genetic_algorithm, operators, fitness, validation
services/                   data_loader (CSV/XLSX + validation), osrm_service, distance_matrix, cache_manager
visualization/              folium_map, cartesian_map (projection locale), map_host (WebView2)
optimization/               optuna_optimizer (MOTPE, Pareto, recommandation), experiment_runner (comparaison)
agents/                     gemini_client, web_agent, web_search (DuckDuckGo), semantic_agent, arbiter_agent,
                            agent_manager, schemas, base
interface/                  main_window, quadrant_1..4, optuna_dialog, agents_dialog, comparison_dialog,
                            map_embed (Win32), theme
tests/                      83 tests pytest (données synthétiques, OSRM, Gemini et DuckDuckGo simulés ; 1 test Gemini
                            réel activé par TSP40_LIVE_GEMINI=1)
```

## Limites connues

- Le serveur OSRM public est fourni sans garantie et avec une limite d'usage. L'application n'y
  fait que 2 types d'appels (Table une seule fois, Route une fois par tournée, mis en cache).
  Un serveur personnel se configure avec `OSRM_BASE_URL` dans `.env`.
- La carte intégrée fonctionne sous Windows uniquement. Le bouton « Ouvrir dans le navigateur »
  sert de secours.

---

# Application ACO — colonie de fourmis (Ant System)

Seconde application, **indépendante de l'application AG** (qui reste lancée par `main.py`).
Même problème : TSP fermé sur les points de Sidi Maarouf / Aïn Chock / Bouskoura, départ et
retour au point n=0 (Gare Casa Oasis), distance **routière** OSRM (matrice asymétrique en cache).

## Lancement

```powershell
.venv\Scripts\python.exe main_aco.py
```

Option `--autorun` : lance automatiquement un run dès que la matrice est prête.
Journal : `data/app_aco.log`. Chaque run est enregistré dans `data/results/aco_run_*.json`.

## Taille du problème (40 / 60 / 80)

La combo box du quadrant 1 propose 40, 60 et 80 points. Les points sont lus dans
`data/input/points_tsp{N}.csv` (mêmes colonnes que le fichier de 40 points : `n, nom, lat, lon`,
identifiants 0..N−1, départ n=0). **Seul le fichier de 40 points existe aujourd'hui** : si l'on
choisit 60 ou 80, l'application affiche « Fichier de 60 points absent » et ne lance rien. Aucun
point GPS n'est inventé. Dès qu'un fichier est ajouté, la matrice OSRM N×N est demandée une fois
puis mise en cache (le nom du cache dépend des coordonnées, donc une matrice par taille).

## Algorithme (module `aco/ant_system.py`, indépendant de l'interface)

- η_ij = 1 / d_ij (d en km, distance routière OSRM). τ0 = m / L_nn sur tous les arcs (m fourmis,
  L_nn = tour du plus proche voisin, valeur recommandée par Dorigo).
- Chaque fourmi part du point 0 ; depuis i, elle choisit j non visité avec la probabilité
  τ_ij^α · η_ij^β / Σ τ_il^α · η_il^β, puis revient au point 0.
- Évaporation τ ← (1 − ρ)·τ, puis dépôt Q / L_k (Q = 1, L_k en km) sur les arcs de chaque fourmi.
- La matrice étant asymétrique, la phéromone est portée par les arcs **orientés** i→j.
- Le meilleur tour global est conservé à chaque cycle. Les m fourmis sont construites en
  parallèle avec NumPy (80 points × 1000 cycles ≈ 10 s sur ce poste).
- Graine aléatoire configurable : même graine = même résultat.
- Aucun appel OSRM pendant les cycles : seule la matrice en cache est utilisée. OSRM Route est
  appelé une fois, après le run, pour tracer le trajet réel (également mis en cache).

| Paramètre | Défaut | Bornes |
|---|---|---|
| Nombre de fourmis | N (nombre de points) | entier ≥ 1 |
| ρ (évaporation) | 0,5 | strictement entre 0 et 1 |
| α (phéromone) | 1 | > 0 |
| β (proximité) | 3 | > 0 |
| Cycles | 500 | 100 / 500 / 1000 |

Une saisie hors bornes affiche un message d'erreur en français et le run n'est pas lancé.

## Interface (4 quadrants)

1. **Définition et réglage** : ville, nombre de points, réglages ACO, boutons BC1 Réglage manuel,
   BC2 Optuna, BC3 KerasTuner, BC4 SMA, BC5 Run ACO, bouton Arrêter, barre de progression,
   zone « Verdicts des agents » (3 cadres).
2. **Convergence et métriques** : meilleure distance globale par cycle (en direct), meilleure du
   cycle, moyenne des fourmis ; Tout livré ? (tour valide : chaque point une fois + retour),
   distance parcourue (km), km par livraison (distance ÷ (N−1) livraisons), temps d'exécution,
   cycle du meilleur résultat.
3. **Plan cartésien** : projection locale en km (module partagé avec l'AG), départ en étoile,
   ordre de visite fléché, retour au départ en pointillé.
4. **Carte réelle** : Folium + OSRM Route intégrés (WebView2, même technique que l'AG).

La simulation tourne dans un thread ; l'interface est mise à jour via `after` et ne gèle pas.

## Version 1 (actuelle)

- BC1 et BC5 fonctionnels. BC2, BC3, BC4 visibles mais grisés « (V2) » ; les cadres des agents
  affichent « disponible en V2 ».
- 35 tests `tests/test_aco.py` : validité des tours (permutation, pas de doublon, départ fixe),
  évaporation, dépôt Q/L_k sur les arcs orientés, reproductibilité par graine, bornes des
  paramètres.

Résultat mesuré (40 points, réglages par défaut, graine 42) : 88,39 km, tour valide, meilleur
trouvé au cycle 13. On observe la stagnation classique de l'Ant System : après une vingtaine de
cycles, la colonie se concentre sur des tours d'environ 100 km et ne retrouve plus le meilleur.
Le réglage automatique (V2) servira à explorer ρ, α, β et le nombre de fourmis.

## Fichiers ajoutés

```
main_aco.py                       point d'entrée ACO
aco/ant_system.py                 moteur Ant System (ACOParams, AntSystem, ACOResult)
aco/settings_aco.py               tailles 40/60/80, cycles, valeurs par défaut, chemin des fichiers
interface_aco/main_window_aco.py  fenêtre ACO (réutilise Quadrant3, Quadrant4, theme)
interface_aco/quadrant1_aco.py    réglages, BC1–BC5, verdicts des agents
interface_aco/quadrant2_aco.py    convergence et métriques
tests/test_aco.py                 tests du moteur ACO
```

Seule modification d'un module existant : `visualization/folium_map.route_map` accepte un
paramètre `algo` (défaut `"AG"`, donc comportement de l'application AG inchangé) pour afficher
« Parcours optimisé (ACO) » dans la légende de la carte. Les cartes ACO sont écrites dans
`data/cache/aco_map_*.html` pour ne pas écraser celles de l'application AG.

## Version 2 — réglage automatique (BC2, BC3, BC4)

### BC2 · Optuna (`aco/optuna_aco.py`)

- Réutilise le principe et une partie du code Optuna de l'application AG (`startup_trials`, TPE
  multivarié multiobjectif, front de Pareto, recommandation par le point le plus proche de l'idéal).
- Espace : fourmis 10 à 2N (pas de 5), ρ 0,05–0,95, α 0,5–5, β 1–10, cycles 100/500/1000.
- **20 trials au maximum, 2 graines par trial par défaut** (les mêmes graines pour tous les trials) :
  chaque configuration est jugée sur la moyenne, pas sur un tirage chanceux.
- Deux objectifs minimisés séparément : distance moyenne et temps moyen. Avec la distance seule,
  Optuna choisirait toujours 1000 cycles (à graine égale, ajouter des cycles ne peut pas allonger le
  meilleur tour).
- Fenêtre : tableau des trials (moyenne, écart-type, temps), nuage distance/temps avec front de
  Pareto, configuration retenue, boutons « Appliquer et lancer », « Appliquer sans lancer »,
  « Appliquer le trial sélectionné ». Résultats : `data/results/aco_optuna_*.json`.
- Test réel (40 points, 6 trials × 3 graines) : 87,87 km ± 0,50 en moyenne (75 fourmis, ρ = 0,868,
  α = 1,01, β = 2,90, 100 cycles), contre 88,39 km pour le réglage par défaut (graine 42).

### BC4 · SMA (`agents_aco/`)

- Réutilise les agents Gemini de l'AG : client Gemini, recherche DuckDuckGo réelle, appel JSON validé
  avec une relance corrective, vérification des citations [S#], sauvegarde de session.
- Seuls les paramètres changent (fourmis, ρ, α, β, cycles) : schémas et validation Python stricte dans
  `schemas_aco.py` (ρ strictement entre 0 et 1, α > 0, β > 0, cycles ∈ {100, 500, 1000}).
- Agent web (sources DuckDuckGo citées), agent sémantique/scientifique (analyse théorique), agent
  arbitre : il tranche chaque divergence et doit remplir `basis`, la liste de ce sur quoi il fonde sa
  décision (sources vérifiées, arguments, mesures). Un `basis` vide est refusé.
- Les trois cadres « Verdicts des agents » du quadrant 1 affichent paramètres et justification.
- **Validation humaine** : Oui = appliquer et lancer ; Non = appliquer sans lancer (modifiable,
  mode « SMA modifié ») ; Annuler = refuser. La décision est enregistrée dans `aco_agents_*.json`.
- Modèle : pas de sondage des modèles lents. Le SMA utilise le dernier modèle qui a répondu
  (`data/cache/gemini_last_model.txt`, initialement gemini-3.5-flash-lite) ou `GEMINI_MODEL` du `.env`.
- Test réel du 2026-11-23 : l'agent web a répondu (4 sources citées, vérifiées) ; l'agent
  scientifique et l'arbitre ont reçu une erreur 503 (« high demand ») de Gemini. L'erreur est affichée
  dans les cadres, rien n'est appliqué sans décision de l'arbitre.

### BC3 · KerasTuner — historique de l'installation (voir la section « BC3 · KerasTuner (oracle RandomSearch) » plus bas)

- Implémenté : `Tuner` personnalisé dont `run_trial` lance l'ACO sur plusieurs graines et renvoie la
  distance moyenne, oracle **RandomSearch**, 15 trials par défaut. Aucun réseau de neurones.
- keras-tuner 1.4.8 et keras 3.15.1 sont compatibles avec Python 3.13 et sont installés. Mais Keras 3
  exige un backend à l'import (JAX, TensorFlow ou PyTorch) et son backend « numpy » importe JAX.
  L'installation de JAX a dépassé la minute prévue : elle a été arrêtée, donc BC3 reste grisé avec un
  message. Pour l'activer :

```powershell
.venv\Scripts\python.exe -m pip install jax
```

  Au démarrage, l'application vérifie l'import de KerasTuner et active BC3 automatiquement.

### Comparaison Manuel / Optuna / KerasTuner / SMA

Reportée : le bouton « ⚖ Comparer (plus tard) » est visible mais grisé.

### BC3 · KerasTuner (oracle RandomSearch, sans réseau de neurones)

KerasTuner est conçu pour régler des réseaux Keras, mais son cœur est un **oracle** : un moteur
qui propose des hyperparamètres (`hp.Int`, `hp.Float`, `hp.Choice`), reçoit un score et garde le
meilleur. Cet oracle ne dépend d'aucun réseau. On l'utilise donc tel quel avec un `Tuner`
personnalisé dont `run_trial` :

1. lit les valeurs proposées : fourmis 20–80 (pas de 5), ρ 0,05–0,9, α 0,5–3, β 1–6,
   cycles 100/500/1000 ;
2. lance l'ACO sur la matrice OSRM en cache (aucun appel réseau), sur 2 graines ;
3. renvoie la distance moyenne (objectif `distance_km`, à minimiser) ; le temps est enregistré
   comme métrique secondaire.

Créer un faux réseau de neurones pour « justifier » KerasTuner n'apporterait rien : le score serait
le même, avec du calcul inutile et un résultat trompeur. L'oracle RandomSearch tire 15 configurations
au hasard dans l'espace (graine fixée, donc reproductible). C'est une référence simple à comparer
au TPE d'Optuna (BC2).

Keras 3 exige un backend pour s'importer, même sans réseau. L'appli fixe `KERAS_BACKEND` avant
tout import de keras : valeur du `.env` si présente, sinon le premier installé parmi `torch`
(PyTorch CPU, conseillé), `jax`, `tensorflow`. Installation :

```powershell
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Fenêtre BC3 : barre de progression des trials, tableau (paramètres, distance moyenne, écart-type,
temps), meilleure configuration avec « Appliquer au quadrant 1 et lancer l'ACO ». La recherche
tourne dans un thread ; « ■ Arrêter » interrompt le trial en cours et termine la recherche.
Résultats : `data/results/aco_kerastuner_*.json`.
