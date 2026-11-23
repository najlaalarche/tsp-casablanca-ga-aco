# PROMPT MASTER — TSP 40 : Application intelligente d'optimisation par algorithme génétique, Optuna et système multi-agents IA

## 1. RÔLE ET MISSION

Agis comme une équipe senior composée de :

- Un ingénieur Python spécialisé en architecture logicielle.
- Un expert en recherche opérationnelle et problèmes d'optimisation combinatoire.
- Un spécialiste des algorithmes génétiques (GA).
- Un expert en Optuna et optimisation d'hyperparamètres.
- Un ingénieur IA spécialisé en systèmes multi-agents et Gemini.
- Un développeur d'interfaces graphiques Tkinter.
- Un spécialiste SIG, cartographie, Folium et OSRM.

Ta mission est de concevoir et développer une application Python complète, professionnelle, modulaire et fonctionnelle permettant de résoudre un problème de Traveling Salesman Problem (TSP) portant sur exactement 40 points géographiques GPS.

L'application devra proposer trois modes de configuration de l'algorithme génétique :

1. Configuration manuelle.
2. Optimisation automatique des hyperparamètres par Optuna.
3. Configuration intelligente par un système multi-agents composé de trois agents IA.

L'application devra visualiser les résultats en temps réel dans une interface graphique organisée en quatre quadrants.

**Il ne s'agit pas de créer une simple démonstration visuelle. Il faut développer un véritable système d'optimisation utilisant les données GPS réelles, les distances routières réelles et un algorithme génétique opérationnel.**

---

# 2. CONTEXTE ET DÉFINITION DU PROBLÈME TSP

## 2.1 Données d'entrée

Un document contenant les coordonnées GPS des 40 points sera fourni.

Contraintes obligatoires :

- Extraire les 40 points depuis le document fourni.
- Utiliser exactement les coordonnées présentes dans le document.
- Chaque point doit posséder un identifiant unique.
- Chaque point doit comporter une latitude et une longitude.
- Préserver les noms et les identifiants originaux.
- Vérifier que les coordonnées sont valides.
- Ne jamais inventer, supprimer ou modifier arbitrairement un point.

Si le document est ambigu ou incomplet, demander une validation humaine avant de lancer les calculs.

Prévoir une fonctionnalité d'importation permettant de charger les données depuis :
- Excel (.xlsx).
- CSV (.csv).
- Le document original, si son format est exploitable.

Créer une table de données interne structurée :

ID | Nom du point | Latitude | Longitude

Le système doit vérifier automatiquement que 40 points distincts sont présents.

## 2.2 Nature du problème

Résoudre un problème TSP fermé (Closed TSP).

L'objectif est de :

1. Partir d'un point initial fixe défini dans le document.
2. Visiter chacun des 40 points exactement une fois.
3. Respecter l'unicité de chaque visite.
4. Revenir au point de départ.
5. Minimiser la distance totale parcourue sur le réseau routier réel.

Le point de départ est fixe et ne doit jamais être modifié par l'algorithme génétique.

## 2.3 Fonction objectif

Minimiser :

D_total = somme des distances routières entre chaque paire de points consécutifs, y compris le retour au point de départ.

Utiliser les distances routières obtenues à partir du service OSRM.

Ne pas confondre distance euclidienne, distance géodésique et distance routière.

La distance euclidienne peut être utilisée uniquement comme outil secondaire d'analyse ou d'initialisation, jamais comme remplacement silencieux de la distance routière réelle.

---

# 3. ARCHITECTURE GÉNÉRALE DE L'APPLICATION

Développer une application desktop avec Python 3.13.

Technologies principales :

- Python 3.13.
- Tkinter / ttk pour l'interface graphique.
- Matplotlib pour les graphiques.
- Folium pour les cartes géographiques.
- OSRM pour le calcul des distances et des itinéraires routiers.
- Optuna pour l'optimisation automatique.
- Google Gemini API pour les agents IA.
- Pandas et NumPy pour le traitement des données.
- OpenPyXL pour les exports Excel.
- ReportLab pour les rapports PDF.
- Pytest pour les tests automatisés.

Environnement :

- Visual Studio Code.
- Utilisation de l'environnement Python Antigravity comme assistant de développement si disponible.
- Environnement virtuel .venv.
- Fichier requirements.txt.
- Fichier .env pour les clés API.

Utiliser une architecture modulaire professionnelle.

Séparer strictement :
- La logique métier.
- L'algorithme génétique.
- Les services cartographiques.
- Les agents IA.
- L'optimisation Optuna.
- L'interface graphique.
- Les exports.
- Les tests.

---

# 4. INTERFACE GRAPHIQUE — DASHBOARD À QUATRE QUADRANTS

Créer une interface professionnelle en Dark Mode.

L'interface doit être moderne, élégante, ergonomique et adaptée à une présentation académique et professionnelle.

Prévoir :

- Une fenêtre principale redimensionnable.
- Une disposition en quatre quadrants redimensionnables.
- Une barre de navigation ou un en-tête.
- Une barre de progression de 0 à 100 %.
- Des indicateurs de performance.
- Des boutons clairement identifiables.
- Des notifications et messages d'erreur.
- Une mise à jour dynamique pendant la simulation.

## QUADRANT 1 — DÉFINITION DU PROBLÈME ET CONFIGURATION

Ce quadrant constitue le panneau de contrôle principal.

### A. Paramètres de l'algorithme génétique

Permettre à l'utilisateur de configurer les paramètres suivants :

**Population :**
- 10 individus.
- 50 individus.
- 100 individus.

**Méthode de sélection :**
- Roue biaisée (Roulette Wheel Selection).
- Sélection par tournoi (Tournament Selection).

**Taux de mutation :**
- 0.01.
- 0.05.
- 0.10.
- 0.20.
- Possibilité de saisie manuelle d'une valeur personnalisée.

**Taux de croisement :**
- 0.2.
- 0.5.
- 0.8.

**Nombre de générations :**
- 100.
- 500.
- 1000.

**Élitisme :**
- Activé par défaut.
- Conserver les meilleurs individus d'une génération à l'autre.

Utiliser des opérateurs génétiques adaptés aux permutations TSP :

- Initialisation d'individus sous forme de permutations valides.
- Croisement préservant les permutations, par exemple Order Crossover (OX) ou PMX.
- Mutation par échange (Swap Mutation) ou inversion (Inversion Mutation).
- Conservation du point de départ fixe.
- Vérification de l'absence de doublons.
- Conservation du meilleur individu.

Ne jamais utiliser un croisement binaire classique qui pourrait produire des individus invalides.

### B. Trois modes de configuration

Créer trois boutons principaux.

**BOUTON 1 — RÉGLAGE MANUEL**

Permettre à l'utilisateur de sélectionner tous les paramètres de l'AG et de lancer la simulation.

**BOUTON 2 — RÉGLAGE AUTOMATIQUE PAR OPTUNA**

Lancer une optimisation automatique des hyperparamètres.

Prévoir :
- Nombre de trials personnalisable.
- Sampler TPE.
- Optimisation des paramètres de population, mutation, croisement, sélection et nombre de générations.
- Affichage de la progression des trials.
- Affichage de la meilleure configuration trouvée.
- Affichage de la valeur de la fonction objectif.
- Possibilité de consulter l'historique des essais.

L'objectif doit prendre en compte deux critères :
1. Minimiser la distance totale parcourue.
2. Minimiser le temps d'exécution.

Traiter ces deux objectifs comme des objectifs distincts avec Optuna multiobjectif et son sampler TPE adapté (MOTPE), puis présenter les solutions non dominées et une recommandation finale explicable.

Ne pas additionner directement des kilomètres et des secondes sans normalisation ni justification.

**BOUTON 3 — RÉGLAGE PAR AGENTS IA**

Lancer le système multi-agents décrit dans la section 6.

Afficher les propositions des trois agents, puis permettre à l'utilisateur de valider les paramètres recommandés par l'arbitre.

### C. Lancement de la simulation

Prévoir :
- Un bouton LANCER LA SIMULATION.
- Un bouton PAUSE / REPRISE.
- Un bouton ARRÊTER.
- Un bouton EXÉCUTION PAS À PAS.
- Une barre de progression de 0 à 100 %.

La simulation doit pouvoir être exécutée sans bloquer l'interface graphique.

Utiliser des threads ou un mécanisme de tâches asynchrones approprié et communiquer avec Tkinter via sa boucle principale (after).

Gérer correctement les exceptions et les interruptions.

---

## QUADRANT 2 — ANALYSE DES PERFORMANCES

Ce quadrant doit contenir les graphiques permettant de suivre l'évolution de l'algorithme génétique.

### A. Courbe de convergence

Créer une courbe représentant :

- Axe X : numéro de génération.
- Axe Y : meilleure distance totale trouvée (km).

Afficher :
- La meilleure distance à chaque génération.
- L'évolution de la meilleure solution globale.
- La progression de l'algorithme.

La courbe doit être actualisée en temps réel.

### B. Évolution de la distance parcourue

Afficher un graphique complémentaire représentant l'évolution de la distance de la meilleure solution trouvée au cours des générations.

### C. Indicateurs de performance

Afficher :

- Distance totale optimale (km).
- Temps d'exécution (secondes).
- Numéro de génération du meilleur résultat.
- Taux d'amélioration (%).
- Fitness de la meilleure solution.

Utiliser des indicateurs visuels lisibles.

---

## QUADRANT 3 — VISUALISATION CARTÉSIENNE DU TSP

Créer une visualisation cartésienne avec Matplotlib.

Afficher :

- Les 40 points.
- Leurs identifiants.
- Le point de départ avec une couleur distinctive.
- Le parcours optimisé.
- Les connexions entre les points.
- Le retour au point initial.

Le parcours doit être affiché dans l'ordre réel de visite.

Prévoir :
- Zoom.
- Déplacement.
- Actualisation dynamique.
- Légende.
- Distances entre les étapes si possible.

Important : les coordonnées GPS ne doivent pas être présentées comme des coordonnées cartésiennes métriques. Utiliser une projection locale adaptée pour obtenir un véritable plan X/Y, ou afficher explicitement longitude en X et latitude en Y.

Cette visualisation est complémentaire de la carte routière réelle.

---

## QUADRANT 4 — CARTOGRAPHIE RÉELLE AVEC FOLIUM ET OSRM

Créer une carte géographique interactive.

Utiliser :
- Folium.
- Leaflet.
- OSRM.
- OpenStreetMap.

### A. Carte initiale

Afficher les 40 points géographiques avec :
- Des marqueurs numérotés.
- Les identifiants originaux.
- Le nom de chaque lieu.
- Les coordonnées GPS.
- Un marqueur distinctif pour le point de départ.

### B. Carte du parcours optimisé

Après la simulation :

- Afficher le trajet routier réel.
- Respecter l'ordre de visite calculé par l'AG.
- Afficher les routes entre les points.
- Revenir au point initial.
- Afficher les informations de chaque étape.

Utiliser le service OSRM Route pour obtenir la géométrie réelle du trajet.

### C. Calcul des distances

Utiliser l'API OSRM Table pour construire une matrice de distances routières entre les 40 points.

La matrice doit être conservée en mémoire et sauvegardée localement pour éviter de répéter inutilement les appels API.

Attention :
- Les distances routières peuvent être asymétriques.
- A vers B n'est pas nécessairement égal à B vers A.
- Utiliser les distances en mètres fournies par OSRM.
- Convertir en kilomètres pour les affichages.
- Gérer les routes indisponibles.
- Prévoir des délais d'attente, des tentatives limitées et un cache.
- Respecter les conditions d'utilisation du serveur public OSRM.
- Ne pas lancer une requête réseau pour chaque individu de chaque génération.

Si OSRM est indisponible, afficher clairement une erreur et ne pas présenter des distances euclidiennes comme des distances routières.

Prévoir la possibilité de configurer ultérieurement un serveur OSRM personnel.

### D. Intégration dans Tkinter

La carte Folium doit être intégrée dans le quatrième quadrant.

Utiliser un composant de navigateur embarqué compatible avec Tkinter et capable d'exécuter correctement JavaScript et Leaflet.

Vérifier réellement sa compatibilité avec Python 3.13.

Ne pas prétendre qu'une carte est intégrée si elle s'ouvre simplement dans un navigateur externe.

---

# 5. ALGORITHME GÉNÉTIQUE — LOGIQUE DE RÉSOLUTION

Développer un moteur AG indépendant de l'interface graphique.

Le moteur doit recevoir :
- Les 40 points.
- La matrice des distances routières.
- Les paramètres de l'algorithme.

Il doit retourner :
- Le meilleur ordre de visite.
- La distance totale.
- La meilleure fitness.
- L'évolution des générations.
- Le temps d'exécution.
- Les statistiques de la simulation.

### Fonction fitness

Utiliser une fonction adaptée à un problème de minimisation.

Par exemple :

Fitness = 1 / (1 + distance totale)

La fonction doit être définie de manière cohérente et stable.

### Contraintes de validité

Chaque individu doit respecter :

- 40 points exactement.
- Chaque point visité une seule fois.
- Point de départ fixe.
- Retour au point de départ.
- Aucun point supplémentaire.
- Aucun point manquant.

### Reproductibilité

Prévoir :
- Une graine aléatoire configurable.
- La conservation des paramètres utilisés.
- L'enregistrement des résultats.
- La possibilité de reproduire une simulation.

---

# 6. SYSTÈME MULTI-AGENTS IA — GEMINI API

Créer un système composé de trois agents spécialisés.

Utiliser l'API officielle Google Gemini avec le SDK Python actuel google-genai.

Ne pas utiliser d'anciennes bibliothèques dépréciées.

Vérifier le nom exact d'un modèle Gemini compatible et disponible via l'API officielle avant de configurer l'application.

Ne pas inventer l'existence d'un modèle Gemini 3.8.

La clé API doit être stockée dans un fichier .env et ne jamais apparaître dans le code source ou les journaux.

## AGENT 1 — WEB RESEARCH AGENT

Mission :

Effectuer des recherches sur les meilleures stratégies d'optimisation du TSP par algorithme génétique.

Il doit :
- Rechercher les bonnes pratiques scientifiques.
- Étudier les paramètres AG adaptés à un TSP de 40 points.
- Analyser les stratégies de sélection.
- Étudier les taux de mutation et de croisement.
- Identifier les opérateurs génétiques adaptés.
- Proposer une configuration argumentée.

Ses recommandations doivent être fondées sur des sources vérifiables.

Ne pas prétendre effectuer une recherche web si aucun outil de recherche n'est disponible.

## AGENT 2 — SCIENTIFIC / SEMANTIC AGENT

Mission :

Analyser le problème sous un angle scientifique et théorique.

Il doit :
- Analyser les caractéristiques du TSP.
- Exploiter les connaissances scientifiques sur les algorithmes génétiques.
- Étudier les méthodes d'optimisation combinatoire.
- Identifier les risques de convergence prématurée.
- Analyser les interactions entre les hyperparamètres.
- Proposer une configuration techniquement justifiée.

## AGENT 3 — REASONER / ARBITER AGENT

Mission :

Recevoir les propositions des deux premiers agents et construire une recommandation finale.

Il doit :
- Comparer les recommandations.
- Identifier les points de convergence.
- Identifier les contradictions.
- Évaluer la cohérence des paramètres.
- Écarter les propositions non justifiées.
- Produire une configuration finale argumentée.

L'arbitre doit rechercher un consensus entre les agents.

### Format de communication

Les trois agents doivent échanger des données structurées en JSON.

Exemple de sortie :

{
  "population_size": 50,
  "selection_method": "tournament",
  "mutation_rate": 0.05,
  "crossover_rate": 0.8,
  "generations": 500,
  "elitism": true,
  "justification": "Configuration proposée après analyse des contraintes du TSP."
}

Les paramètres doivent être validés par Python avant leur utilisation.

### Validation humaine

Le système ne doit pas lancer automatiquement l'AG après la décision de l'arbitre.

Afficher une fenêtre de validation présentant :
- Les propositions des agents.
- La configuration finale.
- Les justifications.
- Les éventuelles divergences.

L'utilisateur doit pouvoir :
- Accepter la configuration.
- Modifier les paramètres.
- Refuser la recommandation.

Une fois validée, la simulation peut commencer.

---

# 7. OPTIMISATION AUTOMATIQUE AVEC OPTUNA

Créer un module indépendant pour Optuna.

Paramètres optimisables :

- Population : 10, 50, 100.
- Sélection : roulette, tournoi.
- Mutation : valeurs comprises dans une plage définie, avec 0.01, 0.05, 0.10 et 0.20 comme valeurs de référence.
- Croisement : 0.2, 0.5, 0.8.
- Générations : 100, 500, 1000.

Utiliser TPESampler / MOTPE pour l'optimisation multiobjectif.

Permettre de définir le nombre de trials directement depuis l'interface.

Afficher :
- Numéro du trial.
- Paramètres testés.
- Distance obtenue.
- Temps d'exécution.
- Meilleures solutions non dominées.
- Configuration recommandée.

Les expériences doivent utiliser des conditions comparables, une graine contrôlée et la même matrice de distances.

La recommandation finale doit expliquer le compromis entre qualité de la solution et temps de calcul.

---

# 8. COMPARAISON DES TROIS MODES

Créer un module permettant de comparer :

1. AG avec réglage manuel.
2. AG optimisé par Optuna.
3. AG configuré par les agents IA.

Présenter les résultats dans un tableau comparatif :

| Indicateur | Manuel | Optuna | Multi-agents |
|---|---|---|---|
| Distance totale | | | |
| Temps d'exécution | | | |
| Nombre de générations | | | |
| Meilleure fitness | | | |
| Taux d'amélioration | | | |

Ne jamais remplir les cellules avec des valeurs fictives.

Chaque résultat doit provenir d'une véritable simulation.

---

# 9. RÉPÉTITIONS ET VALIDATION STATISTIQUE

Prévoir 10 exécutions indépendantes pour chaque mode.

Chaque exécution doit utiliser :
- La même matrice de distances.
- Les mêmes contraintes TSP.
- Des graines aléatoires contrôlées.
- Les paramètres correspondant au mode testé.

Calculer :

- Meilleure distance.
- Distance moyenne.
- Pire distance.
- Écart-type.
- Temps moyen d'exécution.

Afficher les résultats sous forme de graphiques et de tableaux.

Le système doit distinguer les résultats d'une exécution unique des statistiques issues des 10 répétitions.

---

# 10. EXPORTATION ET HISTORIQUE

Créer un système de sauvegarde des simulations.

Pour chaque simulation, conserver :

- Date et heure.
- Mode utilisé.
- Paramètres de l'AG.
- Distance totale.
- Temps d'exécution.
- Ordre de visite.
- Résultats de convergence.
- Résultats des agents, si applicable.
- Statistiques des répétitions.

Prévoir trois formats d'exportation :

**CSV / Excel**
- Paramètres.
- Distances.
- Ordre des points.
- Statistiques.
- Résultats comparatifs.

**PDF**
Créer un rapport professionnel contenant :

1. Présentation du problème TSP.
2. Présentation des 40 points.
3. Méthodologie de résolution.
4. Paramètres utilisés.
5. Justification des agents IA.
6. Graphique de convergence.
7. Carte géographique du parcours optimal.
8. Distance totale.
9. Temps d'exécution.
10. Résultats statistiques.
11. Conclusion.

Prévoir un bouton permettant de consulter l'historique des simulations.

---

# 11. ARCHITECTURE DES DOSSIERS

Créer une structure modulaire semblable à celle-ci :

TSP_40/
│
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── config/
│   ├── settings.py
│   └── constants.py
│
├── data/
│   ├── input/
│   ├── processed/
│   ├── cache/
│   └── results/
│
├── core/
│   ├── tsp_problem.py
│   ├── genetic_algorithm.py
│   ├── fitness.py
│   ├── operators.py
│   └── validation.py
│
├── optimization/
│   ├── optuna_optimizer.py
│   └── experiment_runner.py
│
├── agents/
│   ├── web_agent.py
│   ├── semantic_agent.py
│   ├── arbiter_agent.py
│   ├── agent_manager.py
│   └── schemas.py
│
├── services/
│   ├── osrm_service.py
│   ├── distance_matrix.py
│   ├── data_loader.py
│   └── cache_manager.py
│
├── visualization/
│   ├── convergence.py
│   ├── cartesian_map.py
│   └── folium_map.py
│
├── interface/
│   ├── main_window.py
│   ├── dashboard.py
│   ├── quadrant_1.py
│   ├── quadrant_2.py
│   ├── quadrant_3.py
│   ├── quadrant_4.py
│   ├── controls.py
│   └── dialogs.py
│
├── reports/
│   ├── excel_export.py
│   ├── csv_export.py
│   └── pdf_report.py
│
└── tests/
    ├── test_genetic_algorithm.py
    ├── test_operators.py
    ├── test_distance_matrix.py
    ├── test_validation.py
    ├── test_optuna.py
    └── test_agents.py

Cette architecture peut être améliorée si une meilleure organisation est justifiée.

---

# 12. TESTS ET VALIDATION

Créer des tests unitaires et fonctionnels.

Tester notamment :

1. Importation correcte des 40 points.
2. Validation des coordonnées GPS.
3. Construction de la matrice OSRM.
4. Validité des permutations.
5. Respect du point de départ.
6. Retour au point initial.
7. Absence de doublons.
8. Fonction fitness.
9. Opérateurs de croisement.
10. Opérateurs de mutation.
11. Conservation de l'élitisme.
12. Convergence de l'AG.
13. Fonctionnement d'Optuna.
14. Communication avec Gemini.
15. Validation des réponses des agents.
16. Génération des cartes.
17. Export CSV, Excel et PDF.
18. Réactivité de l'interface.

Prévoir des tests avec des données synthétiques pour vérifier le fonctionnement du moteur, mais ne jamais les confondre avec les résultats du véritable problème à 40 points.

---

# 13. EXIGENCES DE QUALITÉ ET CONTRAINTES

Respecter impérativement les règles suivantes :

- Aucun point GPS inventé.
- Aucune distance routière fictive.
- Aucun résultat de simulation simulé présenté comme réel.
- Aucun appel API inutile à chaque génération.
- Aucune clé API exposée.
- Aucun blocage de l'interface graphique.
- Aucune permutation invalide.
- Aucun changement du point de départ.
- Aucun résultat non vérifié.
- Aucune dépendance dépréciée sans justification.

Utiliser des logs pour enregistrer les événements importants.

Prévoir une gestion des erreurs :
- Absence de connexion Internet.
- Indisponibilité OSRM.
- Clé Gemini invalide.
- Erreur de lecture du document.
- Coordonnées invalides.
- Échec d'exportation.
- Interruption de simulation.

Les erreurs doivent être compréhensibles pour l'utilisateur.

---

# 14. MÉTHODOLOGIE DE DÉVELOPPEMENT OBLIGATOIRE

Ne développe pas toute l'application en un seul bloc de code.

Travaille progressivement en suivant les étapes suivantes :

**PHASE 1 — Analyse des données**
- Examiner le document contenant les 40 points.
- Extraire les coordonnées.
- Vérifier les identifiants et les coordonnées.
- Identifier le point de départ.
- Présenter les données extraites pour validation.

Ne jamais commencer l'optimisation réelle avant la validation des données.

**PHASE 2 — Architecture**
- Créer le projet.
- Installer les dépendances.
- Configurer Python 3.13.
- Créer les modules.
- Tester les imports.

**PHASE 3 — Moteur TSP**
- Développer le modèle mathématique.
- Construire les distances.
- Développer l'AG.
- Tester les opérateurs.
- Vérifier la validité des solutions.

**PHASE 4 — Interface graphique**
- Construire le dashboard.
- Créer les quatre quadrants.
- Ajouter les contrôles.
- Intégrer les graphiques.
- Intégrer Folium.

**PHASE 5 — Optuna**
- Développer l'optimisation automatique.
- Connecter les résultats à l'interface.
- Vérifier le fonctionnement multiobjectif.

**PHASE 6 — Agents IA**
- Développer les trois agents.
- Intégrer Gemini.
- Construire le mécanisme de consensus.
- Ajouter la validation humaine.

**PHASE 7 — Expérimentations**
- Exécuter les trois modes.
- Réaliser les 10 répétitions.
- Comparer les résultats.
- Vérifier les performances.

**PHASE 8 — Exports et finalisation**
- Créer les exports.
- Créer les rapports PDF.
- Ajouter l'historique.
- Exécuter les tests.
- Rédiger le README.

À chaque phase :
- Présenter les fichiers créés.
- Fournir leur code complet.
- Exécuter les tests pertinents.
- Corriger les erreurs avant de continuer.
- Ne pas déclarer une phase terminée sans vérification.

---

# 15. LIVRABLE FINAL ATTENDU

À la fin du développement, fournir :

1. L'intégralité du code source.
2. Une architecture modulaire fonctionnelle.
3. Le fichier requirements.txt.
4. Le fichier .env.example.
5. Le README complet.
6. Les instructions d'installation sous Windows.
7. Les instructions de lancement depuis VS Code.
8. Les instructions de configuration de Gemini API.
9. Les tests automatisés.
10. Une application desktop opérationnelle.

Le projet doit pouvoir être lancé avec :

python main.py

L'application doit être réellement fonctionnelle, et non uniquement esthétique.

## INSTRUCTION FINALE

Commence par examiner le document source contenant les 40 points GPS.

Si le document n'est pas encore joint, demande-moi de le fournir.

Ne génère aucune donnée géographique fictive pour remplacer les données réelles.

Avant d'écrire le code, présente-moi :
- Ton analyse du problème.
- Les éventuelles ambiguïtés.
- L'architecture technique proposée.
- Les dépendances et leur compatibilité avec Python 3.13.
- Les éventuelles limitations des services publics OSRM et Gemini.

Ensuite, commence le développement phase par phase.

Agis comme un ingénieur logiciel senior : privilégie la fiabilité mathématique, la qualité de l'architecture, la précision des résultats et l'expérience utilisateur plutôt que la rapidité de génération du code.