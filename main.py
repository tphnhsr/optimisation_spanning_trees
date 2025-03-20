import pulp as pl
import networkx as nwx
import os
import itertools
import glpk
import time
import random
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


#Chemin vers le cplex.exe
path_to_cplex = '/Applications/CPLEX_Studio_Community2211/cplex/bin/x86-64_osx'


# création des graphes décrits par les instances
def creer_graphe(s):
    # Graphe vide
    G = nwx.Graph()
    is_empty = 1

    # ouverture de l'instance
    file = open('Instances/Spd_Inst_Rid_Final2/'+str(s), 'r')
    lignes = file.readlines()

    for line in lignes:
        current_line = str.split(line)

        # on ajoute tous les sommets
        if is_empty == 1:
            G.add_nodes_from(range(1, int(current_line[0])+1))
            is_empty = 0

        else:
            G.add_edge(int(current_line[0]), int(current_line[1]))

    return G


def nombre_arcs(graphe):
    # Renvoie le nombre d'arêtes dans le graphe
    return graphe.number_of_edges()


def nombre_sommets(graphe):
    # Renvoie le nombre d'arêtes dans le graphe
    return graphe.number_of_nodes()


#x_e vaut 1 si l'arete e est sélectionnée, 0 sinon
#y_v vaut 1 si le sommet v est un sommet de branchement, 0 sinon
class SolveurPLNE:
    def __init__(self, G):
        self.G = G
        self.VG = G.nodes()
        self.EG = G.edges()

        self.DG = nwx.DiGraph()
        for (u, v) in self.EG:
            self.DG.add_edge(u, v)
            self.DG.add_edge(v, u)


    def defineProblem(self):
        # Initialize the problem
        self.problem = pl.LpProblem('MinimumBranchVertices', pl.LpMinimize)

        # Initialize variables
        self.X = pl.LpVariable.dicts('X', self.DG.edges(), 0, 1, pl.LpBinary)
        self.Y = pl.LpVariable.dicts('Y', self.VG, 0, 1, pl.LpBinary)

    def objectiveFunction(self):
        # Objective function: Minimize the sum of Y
        self.problem += pl.lpSum(self.Y.values()), "Minimize the sum of Y"

    def addConstraints(self):
        # Constraint (3): Sum of Xe for all edges incident to v should be |VG| - 1
        for v in self.VG:
            self.problem += pl.lpSum(self.X[e] for e in self.G.edges(v)) == len(self.VG) - 1, f"Constraint_3_{v}"

        # Constraint (4): Sum of Xe for all edges in E(S) should be less than or equal to |S| - 1
        for S in range(1, len(self.VG)):
            for subset in combinations(self.VG, S):
                edges_in_subset = self.G.subgraph(subset).edges()
                self.problem += pl.lpSum(self.X[e] for e in edges_in_subset) <= len(subset) - 1, f"Constraint_4_{subset}"

        # Constraint (5): Sum of Xe for all edges incident to v minus 2 must be less than or equal to dg(v) * Yv
        for v in self.VG:
            self.problem += pl.lpSum(self.X[e] for e in self.G.edges(v)) - 2 <= self.G.degree(v) * self.Y[v], f"Constraint_5_{v}"

    def solve(self):
        self.problem.solve()
        # on vérifie si une solution a été obtenue
        if self.problem.status == pl.LpStatusOptimal:
            print("Solution optimale trouvée!")
            # affichage de X et Y
            for e in self.EG:
                print(f"X_{e} = {self.X[e].value()}")
            for v in self.VG:
                print(f"Y_{v} = {self.Y[v].value()}")
        else:
            print("Une solution optimale n'a pas pu être trouvée.")

    def main(self):
        self.defineProblem()
        self.objectiveFunction()
        self.addConstraints()
        self.solve()


class SolveurPLNECP:
    def __init__(self, nb_sommet, nb_aretes, G):
        self.ns = nb_sommet
        self.na = nb_aretes
        self.G = G.to_directed()  # graphe orienté
        self.E_prime = [(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()]  # E' avec les arcs symétriques
        self.arcs = list(G.edges())  # List of arcs in the symmetrical graph

    def defineProblem(self):
        # problème de minimisation (somme des Y)
        self.problem = pl.LpProblem('PL', pl.LpMinimize)

        # Variables Y pour les sommets
        self.Y = [pl.LpVariable(f'Y_{i}', cat=pl.LpBinary) for i in range(1, self.ns + 1)]

        # Variables X pour les arcs dans E' et variables de flot F
        self.X = {}
        self.F = {}
        for u, v in self.E_prime:
            self.X[(u, v)] = pl.LpVariable(f'X_{u}_{v}', cat=pl.LpBinary)
            self.F[(u, v)] = pl.LpVariable(f'F_{u}_{v}', lowBound=0, cat=pl.LpContinuous)

    def objectiveFunction(self):
        # La fonction objectif est la même que pour Solveur1 : minimiser la somme des Y
        self.problem += pl.lpSum(self.Y), "Minimize_sum_of_Y"

    def addConditions(self):
        # Ajout des conditions spécifiques au problème de flux

        # (9) Pour chaque sommet v dans VG, la somme des Xu,v pour (u,v) dans A-(v) est égale à 1
        for v in self.G.nodes():
            #G.predecessors renvoie les prédécesseurs de v dans u, soit u tq (u,v) dans G
            self.problem += pl.lpSum(self.X[(u, v)] for u in self.G.predecessors(v)) == 1, f"Condition_9_{v}"

        # Choisissons un sommet arbitraire comme source s
        # Améliorable?
        s = list(self.G.nodes())[0]
        VG_without_s = set(self.G.nodes()) - {s}

        # (10) et (11) Conditions de conservation du flot
        for v in self.G.nodes():
            if v != s:
                self.problem += (pl.lpSum(self.F[(s, v)] for v in self.G.successors(s)) - pl.lpSum(self.F[(v, s)] for v in self.G.predecessors(s))) == self.ns - 1, f"Condition_10_{v}"

        for v in VG_without_s:
            self.problem += (pl.lpSum(self.F[(v, u)] for u in self.G.successors(v)) - pl.lpSum(self.F[(u, v)] for u in self.G.predecessors(v))) == -1, f"Condition_11_{v}"

        # (12) Contraintes sur les variables de flot en fonction des variables X
        for e in self.G.edges():
            self.problem += self.X[e] <= self.F[e], f"Condition_12a_{e}"
            self.problem += self.F[e] <= self.ns * self.X[e], f"Condition_12b_{e}"


        # (13) Conditions liées au degré des sommets
        for v in self.G.nodes():
            #on prend directement toutes les arêtes incidentes à v
            incident_edges = self.G.edges(v)
            self.problem += (pl.lpSum(self.X[e] for e in incident_edges) - 2 <= self.G.degree(v, weight=None) * self.Y[v - 1]), f"Condition_13_{v}"

    def solve(self):
        # Solution et affichage des résultats
        self.problem.solve()
        solution = []
        if self.problem.status == pl.LpStatusOptimal:
            print("Solution optimale trouvée!")
            # Affichage des valeurs des variables de décision Y
            print("\nValeurs des variables de décision Y (sommets de branchement):")
            for v in self.G.nodes():
                print(f"Y_{v} = {self.Y[v-1].value()}")
            # Affichage des valeurs des variables de décision X
            print("\nValeurs des variables de décision X (arêtes sélectionnées):")
            for u, v in self.E_prime:
                solution.append(self.X[(u, v)].value())
                print(f"X_{u}_{v} = {self.X[(u, v)].value()}")

            selected_edges = []
            for u, v in self.E_prime:
                if self.X[(u, v)].value() == 1:
                    selected_edges.append(self.X[(u, v)])
                print(f"X_{u}_{v} = {self.X[(u, v)].value()}")
            print("\nListe des arêtes sélectionnées dans la solution optimale:")
            print(selected_edges)

        else:
            print("Une solution optimale n'a pas pu être trouvée.")
            for i in range(self.na):
                solution.append(0)

        return solution

    def main(self):

        self.defineProblem()
        self.objectiveFunction()
        self.addConditions()
        self.solve()


class SolveurPLNECPM:
    def __init__(self, nb_sommet, nb_arcs, G):
        self.ns = nb_sommet
        self.na = nb_arcs
        self.G = G
        self.edges = list(G.edges())  # Liste des arêtes du graphe

        # Create a symmetric directed graph from the undirected graph G
        self.DG = nwx.DiGraph()
        for (u, v) in self.edges:
            self.DG.add_edge(u, v)
            self.DG.add_edge(v, u)

        # Arc sets for each node
        self.A_plus = {v: self.DG.out_edges(v) for v in self.DG.nodes()}
        self.A_minus = {v: self.DG.in_edges(v) for v in self.DG.nodes()}

    def defineProblem(self):
        # Initialize the problem
        self.problem = pl.LpProblem('PL', pl.LpMinimize)

        # Initialize variables

        # Améliorable?
        self.s = list(self.G.nodes())[0]
        self.Y = {v: pl.LpVariable(f'y_{v}', cat='Binary') for v in self.G.nodes()}
        self.X = {(u, v): pl.LpVariable(f'x_{u}_{v}', cat='Binary') for u, v in self.DG.edges()}
        self.F = {(u, v, k): pl.LpVariable(f'f_{u}_{v}_{k}', lowBound=0)
                  for u, v in self.DG.edges() for k in self.G.nodes() if k != self.s}

    def objectiveFunction(self):
        # La fonction objectif est la même que pour Solveur1 : minimiser la somme des Y
        self.problem += pl.lpSum(self.Y.values()), "Minimize_sum_of_Y"

    def addConditions(self):

        # Constraint (18)
        for v in self.G.nodes():
            if v != self.s:
                self.problem += pl.lpSum(self.X[(u, v)] for u, v in self.A_minus[v]) == 1, f"Constraint_18_{v}"

        # Constraint (19)
        for v in self.G.nodes():
            for k in self.G.nodes():
                if v != self.s and v != k and k != self.s:
                    self.problem += (pl.lpSum(self.F[(v, u, k)] for v, u in self.A_plus[v]) - pl.lpSum(self.F[(u, v, k)] for u, v in self.A_minus[v])) == 0, f"Constraint_19_{v}_{k}"

        # Constraint (20)
        for k in self.G.nodes():
            if k != self.s:
                self.problem += (pl.lpSum(self.F[(self.s, v, k)] for self.s, v in self.A_plus[self.s]) -
                                 pl.lpSum(self.F[(u, self.s, k)] for u, self.s in
                                          self.A_minus[self.s])) == 1, f"Constraint_20_{k}"

        # Constraint (21)
        for k in self.G.nodes():
            if k != self.s:
                self.problem += (pl.lpSum(self.F[(k, u, k)] for k, u in self.A_plus[k]) -
                                 pl.lpSum(self.F[(i, k, k)] for i, k in self.A_minus[k])) == -1, f"Constraint_21_{k}"

        # Constraint (22)
        for u, v in self.DG.edges():
            for k in self.G.nodes():
                if k != self.s:
                    self.problem += self.F[(u, v, k)] <= self.X[(u, v)], f"Constraint_22_{u}_{v}_{k}"

        # Constraint (23)
        for v in self.G.nodes():
            self.problem += (pl.lpSum(self.X[(v, u)] for v, u in self.A_plus[v]) +
                             pl.lpSum(self.X[(u, v)] for u, v in self.A_minus[v]) - 2) <= self.G.degree(v) * self.Y[
                                v], f"Constraint_23_{v}"

    def solve(self):
        # Solve the problem
        self.problem.solve()

        # Check the status of the solution
        solution = []
        if self.problem.status == pl.LpStatusOptimal:
            print("Solution optimale trouvée!")
            # Display the values of the decision variables Y and X
            print("\nValeurs des variables de décision Y (sommets de branchement):")
            for v in self.G.nodes():
                print(f"Y_{v} = {self.Y[v].value()}")
            selected_edges = []
            print("\nValeurs des variables de décision X (arêtes sélectionnées):")
            for (u, v) in self.DG.edges():
                print(f"X_{u}_{v} = {self.X[(u, v)].value()}")
                solution.append(self.X[(u, v)].value())
                if self.X[(u, v)].value() == 1:
                    selected_edges.append(self.X[(u, v)])
                # For flows, we iterate over all possible k values, except for the source
            #print("\nValeurs des variables de flot f :")
            #for (u, v) in self.DG.edges():
                #for k in self.G.nodes():
                    #if k != self.s:
                        #print(f"F_{u}_{v}_{k} = {self.F[(u, v, k)].value()}")
            print("\nListe des arêtes sélectionnées dans la solution optimale:")
            print(selected_edges)

        else:
            print("Une solution optimale n'a pas pu être trouvée.")
            for i in range(self.na):
                solution.append(0)
        print(solution)
        return solution

    def main(self):

        self.defineProblem()
        self.objectiveFunction()
        self.addConditions()
        self.solve()


class SolveurMartin:
    def __init__(self, nb_sommet, nb_aretes, G):
        self.G = G
        self.V = G.nodes()
        self.E = G.edges()
        self.na = nb_aretes

        self.DG = nwx.DiGraph()
        for (u, v) in self.E:
            self.DG.add_edge(u, v)
            self.DG.add_edge(v, u)

    def defineProblem(self):
        # Initialize the problem
        self.problem = pl.LpProblem('PL', pl.LpMinimize)

        # Initialize variables
        self.X = {(i, j): pl.LpVariable(f'x_{i}_{j}', cat='Binary') for i, j in self.DG.edges()}
        self.Y = {(i, j, k): pl.LpVariable(f'y_{i}_{j}_{k}', cat='Binary') for i, j in self.DG.edges() for k in self.V}
        self.Z = {v: pl.LpVariable(f'z_{v}', cat='Binary') for v in self.V}

    def objectiveFunction(self):
        # The objective function is to minimize the sum of Z
        self.problem += pl.lpSum(self.Z.values()), "Minimize_sum_of_Z"

    def addConditions(self):
        # Constraint (27a)
        self.problem += pl.lpSum(self.X.values()) == len(self.V) - 1, "Constraint_27a"

        # Constraint (27b)
        for i, j in self.E:
            for k in self.V:
                self.problem += self.Y[(i, j, k)] + self.Y[(j, i, k)] == self.X[(i, j)], f"Constraint_27b_{i}_{j}_{k}"

        # Constraint (27c)
        for i, j in self.E:
            self.problem += pl.lpSum(self.Y[(i, j, k)] for k in self.V if k != i and k != j) + self.X[(i, j)] == 1, f"Constraint_27c_{i}_{j}"

        # Constraint (27d)
        for i in self.V:
            self.problem += pl.lpSum(self.X[(i, j)] for j in self.G.neighbors(i)) - len(list(self.G.neighbors(i))) * self.Z[i] <= 2, f"Constraint_27d_{i}"

    def solve(self):
        # Solve the problem
        self.problem.solve()

        # Check the status of the solution
        solution=[]
        if self.problem.status == pl.LpStatusOptimal:
            print("Solution optimale trouvée!")
            # Display the values of the decision variables Z
            for v in self.V:
                print(f"Z_{v} = {self.Z[v].value()}")
            # Display the values of the decision variables X
            for i, j in self.E:
                print(f"X_{i}_{j} = {self.X[(i, j)].value()}")
                solution.append(self.X[(i, j)].value())
            # Display the values of the decision variables Y
            for i, j in self.E:
                for k in self.V:
                    print(f"Y_{i}_{j}_{k} = {self.Y[(i, j, k)].value()}")
        else:
            print("Une solution optimale n'a pas pu être trouvée.")
            for i in range(self.na):
                solution.append(0)
        print(solution)
        return solution

    def main(self):
        self.defineProblem()
        self.objectiveFunction()
        self.addConditions()
        self.solve()


class SolveurCycle:
    def __init__(self, nb_sommet, nb_aretes, G):
        self.original_graph = G
        self.G = G.copy()  # Nous travaillons avec une copie pour ne pas modifier le graphe original

    def compute_base_cycle(self):
        # Calcule une base de cycles du graphe
        self.base_cycle = list(nwx.cycle_basis(self.G))

    def break_cycles(self):
        # "Casse" les cycles en retirant une arête de chaque cycle
        for cycle in self.base_cycle:
            for i in range(len(cycle)):
                if self.G.has_edge(cycle[i], cycle[(i + 1) % len(cycle)]):
                    self.G.remove_edge(cycle[i], cycle[(i + 1) % len(cycle)])
                    break  # Nous cassons un cycle à la fois

    def is_connected(self):
        # Vérifie si le graphe est connexe
        return nwx.is_connected(self.G)

    def solve(self):
        # Résout le problème MBVST
        self.compute_base_cycle()
        self.break_cycles()

        # Si en retirant des arêtes le graphe n'est pas connexe, on réitère
        while not self.is_connected():
            # Reconnecte les composantes connexes en réintroduisant les arêtes nécessaires
            self.reconnect_components()
            self.compute_base_cycle()
            self.break_cycles()

        return self.G

    def reconnect_components(self):
        # Réintroduit les arêtes entre les composantes connexes pour les reconnecter
        components = list(nwx.connected_components(self.G))
        if len(components) > 1:
            for i in range(len(components) - 1):
                for v in components[i]:
                    for u in components[i + 1]:
                        if self.original_graph.has_edge(u, v):
                            self.G.add_edge(u, v)
                            break


def extract_features_graph(graph):
    #on transforme le texte décrivant le graphe dans chaque fichier en un graphe.
    list_deg_u = []
    list_deg_v = []
    list_is_in_cycle = []
    list_redondance = calculer_redondance_des_aretes(graph)
    list_conn_locale = calculer_connectivite_locale(graph)
    list_cluster_coeff = calculer_coefficient_clustering(graph)
    #list_graph_id = []

    list_edges_cycle = list(nwx.find_cycle(graph))

    #on parcourt les arêtes de ce graphe
    for (u, v) in graph.edges():
        #list_graph_id.append(graph[1].rsplit('.',1)[0])
        list_deg_u.append(graph.degree(u))
        list_deg_v.append(graph.degree(v))
        #on vérifie sur l'arc est dans un cycle
        if (u, v) in list_edges_cycle:
            list_is_in_cycle.append(1)
        else:
            list_is_in_cycle.append(0)

    #'graph_id': list_graph_id,
    return pd.DataFrame({'degre_u': list_deg_u, 'degre_v': list_deg_v, 'is_in_cycle': list_is_in_cycle, 'clustering_coefficient': list_cluster_coeff, 'redondance_edges': list_redondance, 'connectivite_locale': list_conn_locale})


def calculer_centralite_des_aretes(graph):
    centralite = nwx.edge_betweenness_centrality(graph)
    return [centralite[edge] if edge in centralite else 0 for edge in graph.edges()]


def calculer_coefficient_clustering(graph):
    clustering = nwx.clustering(graph)
    return [(clustering[u] + clustering[v]) / 2 for u, v in graph.edges()]


def calculer_connectivite_locale(graph):
    return [graph.degree(u) + graph.degree(v) for u, v in graph.edges()]


def calculer_redondance_des_aretes(graph):
    redondance = []
    for u, v in graph.edges():
        paths = len(list(nwx.all_simple_paths(graph, u, v, cutoff=3))) - 1
        redondance.append(paths)
    return redondance


class Apprentissage:
    def __init__(self, graphs):
        self.graphs = graphs  # Liste des graphes (instances) que l'on utilise
        self.model = None  # Le modèle d'apprentissage sera initialisé plus tard dans le code

    def extract_features(self, graph):
        #on transforme le texte décrivant le graphe dans chaque fichier en un graphe.
        G = creer_graphe(graph[1])
        nb_sommet = nombre_sommets(G)
        nb_aretes = nombre_arcs(G)
        list_redondance = calculer_redondance_des_aretes(G)
        list_conn_locale = calculer_connectivite_locale(G)
        list_cluster_coeff = calculer_coefficient_clustering(G)
        list_centralite_arete = calculer_centralite_des_aretes(G)
        list_deg_u = []
        list_deg_v = []
        list_is_in_cycle = []
        list_is_in_solution = []
        list_graph_id = []

        list_edges_cycle = list(nwx.find_cycle(G))

        #on récupère les arêtes dans la solution
        solveur = SolveurCycle(nb_sommet, nb_aretes, G)
        spanning_tree = solveur.solve()
        list_aretes_solution = spanning_tree.edges()

        #on parcourt les arêtes de ce graphe
        for (u, v) in G.edges():
            list_graph_id.append(graph[1].rsplit('.',1)[0])
            list_deg_u.append(G.degree(u))
            list_deg_v.append(G.degree(v))
            #on vérifie sur l'arc est dans un cycle
            if (u, v) in list_edges_cycle:
                list_is_in_cycle.append(1)
            else:
                list_is_in_cycle.append(0)

            if (u, v) in list_aretes_solution:
                list_is_in_solution.append(1)
            else:
                list_is_in_solution.append(0)
        #'graph_id': list_graph_id,
        #'clustering_coefficient': list_cluster_coeff, 0.77
        #'redondance_edges': list_redondance,  0.77
        #'connectivite_locale': list_conn_locale,   0.76
        #'centralite_edge': list_centralite_arete,  0.72
        return pd.DataFrame({'degre_u': list_deg_u, 'degre_v': list_deg_v, 'is_in_cycle': list_is_in_cycle, 'clustering_coefficient': list_cluster_coeff, 'redondance_edges': list_redondance, 'connectivite_locale': list_conn_locale, 'is_in_solution': list_is_in_solution})

    def prepare_data(self):
        dataframes = []
        for graph in enumerate(self.graphs):
            graph_features = self.extract_features(graph)
            dataframes.append(graph_features)

        # Concaténation en un seul DataFrame : ce sera alors le df que l'on utilise pour nos données
        all_data = pd.concat(dataframes)
        return all_data

    def train_model(self):
        all_data = self.prepare_data()

        # Séparation des variables caractéristiques (X) et de la variable cible (Y)
        X = all_data.drop('is_in_solution', axis=1)
        y = all_data['is_in_solution']

        # Division en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Création et entraînement du modèle
        # On modifie si besoin le modèle ici

        #modèle randomforest classifier
        #self.model = RandomForestClassifier()

        #modèle SVM classifier
        self.model = SVC()

        #modèle KNN classifier
        #self.model = KNeighborsClassifier()

        #modèle XGboost classifier
        #self.model = XGBClassifier()

        self.model.fit(X_train, y_train)

        # Prédiction et évaluation
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy}')

    def predict(self, test_data):
        # Prédire sur de nouvelles données
        return self.model.predict(test_data)


def comparer_solveurs(graphes, apprentissage_model):
    #'SolveurPLNE': SolveurPLNE,
    #, 'SolveurCycle': SolveurCycle
    solveurs = {'SolveurPLNE': SolveurPLNE, 'SolveurPLNECP': SolveurPLNECP, 'SolveurPLNECPM': SolveurPLNECPM, 'SolveurMartin': SolveurMartin}
    resultats = {}

    # Évaluation des solveurs
    for nom, fonction_solveur in solveurs.items():
        solutions = []
        temps_debut = time.time()
        for graphe in graphes:
            G = creer_graphe(graphe)
            nb_aretes = nombre_arcs(G)
            nb_sommet = nombre_sommets(G)
            solveur = SolveurCycle(nb_sommet, nb_aretes, G)
            graphe_obtenu = solveur.solve()
            solutions.append(graphe_obtenu)
        temps_fin = time.time()

        #justesse = calcule_justesse(graphes, solutions, fonction_solveur)

        resultats[nom] = {'temps': temps_fin - temps_debut}

    # Evaluation pour SolveurCycle
    solutions = []
    temps_debut = time.time()
    for graphe in graphes:
        G = creer_graphe(graphe)
        nb_aretes = nombre_arcs(G)
        nb_sommet = nombre_sommets(G)
        solveur = SolveurCycle(nb_sommet, nb_aretes, G)
        graphe_obtenu = solveur.solve()
        solutions.append(graphe_obtenu)
    temps_fin = time.time()
    #justesse = calcule_justesse(graphes, solutions, 'cycle')
    resultats['SolveurCycle'] = {'temps': temps_fin - temps_debut}

    # Évaluation pour SVM
    temps_debut = time.time()
    predictions = []
    for graphe in graphes:
        G = creer_graphe(graphe)
        features_graphe_test = extract_features_graph(G)
        predictions.append(apprentissage_model.predict(features_graphe_test))
    temps_fin = time.time()

    #justesse = calcule_justesse(graphes, predictions, 'svm')
    #, 'justesse': justesse
    resultats['Apprentissage SVM'] = {'temps': temps_fin - temps_debut}

    return resultats


def convertir_en_graphe_networkx(donnees_solution, solveur):
    if solveur == 'cycle' or solveur == 'SolveurPLNECP':
        return donnees_solution
    else:
        G = nwx.Graph()
        G.add_edges_from(donnees_solution)
        return G


def calcule_justesse(graphes, solutions, solveur):
    justesse = 0
    for solution in solutions:
        # Conversion des solutions en graphes NetworkX
        #solutions_networkx = convertir_en_graphe_networkx(solution, solveur)

        # on vérifie si la solution obtenue est connexe
        est_connexe = nwx.is_connected(solutions)

        # on vérifie si la solution obtenue contient des cycles
        contient_cycle = len(list(nwx.cycle_basis(solutions))) > 0

        # Calculer la justesse
        if est_connexe and not contient_cycle:
            justesse = justesse + 1
        else:
            justesse = justesse

    justesse = justesse / len(graphes)

    return justesse


# Partie pour tester par apprentissage :
list_files_all = os.listdir('/Users/typh/Documents/OPTU2S5/Instances/Spd_Inst_Rid_Final2/')
#on filtre pour enlever le fichier .DS_Store (sur mac...)
list_files = [f for f in list_files_all if f.endswith('.txt')]
# Calcule le nombre de graphes à sélectionner
#nombre_a_choisir = int(len(list_files) * 0.3)
# Sélection aléatoire des graphes qui nous serviront à entraîner le modèle
#graphes_choisis = random.sample(list_files, nombre_a_choisir)
# Obtient la liste des graphes non choisis (sur lesquels on veut prédire les arêtes)
#graphes_non_choisis = [f for f in list_files if f not in graphes_choisis]

#apprentissage = Apprentissage(graphes_choisis)
#apprentissage.train_model()


#results = comparer_solveurs(graphes_non_choisis, apprentissage)
#print(results)

# on peut maintenant prédire sur un des autres arbres
#graphe_test = random.sample(graphes_non_choisis, 1)
#print(graphe_test[0])
#G_test = creer_graphe(graphe_test[0])
#G_test = creer_graphe('Spd_RF2_40_50_611.txt')
#features_graphe_test = extract_features_graph(G_test)
#predictions = apprentissage.predict(features_graphe_test)
#results = pd.DataFrame({'edges': G_test.edges(), 'is_in_solution': predictions})
#print(results)

#on regarde à quoi ressemble le graphe obtenu
#G_result = nwx.Graph()
#G_result.add_nodes_from(G_test.nodes())
#for i in results.index:
    #if results['is_in_solution'][i] == 1:
        #(u, v) = results['edges'][i]
        #G_result.add_edge(u, v)
#nwx.draw(G_result, with_labels=True, edgelist=G_result.edges())
#plt.show()





#pour tester
G = creer_graphe("Spd_RF2_20_34_267.txt")
nb_aretes = nombre_arcs(G)
nb_sommet = nombre_sommets(G)

#solveur = SolveurPLNE(nb_sommet, nb_aretes, G)
#solveur = Solveur1(G)
#solveur.main()

solveur = SolveurPLNECPM(nb_sommet, nb_aretes, G)
solveur.defineProblem()
solveur.objectiveFunction()
solveur.addConditions()
results = solveur.solve()
print('hop')
print(G.edges())
edges = G.edges()
print(edges[0])
G_result = nwx.Graph()
G_result.add_nodes_from(G.nodes())
for i in range(len(results)):
    if results[i] == 1.0:
        (u, v) = edges[i]
        G_result.add_edge(u, v)
nwx.draw(G, with_labels=True, edgelist=G_result.edges())
plt.show()

#list_files = os.listdir('/Users/Documents/Optimisation/Instances/Spd_Inst_Rid_Final2/')
#for i in range(3):
    #file = random.choice(list_files)
    #print(file)
    #G = creer_graphe(file)
    #nb_aretes = nombre_arcs(G)
    #nb_sommet = nombre_sommets(G)
    #solveur = Solveur1(nb_sommet, nb_aretes, G)
    #solveur.main()
