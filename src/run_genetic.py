#!/usr/bin/python3

import random, string
import sys, os
import subprocess
from subprocess import STDOUT

class GeneticAlgorithm:
    """
    Klasa predstavlja implementaciju genetskog algoritma za resavanje problema 
    odabira parametara za treniranje neuronske mreze za kompresiju slika.
    Koristi se:
    - Uniformno ukrstanje sa verovatnocom '_crossover_p'
    - Mutacija sa verovatnocom '_mutation_rate'
    - Turnirska selekcija sa parametrom '_tournament_k'
    - Zamena generacije se vrsi tako sto se od jedinki izabranih pri selekciji ukrstanjem
        pravi celokupna nova generacija
    """
    def __init__(self, target, allowed_gene_values):
        self._target = target                               
        self._allowed_gene_values = allowed_gene_values     # Dozvoljene vrednosti koje mogu biti u genu
        self._chromoome_length = 4

        """Parametri genetskog algoritma, eksperimentalno izabrani."""
        self._iterations = 1000                             # Maksimalni dozvoljeni broj iteracija
        self._generation_size = 20                          # Broj jedinki u jednoj generaciji
        self._mutation_rate = 0.01                          # Verovatnoca da se desi mutacija
        self._reproduction_size = 10                        # Broj jedinki koji ucestvuje u reprodukciji
        self._current_iteration = 0                         # Koristi se za interno pracenje iteracija algoritma
        self._crossover_p = 0.5                             # Verovatnoca za odabir bita prvog roditelja pri ukrstanju
        self._top_chromosome = None                         # Hromozom koji predstavlja resenje optimizacionog procesa
        self._tournament_k = 2                              # Parametar k za turnirsku selekciju


    def optimize(self):
        # Generisi pocetnu populaciju jedinki i izracunaj
        # prilagodjenost svake jedinke u populaciji
        chromosomes = self.initial_population()

        # Sve dok uslov zaustavljanja nije zadovoljen
        while not self.stop_condition():
            print("Iteration: %d" % self._current_iteration)

            # Izaberi iz populacije skup jedinki za reprodukciju
            for_reproduction = self.selection(chromosomes)

            # Prikaz korisniku trenutnog stanja algoritma
            the_sum = sum(chromosome.fitness for chromosome in chromosomes)
            print("Reproduction chromos sum fitness: %f" % the_sum)
            print("top solution: %s" % max(chromosomes, key=lambda chromo: chromo.fitness))

            # Primenom operatora ukrstanja i mutacije kreiraj nove jedinke
            # i izracunaj njihovu prilagodjenost.
            # Dobijene jedinke predstavljaju novu generaciju.
            chromosomes = self.create_generation(for_reproduction)
            self._current_iteration += 1
            print()

        # Vrati najkvalitetniju jedinku u poslednjoj populaciji
        if self._top_chromosome:
            return Chromosome(self._top_chromosome, self.fitness(self._top_chromosome))
        else:
            return min(chromosomes, key=lambda chromo: chromo.fitness)


    def create_generation(self, for_reproduction):
        """
        Od jedinki dobijenih u okviru 'for_reproduction' generise novu generaciju
        primenjujuci genetske operatore 'crossover' i 'mutation'.
        Nova generacija je iste duzine kao i polazna.
        """
        new_generation = []
        # Sve dok ne popunimo generaciju
        while len(new_generation) < self._generation_size:
            # Biramo dva nasumicno i vrsimo ukrstanje
            parents = random.sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0].content, parents[1].content)

            # Vrsimo mutaciju nakon ukrstanja
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            # Dodajemo nove hromozome u novu generaciju
            new_generation.append(Chromosome(child1, self.fitness(child1)))
            new_generation.append(Chromosome(child2, self.fitness(child2)))

        return new_generation


    def crossover(self, a, b):
        """
        Vrsi uniformno ukrstanje po verovatnoci self._crossover_p.
        """
        #pravimo potomke koji su prvo identicni roditeljima pa menjamo
        ab = a
        ba = b
        for i in range(len(a)):
            p = random.random()
            if p < self._crossover_p:
                ab[i] = a[i]
                ba[i] = b[i]
            else:
                ab[i] = b[i]
                ba[i] = a[i]
        return (ab, ba)


    def mutation(self, chromosome):
        """Vrsi mutaciju nad hromozomom sa verovatnocom self._mutation_rate."""
        t = random.random()
        if t < self._mutation_rate:
            # dolazi do mutacije
            i = random.randint(0, len(chromosome) - 1)
            chromosome[i] = random.uniform(self._allowed_gene_values[i][0], self._allowed_gene_values[i][1])
        return chromosome


    def selection(self, chromosomes):
        """
        Funkcija bira self._reproduction_size hromozoma koristeci turnirsku selekciju
        koji ce se na dalje koristiti za ukrstanje i smenu generacija.
        """
        selected_chromos = []
        selected_chromos = [self.selection_tournament_pick_one(chromosomes, self._tournament_k) for i in range(self._reproduction_size)]

        return selected_chromos


    def selection_tournament_pick_one(self, chromosomes, k):
        """
        Bira jednu jedinku koristeci turnirsku selekciju.
        Ne vrsi normalizaciju i sortiranje po funkciji prilagodjenosti usled performansi.
        Parametar k definise koliko jedinki iz populacije se izvlaci.
        """

        # Biramo k nasumicnih jedinki iz populacije i trazimo jedinku
        # koja ima najvecu funkciju prilagodjenosti
        # Ovo predstavlja jednu od varijanti turnirske selekcije.
        the_chosen_ones = []
        top_i = None
        for i in range(k):
            pick = random.randint(0, len(chromosomes)-1)
            the_chosen_ones.append(chromosomes[pick])
            if top_i == None or the_chosen_ones[i].fitness < the_chosen_ones[top_i].fitness:
                top_i = i
        return the_chosen_ones[top_i]


    def the_goal_function(self, chromosome):
        """
         Za slucaj da smo nasli optimalno resenje (u opstem slucaju ovo retko znamo)
         pamtimo gen kao optimalni i prekidamo algoritam (bice zadovoljen kriterijum zaustavljanja).
        """
        if abs(chromosome - self._target) < 0.001:
            self._top_chromosome = chromosome

    def fitness(self, chromosome):
        return run_train(chromosome[0], chromosome[1],chromosome[2],chromosome[3])

    def take_random_element(self, fromHere):
        """ Funkcija vraca nasumicno odabran element iz kolekcije 'fromHere'. """
        i = random.randrange(0, len(fromHere))
        return fromHere[i]


    def initial_population(self):
        """Generisemo _generation_size nasumicnih jedinki."""
        allowed_values = self._allowed_gene_values
        init_pop = []                   # inicijalna populacija jedinki
        for i in range(self._generation_size):
            genetic_code = []           # genetski kod koji cemo nasumicno izabrati
            for j in range(self._chromoome_length):
                genetic_code.append( \
                        random.uniform(allowed_values[j][0], allowed_values[j][1]) \
                        )
            # U inicijalnu generaciju dodajemo novi genetski kod
            init_pop.append(genetic_code)

        # Transformisemo listu tako da od liste genetskih kodova postane lista hromozoma
        init_pop = [Chromosome(content, self.fitness(content)) for content in init_pop]
        return init_pop


    def stop_condition(self):
        return self._current_iteration > self._iterations or self._top_chromosome != None


class Chromosome:
    """
    Klasa predstavlja jedan hromozom za koji se cuva njegov genetski kod i vrednost funkcije prilagodjenosti.
    """
    def __init__(self, content, fitness):
        self.content = content
        self.fitness = fitness
        print("Created new chromosome: "+str(self))

    def __str__(self): 
        return "\nlearning rate: "+str(self.content[0])+"\n"+\
                "momentum: "+str(self.content[1])+"\n"+\
                "delta: "+str(self.content[2])+"\n"+\
                "weight decay: "+str(self.content[3])+"\n"+\
                "f="+str(self.fitness)+"\n"
    def __repr__(self): 
        return "\nlearning rate: "+str(self.content[0])+"\n"+\
                "momentum: "+str(self.content[1])+"\n"+\
                "delta: "+str(self.content[2])+"\n"+\
                "weight decay: "+str(self.content[3])+"\n"+\
                "f="+str(self.fitness)+"\n"

def run_train(lr=0.5, momentum=0.95, delta=1e-8, weight_decay=0.0005):

    with open("prototxt_files/compress_solver.prototxt", "r") as solver:
        solver_lines = solver.readlines()

        # rezultat list comprehension-a je lista [indeks] pa izvlacimo indeks sa [0]
        lr_index = [i for i, line in enumerate(solver_lines) if "base_lr:" in line][0]
        momentum_index = [i for i, line in enumerate(solver_lines) if "momentum:" in line][0]
        delta_index = [i for i, line in enumerate(solver_lines) if "delta:" in line][0]
        wd_index = [i for i, line in enumerate(solver_lines) if "weight_decay:" in line][0]

        solver_lines[lr_index] = "base_lr: "+str(lr)+"\n"
        solver_lines[momentum_index] = "momentum: "+str(momentum)+"\n"
        solver_lines[delta_index] = "delta: "+str(delta)+"\n"
        solver_lines[wd_index] = "weight_decay: "+str(weight_decay)+"\n"

    with open("prototxt_files/compress_solver.prototxt", "w") as solver:
        for line in solver_lines:
            solver.write("%s" % line)

    train_output = subprocess.check_output(["/home/lale/caffe/build/tools/caffe", "train", "-solver", "prototxt_files/compress_solver.prototxt"], \
            stderr=STDOUT).decode().split('\n')

    loss_line = [line for line in reversed(train_output) if "l2_error =" in line][0]
    loss_line = loss_line[loss_line.index("l2_error =")+11:]
    loss = float(loss_line[0:loss_line.index(" ")])
    return loss

def main():
    genetic = GeneticAlgorithm(0.0000000001, [[0.1,0.9],[0.91,0.99],[1e-11, 1e-5],[0.0001,0.0009]])
    solution = genetic.optimize()
    print("Solution: "+solution)

if __name__ == "__main__":
    main()
