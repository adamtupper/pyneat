# from neat import config, chromosome, genome
import random, sys
import pickle

import neat

from custom_neat.population import Population
from custom_neat.genome import Genome
from custom_neat.reproduction import Reproduction
from custom_neat.species import SpeciesSet
from custom_neat.config import CustomConfig

from cart_pole import CartPole

if len(sys.argv) > 1:
    # load genome
    try:
        file = open(sys.argv[1], 'r')
    except IOError:
        print(f"Filename: {sys.argv[1]} not found!")
        sys.exit(0)
    else:
        c = pickle.load(file)
        file.close()
else:
    print("Loading winning genome file")
    try:
        file = open('results/solution.pickle', 'rb')
    except IOError:
        print("Winning genome not found!")
        sys.exit(0)
    else:
        c = pickle.load(file)
        file.close()

# load settings file
config = CustomConfig(Genome, Reproduction, SpeciesSet, neat.DefaultStagnation, 'dpnv_config.ini')
print("Loaded genome:")
print(c)
# starts the simulation
simulator = CartPole([c], markov=False)
simulator.run(testing=True)
