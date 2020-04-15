import pickle
from visualize import draw_net

genome = pickle.load(open('results/run-0/solution.pickle', 'rb'))
genome.mutate_add_node(None)
draw_net(genome, filename='results/run-0/solution.gv')