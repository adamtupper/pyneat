#!/usr/bin/env python
#**************************************************#
# A simple script to help in executing the         #
# same experiment for a number of times.           #
#**************************************************#
import math, sys
import re, os

p = re.compile('\d*\d')

total_gens = []
total_nodes = []
total_conns = []
total_evals = []
total_score = []


def average(values):
    ''' Returns the population average '''
    sum = 0.0
    for i in values:
        sum += i
    return sum/len(values)


def stdev(values):
    ''' Returns the population standard deviation '''
    # first compute the average
    u = average(values)
    error = 0.0
    # now compute the distance from average
    for x in values:
        error += (u - x)**2
    return math.sqrt(error/len(values))


def report():
    print(f"\nNumber of runs: {sys.argv[2]}\n")
    print("\t Gen. \t Nodes \t Conn. \t Evals. \t Score \n")
    print(f"average  {average(total_gens):.2f} \t {average(total_nodes):.2f} \t {average(total_conns):.2f} \t {average(total_evals):.2f} \t {average(total_score):.2f}")
    print(f"stdev    {stdev(total_gens):.2f} \t {stdev(total_nodes):.2f} \t {stdev(total_conns):.2f} \t {stdev(total_evals):.2f} \t {stdev(total_score):.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("\nUsage: run.py experiment.py number_of_runs\n")
        sys.exit(0)

    print(f"\nExecuting {sys.argv[1]} for {sys.argv[2]} times\n")
    print("    ==========================================================")
    print("\t N. \tGen. \t Nodes \t Conn. \t Evals.    Score")

    for i in range(int(sys.argv[2])):
        output = os.popen('python '+sys.argv[1]).read()
        try:
            gens, nodes, conns, evals, score = p.findall(output)
        except:  # if anything goes wrong
            print(output)
            if len(output) == 0:
                print("Maximum number of generations reached - got stuck")

        total_gens.append(float(gens))
        total_nodes.append(float(nodes))
        total_conns.append(float(conns))
        total_evals.append(float(evals))
        total_score.append(float(score))
        sys.stdout.flush()
        print(f"\t {i+1} \t {gens} \t {nodes} \t {conns} \t {evals} \t {score}")

    print("    ==========================================================")
    report()
