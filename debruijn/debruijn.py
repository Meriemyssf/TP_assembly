#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Meriem YOUSSEF"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Meriem YOUSSEF"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Meriem YOUSSEF"
__email__ = "mariem.youssef1207@gmail.com"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with fastq_file.open('r') as file:
        while True:
            # Read the 4 lines of a FASTQ entry
            header = file.readline().strip()
            sequence = file.readline().strip()
            plus_line = file.readline().strip()  # the '+' line
            quality = file.readline().strip()

            if not sequence:  # Check for end of file
                break
                
            yield sequence


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    # Sliding window : 
    for i in range(len(read)-kmer_size+1):
        yield read[i:i+kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    dict_kmer = dict()
    for read in read_fastq(fastq_file):
        for kmer in cut_kmer(read, kmer_size):
            dict_kmer[kmer]=0
        for kmer in cut_kmer(read, kmer_size):
            dict_kmer[kmer]+=1
    return dict_kmer


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()
    for kmer, count in kmer_dict.items():
        prefix = kmer[:len(kmer)-1]
        suffix =  kmer[1:len(kmer)]
        graph.add_edge(prefix,suffix, weight = count)
    return graph
    


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        # Tous les nœuds d’un chemin sont supprimés.
        if delete_entry_node and delete_sink_node:
            graph.remove_nodes_from(path)

        # Tous les nœuds d’un chemin sont supprimés sauf le premier et le dernier.
        elif not delete_entry_node and not delete_sink_node:
            graph.remove_nodes_from(path[1:len(path)-1]) 
        # Tous les nœuds d’un chemin sont supprimés sauf le dernier nœud.
        elif delete_entry_node:
            graph.remove_nodes_from(path[:len(path)-1])
        elif delete_sink_node:
            graph.remove_nodes_from(path[1:])
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    stdev_weights = statistics.stdev(weight_avg_list)
    indice_max_path = 0
    max_val = 0
    if stdev_weights>0:
        for i in range(len(weight_avg_list)):
            if weight_avg_list[i]>max_val:
                indice_max_path=i
                max_val = weight_avg_list[i]
    else:
        stdev_length = statistics.stdev(path_length)
        if stdev_length > 0:
            #Chemin le plus long : 
            for i in range(len(path_length)):
                if path_length[i]>max_val:
                    indice_max_path=i
                    max_val = path_length[i]
        else : 
            indice_max_path = random.randint(0, len(path_length))
    path_to_remove = path_list[:indice_max_path] + path_list[indice_max_path+1:]
    # Filtering the graph:
    graph_cleaned = remove_paths(graph, path_to_remove, delete_entry_node, delete_sink_node)
    return graph_cleaned



def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    path_length = []
    path_weight = []
    path_list = list(all_simple_paths(graph, ancestor_node, descendant_node))
    for path in path_list:
        path_length.append(len(path))
        path_weight.append(path_average_weight(graph, path))
    
    graph_cleaned = select_best_path(graph, path_list, path_length, path_weight, delete_entry_node=False, delete_sink_node=False)
    return graph_cleaned


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    ancestor = None
    node_stock = None
    for node in list(graph.nodes):
        predecesseurs = list(graph.predecessors(node))
        if len(predecesseurs) > 1:
            for i in range(len(predecesseurs) - 1):
                for j in range(i + 1, len(predecesseurs)):
                    ancestor = lowest_common_ancestor(graph, predecesseurs[i], predecesseurs[j])
                    node_stock=node
                    if ancestor is not None:
                        bubble=True
                        break
        if bubble:
            break
    if bubble:
        # Deleting the node : 
        graph = solve_bubble(graph, ancestor, node_stock)
        graph = simplify_bubbles(graph)
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in list(graph.nodes):
        all_paths_for_node = []
        predecessors = list(graph.predecessors(node))
        if len(predecessors)>1:
                for start_node in starting_nodes:
                    if node not in starting_nodes:
                        paths =  list(all_simple_paths(graph, start_node, node))
                        # Append the first path found (assuming there is only one relevant path)
                        all_paths_for_node.append(paths[0])
                # If there are multiple paths for the current node
                if len(all_paths_for_node)>1 :
                    path_length = [len(path) for path in all_paths_for_node]
                    # If path length is 1, we use the edge weight; otherwise, we calculate the average weight of the path
                    path_weigths = [path_average_weight(graph, all_paths_for_node[i])  
                                    if path_length[i] >1 else graph[paths[i][0]][paths[i][1]]["weight"] 
                                    for i in range(len(all_paths_for_node)) ]
                    graph = select_best_path(graph, all_paths_for_node,path_length, 
                                             path_weigths, delete_entry_node=True, delete_sink_node=False)
                    break
    return graph



def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in list(graph.nodes):
        all_paths_for_node = []
        successors = list(graph.successors(node))
        if len(successors)>1:
                for end_node in ending_nodes:
                    if node not in ending_nodes:
                        paths =  list(all_simple_paths(graph, node, end_node))
                        # Append the first path found (assuming there is only one relevant path)
                        all_paths_for_node.append(paths[0])
                # If there are multiple paths for the current node
                if len(all_paths_for_node)>1 :
                    path_length = [len(path) for path in all_paths_for_node]
                    # If path length is 1, we use the edge weight; otherwise, we calculate the average weight of the path
                    path_weigths = [path_average_weight(graph, all_paths_for_node[i])  
                                    if path_length[i] >1 else graph[paths[i][0]][paths[i][1]]["weight"] 
                                    for i in range(len(all_paths_for_node)) ]
                    graph = select_best_path(graph, all_paths_for_node,path_length, 
                                             path_weigths, delete_entry_node=False, delete_sink_node=True)
                    break
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    list_statrting_nodes = []
    nodes = list(graph.nodes())
    for node in nodes:
        if not any(graph.predecessors(node)):
            list_statrting_nodes.append(node)
    return list_statrting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    list_sink_nodes = []
    nodes = list(graph.nodes())
    for node in nodes:
        if not any(graph.successors(node)):
            list_sink_nodes.append(node)
    return list_sink_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []

    # Iterate over all pairs of starting and sink nodes
    for starting_node in starting_nodes:
        for ending_node in ending_nodes:
            # If there is a pathway between the two nodes = contig
            if has_path(graph, starting_node, ending_node):
                for path in all_simple_paths(graph, starting_node, ending_node): 
                    contig = path[0]
                    for i in range(1, len(path)):
                        contig+=path[i][-1]
                    contigs.append((contig,len(contig)))
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w") as f_out :
        for i in range(len(contigs_list)) :
            seq, longueur = contigs_list[i]
            # Writing the header:
            f_out.write(f">contig_{i} len={longueur}\n")
            f_out.write(f"{textwrap.fill(seq, width=80)}\n")



def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()
    print(f"Fichier FASTQ : {args.fastq_file}")
    print(f"Taille des k-mers : {args.kmer_size}")
    print(f"Fichier de sortie : {args.output_file}")
    if args.graphimg_file:
        print(f"Fichier de graphique : {args.graphimg_file}")
    # Build the graph:
    dict_kmer = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(dict_kmer)
    # Get starting and ending nodes
    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)
    # Simplify graph's structures 
    graph = simplify_bubbles(graph)
    graph = solve_entry_tips(graph, starting_nodes)
    graph = solve_out_tips(graph, ending_nodes)
    # Reset the starting and ending nodes
    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)
    # Generate and save contigs
    contigs = get_contigs(graph, starting_nodes, ending_nodes)
    save_contigs(contigs, args.output_file)
    # Plot the graph
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)

if __name__ == "__main__":  # pragma: no cover
    main()
