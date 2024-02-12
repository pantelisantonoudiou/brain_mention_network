# -*- coding: utf-8 -*-

#### ---- Import ---- ####
import os
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
#### ---------------- ####

# load retrieved abstracts and brain region seed list
papers = pd.read_csv('compiled_articles.csv')
abstracts = papers['Abstract']
brain_regions = pd.read_csv('brain_regions.csv')['0']
print('data loaded.')

# search for brain regions in abstracts
df = pd.DataFrame(data=np.zeros((len(abstracts), len(brain_regions))), columns=brain_regions, dtype=bool)
for region in df.columns:
    df[region] = abstracts.str.contains(region, regex=False, case=False)
print('Searching abstracts completed:')
print('Brain regions were detected in', np.sum(df.sum(axis=1) > 0), 'abstracts.')

# calculate number of times each brain region is involved
trim_threshold = 20
cols = df.columns[df.sum(axis=0) > trim_threshold]
filtered = df[cols].copy()
brain_region_counts = filtered.sum(axis=0)

# calculate co-mentions
co_mentions = pd.DataFrame(0, index=filtered.columns, columns=filtered.columns)
for region1, region2 in combinations(filtered.columns, 2):
    co_mentions.loc[region1, region2] = (filtered[region1] & filtered[region2]).sum()
    co_mentions.loc[region2, region1] = co_mentions.loc[region1, region2]
print('Number of mentions and co-mentions was calculated.')

# display wordcloud
wordcloud = WordCloud(width=1200, height=800, background_color='white').generate_from_frequencies(brain_region_counts)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# =============================================================================
# Display Network
# =============================================================================

# create a network graph with co-mentions as weights and counts as node size
G = nx.Graph()
for region, count in brain_region_counts.items():
    G.add_node(region, size=count)
for region1, region2 in combinations(brain_region_counts.index, 2):
    weight = co_mentions.loc[region1, region2]
    if weight > 0:
        G.add_edge(region1, region2, weight=weight)

# identify the n most connected regions
n_select = 10
degrees = dict(G.degree())
sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:n_select]
nodes_in_strongest_connections = set()
for u, v, data in sorted_edges:
    nodes_in_strongest_connections.add(u)
    nodes_in_strongest_connections.add(v)
node_colors = ['green' if node in nodes_in_strongest_connections else '#89849c' for node in G.nodes()]
edge_colors = ['green' if ((u, v) in [(e[0], e[1]) for e in sorted_edges] or (v, u) in [(e[0], e[1]) for e in sorted_edges]) else 'gray' for u, v in G.edges()]


# Calculate the percentiles of edge weights
weight_modifier = 1/50
node_modifier = 5
weights = nx.get_edge_attributes(G, 'weight').values()*weight_modifier
percentiles = np.percentile(list(weights), [33, 66])
node_categories = {node: 'weak' for node in G.nodes()}
def categorize_weight(weight):
    if weight <= percentiles[0]:
        return 'weak'
    elif weight <= percentiles[1]:
        return 'medium'
    else:
        return 'strong'
for u, v, data in G.edges(data=True):
    category = categorize_weight(data['weight'])
    if category == 'strong':
        node_categories[u] = node_categories[v] = 'strong'
    elif category == 'medium' and node_categories[u] != 'strong' and node_categories[v] != 'strong':
        node_categories[u] = node_categories[v] = 'medium'
        
# Use RdPu colormap for both nodes and edges
cmap = cm.get_cmap('RdPu', 3)  # Get RdPu colormap
color_map = {'weak': colors.to_hex(cmap(0.2)), 'medium': colors.to_hex(cmap(0.5)), 'strong': colors.to_hex(cmap(0.8))}

# Assign colors to nodes and edges based on categories
node_colors = [color_map[node_categories[node]] for node in G.nodes()]
edge_colors = [color_map[categorize_weight(G[u][v]['weight'])] for u, v in G.edges()]
        
        
# plot
plt.figure(figsize=(24, 16))
pos = nx.spring_layout(G, k=10, iterations=50)
sizes = [G.nodes[node]['size'] * node_modifier for node in G]
weights = [G[u][v]['weight'] * weight_modifier for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, alpha=0.7)
nx.draw_networkx_edges(G, pos, width=weights, alpha=.5, edge_color=edge_colors)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
plt.axis('off')



