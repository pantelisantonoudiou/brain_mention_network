# -*- coding: utf-8 -*-

#### ---- Import ---- ####
import os
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
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
abstracts_with_found_regions = np.sum(df.sum(axis=1) > 0)
print('Brain regions were detected in', abstracts_with_found_regions, 'abstracts.')

# calculate number of times each brain region is involved
trim_threshold = int(abstracts_with_found_regions*.5/100)
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
wordcloud = WordCloud(width=800, height=1200, background_color='white').generate_from_frequencies(brain_region_counts)
plt.figure(figsize=(10, 20))
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

# Calculate the 90th percentile of node sizes and assign colors
node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
threshold = np.percentile(node_sizes, 90)
small_node_color = '#5495ba'
large_node_color = '#ad2f3e'

node_colors = [small_node_color if size < threshold else large_node_color for size in node_sizes]
edge_colors = []
for u, v in G.edges():
    if G.nodes[u]['size'] >= threshold and G.nodes[v]['size'] >= threshold:
        edge_colors.append(large_node_color)
    else:
        edge_colors.append(small_node_color)
# plot
plt.figure(figsize=(24, 16))
pos = nx.spring_layout(G, k=8, iterations=50)
weight_modifier = 1/35
node_modifier = 5
sizes = [G.nodes[node]['size'] * node_modifier for node in G]
weights = [G[u][v]['weight'] * weight_modifier for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=weights, alpha=.5, edge_color=edge_colors)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
plt.axis('off')



