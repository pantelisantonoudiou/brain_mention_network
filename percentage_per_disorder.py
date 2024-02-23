# -*- coding: utf-8 -*-
#### ---- Import ---- ####
import pandas as pd
import matplotlib.pyplot as plt
#### ---------------- ####

# load retrieved abstracts
papers = pd.read_csv('compiled_articles.csv')
abstracts = papers['Abstract']
df = pd.DataFrame(abstracts)
disoder_list = ['depression',  'major depressive disorder', 'suicide', 'anxiety', 'ADHD',
                'OCD', 'schizophrenia', 'PTSD', 'bipolar disorder']

# Count the occurrences of each mental disorder in the abstracts
disorder_counts = {disorder: sum(disorder in abstract for abstract in abstracts) for disorder in disoder_list}

# Filter out disorders with zero counts
filtered_counts = {disorder: count for disorder, count in disorder_counts.items() if count > 0}
filtered_labels = list(filtered_counts.keys())
filtered_sizes = list(filtered_counts.values())

# Plot the pie chart
fig, ax = plt.subplots(figsize=(14, 14))
wedges, texts, autotexts = ax.pie(filtered_sizes, labels=filtered_labels, wedgeprops={'width': 0.4},
                                  autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)

# Set the label colors to match the pie chart sections
for text, wedge in zip(texts, wedges):
    text.set_color(wedge.get_facecolor())
    text.set_fontsize(22)  # Increase the font size for labels
    text.set_weight('bold')  # Set font to bold

# Increase the font size for autotexts (percentage labels)
for autotext in autotexts:
    autotext.set_fontsize(25)
    autotext.set_weight('bold')

plt.title('Distribution of Abstracts by Mental Disorder', fontsize=28, weight='bold')
plt.show()