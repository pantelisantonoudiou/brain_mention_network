# -*- coding: utf-8 -*-

#### ---- Import ---- ####
import os
import pandas as pd
from nilearn import datasets
#### ---------------- ####

### ----------- create brain region list to search abstracts -------------- ###

# load and clean brain regions
# from Harvard-Oxford
ni_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
atlas_harvard_oxford = pd.Series(ni_atlas['labels'][1:])
atlas_harvard_oxford = atlas_harvard_oxford.str.split('_').str[0]
atlas_harvard_oxford = atlas_harvard_oxford.str.split(',').str[0]

# from AAL
ni_atlas = datasets.fetch_atlas_aal()
atlas_aal = pd.Series(ni_atlas['labels'])
atlas_aal = atlas_aal.str.split('_').str[0]

# from AAL3
atlas_aal3 = pd.read_csv(os.path.join('brain_regions', 'aal3_brain_regions.csv'), header=None)
atlas_aal3 = atlas_aal3[0].str.split(',').str[0]

# from Allen
atlas_allen = pd.read_csv(os.path.join('brain_regions', 'allen_brain_regions.csv'))
atlas_allen = atlas_allen['name'][3:].str.split(',').str[0]

# from fMRI
atlas_fmri = pd.read_csv(os.path.join('brain_regions', 'fmri_brain_regions.csv'), header=None)
atlas_fmri = atlas_fmri[0].str.replace("'","")

# concatenate and clean
brain_regions = pd.concat((atlas_harvard_oxford, atlas_aal, atlas_aal3, atlas_allen, atlas_fmri))
brain_regions = brain_regions.str.lower()
brain_regions = brain_regions.str.replace('left ', '')
brain_regions = brain_regions.str.replace('right ', '')
brain_regions = brain_regions.str.strip()
brain_regions = pd.Series(list(brain_regions.unique()) +\
                          ['striatal', 'basal_ganglia', 'prefrontal',
                           'posterior cingulate cortex', 'limbic'])
brain_regions = brain_regions[~brain_regions.isin(['root','supp', 'frontal', 'parietal', 'occipital','temporal', 'limbic'])]
brain_regions = ' ' + brain_regions
brain_regions.to_csv('brain_regions.csv', index=False)
print('--> brain regions csv created and saved as brain_regions.csv')
### ----------------------------------------------------------------------- ###