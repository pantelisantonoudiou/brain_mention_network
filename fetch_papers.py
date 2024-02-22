# -*- coding: utf-8 -*-

#### ---- Import ---- ####
import time
import json
import pandas as pd
from Bio import Entrez
#### ---------------- ####

def xml_to_df(article_details):
    # Iterate through fetched articles to extract information
    articles_data = []
    for article in article_details['PubmedArticle']:
        pmid = article['MedlineCitation']['PMID']
        title = article['MedlineCitation']['Article']['ArticleTitle']
        # Adjusted part for abstract extraction
        abstract_parts = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
        if abstract_parts:  # Check if there are any parts to the abstract
            # Concatenate all parts of the abstract, handling both string and dictionary types
            abstract = ' '.join([part if isinstance(part, str) else part.get('#text', '') for part in abstract_parts])
        else:
            abstract = 'No Abstract'
        authors_list = article['MedlineCitation']['Article'].get('AuthorList', [])
        authors = "; ".join([f"{author.get('LastName', '')}, {author.get('ForeName', '')}" for author in authors_list])
        publication_date_info = article['MedlineCitation']['Article'].get('ArticleDate', [])
        publication_date = publication_date_info[0]['Year'] if publication_date_info else 'No Date'
        journal = article['MedlineCitation']['Article']['Journal']['Title']
        doi = 'No DOI'
        for article_id in article['PubmedData'].get('ArticleIdList', []):
            if article_id.attributes.get('IdType') == 'doi':
                doi = str(article_id)
                break
    
        # Append article data to the list
        articles_data.append({
            'PMID': pmid,
            'Title': title,
            'Abstract': abstract,
            'Authors': authors,
            'PublicationDate': publication_date,
            'Journal': journal,
            'DOI': doi
        })
    
    # Create a DataFrame from the collected article data
    df = pd.DataFrame(articles_data)
    return df

def fetch_details_in_batches(id_list, batch_size=100, wait_time=1):
    all_dfs = []  # This will hold DataFrames for each batch

    for start in range(0, len(id_list), batch_size):
        end = min(start + batch_size, len(id_list))
        print(f"--> Fetching records: {start+1} to {end} of {len(id_list)}")
        
        try:
            fetch_handle = Entrez.efetch(db="pubmed", 
                                         id=",".join(id_list[start:end]), 
                                         retmode="xml", 
                                         rettype="abstract")
            batch_articles = Entrez.read(fetch_handle)
            fetch_handle.close()

            # Convert batch_articles to DataFrame and store it
            batch_df = xml_to_df(batch_articles)
            all_dfs.append(batch_df)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

        time.sleep(wait_time)

    # Concatenate all DataFrames into one
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame()  # Return an empty DataFrame if no articles were fetched

    return final_df



if __name__ == '__main__':
    
    # Get settings
    with open('settings.json') as f:
        settings = json.load(f)
    Entrez.email = settings['email']
    Entrez.api_key = settings['api_key']
    
    # Construct query and search for articles
    tic = time.time() # start timer
    query_terms = ('("functional connectivity") AND ' +
                   '("depression" OR "major depressive disorder" OR ' +
                   '"suicide" OR "anxiety" OR "ADHD" OR "OCD" OR "schizophrenia" '  +
                   ' OR "post-traumatic stress" OR "PTSD" OR "bipolar disorder")')
    
    # Find number of records
    search_handle = Entrez.esearch(db="pubmed", term=query_terms, retmax=0)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    total_records = int(search_results["Count"])
    print(f"--> Total records found: {total_records}.\n")
    
    # Query all matching records
    search_handle = Entrez.esearch(db="pubmed", term=query_terms, retmax=total_records)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    print('--> Full query performed. \n')
    
    # Fetch all data from query
    id_list = search_results['IdList']
    df = fetch_details_in_batches(id_list, batch_size=100, wait_time=.2)
    
    # Convert fetched data to csv and store
    file_name = 'compiled_articles.csv'
    df.to_csv(file_name, index=False)
    
    print(f"--> Files Saved to {file_name}: Time elapsed = {time.time()-tic}.\n")




