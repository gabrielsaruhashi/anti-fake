from eventregistry import *
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
import json
import pandas as pd
from newspaper import Article

import nltk
import requests
from IPython.display import display, HTML
from util import stop_words

global_df = pd.DataFrame()
eventregistry_api = 'aa1174a5-a633-492d-a182-28d59fc28a34'
watson_api = 'VDaX2sFbNmoKB3gQit67DgC_2y5hnsEwLA1RU49Ko65H'
bodies_df = pd.DataFrame()

CLAIM_CHECK = 0
ARTICLE_CHECK = 1
def getInputArticleKeywords(user_url):
#     url = user_url.decode('utf-8')
    article = Article(user_url)
    article.download()
    article.parse()
    article.nlp()

    keywords = article.keywords
    kws = []
    for word in keywords:
        if len(kws) < 10:
            kws.append(word)

    return kws


def getArticles(keywords):
    global global_df
    global bodies_df
#     bodies_df = pd.DataFrame()
    
    er = EventRegistry(apiKey = eventregistry_api)
    q = QueryArticlesIter(
        keywords = QueryItems.OR(keywords))
        # keywordsLoc = "title",
        # ignoreKeywords = "SpaceX",
        # sourceUri = "nytimes.com")
    q.setRequestedResult(RequestArticlesInfo(sortBy="rel"))
    res = q.execQuery(er, sortBy = "rel", maxItems = 500)
    local_df = pd.DataFrame()
    local_body_df = pd.DataFrame()
    index = 0
    body_count = 0
    
#     print(body_count)
    for article in res:
        data = {
            'source': article['source']['title'].encode('utf-8'),
            'url' : article['url'].encode('utf-8'),
            'text' : article['body'].encode('utf-8')
        }
        
        body = {
            'articleBody': article['body'],
            'Body ID': body_count
        }
        
        local_body_df = pd.concat([local_body_df, pd.DataFrame(body, index=[index])])
        local_df = pd.concat([local_df, pd.DataFrame(data,index=[index])])
#         index += 1
        body_count += 1
    
    # append to global dataframe
    global_df = pd.concat([global_df,local_df])
    bodies_df = pd.concat([bodies_df, local_body_df])
    
# def getKeywords(url):
#     natural_language_understanding = NaturalLanguageUnderstandingV1(
#         version='2018-11-16',
#         iam_apikey=watson_api,
#         url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api'
#     )

#     response = natural_language_understanding.analyze(
#         url=url,
#         features=Features(keywords=KeywordsOptions())).get_result()

#     keywords = []
    
#     for keyword in response['keywords']:
#         if keyword['relevance'] > 0.8 and len(keywords) < 10:
#             keywords.append(keyword['text'].encode('utf-8'))
#     return keywords
    
# given the url of 
def webscrapeMain(evidence, mode):
    '''Main function for webscraping. Given an URL , 
    get back bodies of articles related to the problem'''
    global global_df
    # get keywords 
    if mode == ARTICLE_CHECK:
        kws = getInputArticleKeywords(evidence)
    else:
        kws = evidence
#     kws = getKeywords(url)
    print("The keywords are: {}".format(kws))

    getArticles(kws)
    print(bodies_df)
    global_df = global_df.reset_index(drop=True)
    global_df.to_csv('articles.csv')
    bodies_df.to_csv('bodies.csv')


def generateClaimCSV(claim):
    data = pd.read_csv("articles.csv")
    out = pd.DataFrame()
    # get the total number of articles
    claims = [claim] * len(data.index)
    
#     claims = pd.DataFrame(claim)
    out['id'] = range(len(data.index))
    out['Headline'] = claims
    out['Body ID'] = range(len(data.index))
    out.to_csv('claims.csv')
    print("Succesfully generated claims")

# claim = 'Obama is not American'
# generateClaimCSV(claim)

def azureClaimSearch(claim):
    start_time = time.time()
    # tokenize string
    tokens = claim.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    webscrapeMain(tokens, CLAIM_CHECK)
    # azure_pd = pd.DataFrame()

    # azure_key = '34d4fdab594e46c2b8f4b497042a7260'
    # search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"

    # headers = {"Ocp-Apim-Subscription-Key" : azure_key}
    # params  = {"q": claim, "textDecorations":True, "textFormat":"HTML"}
    # response = requests.get(search_url, headers=headers, params=params)
    # response.raise_for_status()
    # search_results = response.json()
    
    
    # for article in search_results["webPages"]["value"]:
    #     webscrapeMain(article['url'])
    #     # print(json.dumps(article, indent=4))
    #     break
    # generate csv claim to run against the model
    generateClaimCSV(claim)

    print("Webscraping time--- %s seconds ---" % (time.time() - start_time))
