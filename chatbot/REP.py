import csv
import time
import pandas as pd

import os

FILEPATH = "rep_model/reputationDict.csv"
DEFAULT_FILEPATH = 'rep_model/default_rep.csv'

fieldnames = ['source', 'reputation', 'size', 'articles']

class opinion:
    def __init__(self, sourceName, articleId, stance):
        self.sourceName = sourceName.lower()
        self.articleId = articleId
        self.stance = stance

class source:
    """articles #list of strings
    size #int
    reputation #number between -1 and 1 inclusive"""
    def __init__(self, sourceName, reputation, size, articles):
        # print(type(articles))
        # TODO: fix this
        if type(articles) is str:
            articles = articles.split(',')

        self.sourceName = sourceName
        self.reputation = float(reputation)
        self.articles = articles
        self.size = int(size)
    def addArticle(self, articleId, articleValidity):
  
        self.reputation = (self.reputation*self.size+articleValidity)/(self.size+1)


        if not articleId in self.articles:
            self.reputation = (self.reputation*self.size+articleValidity)/(self.size+1)
            self.articles.append(articleId)
            self.size += 1
    # os.remove("rep_model/reputationDict.csv")

class globals:
    sources = {}
    # sources = {'New York Times' : source('New York Times', 1, 300, 0)}


def returnOutput(mlOut):
    """takes the output of our ml and turns it into a final stances
    :param mlOut: a panda dataframe
    """

    loadReputations(FILEPATH)
    for index, row in mlOut.iterrows():
        stance = row['Stance']
        articleId = index
        # articleId = row['Body ID']
        sourceName = row['source']
        op = opinion(sourceName, articleId, stance)
        if index == 0:
            opinions = [op]
        else:
            opinions.append(op)
    stance = avgStance(opinions)
    updateRep(opinions)
    writeToDisk(FILEPATH)
    
    return stance

def avgStance(opinions):
    """takes a list of opinions and calculates the final stance
    :param opinions: a list<opinion> of all opinions to average
    """
    """finalStance #to hold our final stance"""
    finalStance = 0
    for op in opinions:
        # print(type(op))
        # if op.sourceName in globals.sources:
        #     print(globals.sources.get(op.sourceName).reputation)
        #     print(type(globals.sources.get(op.sourceName).reputation))
        #agr
        if op.stance == 0:
            if op.sourceName in globals.sources:
                finalStance -= globals.sources.get(op.sourceName).reputation
        #agree
        elif op.stance == 1:
            if op.sourceName in globals.sources:
                finalStance += globals.sources.get(op.sourceName).reputation
        # discuss
        elif op.stance == 2:
            if op.sourceName in globals.sources:
                finalStance -= globals.sources.get(op.sourceName).reputation/4
    finalStance = finalStance/len(opinions)
    return finalStance

def compareStance(opinion, opinions):
    """compares an article with other articles to determine its reputability
    :param opinion: the article who's validity is to be determined
    :param opinions: the articles to check the article in question against
    """
    finalStance = 0
    for op in opinions:
        if op.sourceName in globals.sources:
            #disagree
            if op.stance == 0:
                if opinion.stance == 0:
                    finalStance += globals.sources.get(op.sourceName).reputation
                elif opinion.stance == 1:
                    finalStance -= globals.sources.get(op.sourceName).reputation
            #agree
            if op.stance == 1:
                if opinion.stance == 1:
                    finalStance += globals.sources.get(op.sourceName).reputation
                elif opinion.stance == 0:
                    finalStance -= globals.sources.get(op.sourceName).reputation
    finalStance = finalStance/len(opinions)
    return finalStance

def updateRep(opinions):
    for op in opinions:
        if not op.sourceName in globals.sources:
            globals.sources.update({op.sourceName : source(op.sourceName, 0, 1, [])})
        globals.sources.get(op.sourceName).addArticle(op.articleId, compareStance(op, opinions))

def loadReputations(filepath):
    with open(filepath) as csvfile:
        # fieldnames = ['source', 'reputation', 'size', 'articles']
        reader = csv.DictReader(csvfile, fieldnames = fieldnames)
        next(reader, None)  # skip the headers
        # TODO: fix this ignore first row
        # isHead = True
        for row in reader:
            # if isHead:
            #     next
            #     isHead = False
        
            globals.sources[row['source']] = source(row['source'], row['reputation'], row['size'], row['articles'])
            # globals.sources[row['source']].size = 100 #Only for defaults

#  run it the first time to get a fresh csv
def loadDefaultRepsFromDisk(filepath):
    with open(filepath) as csvfile:
        # fieldnames = ['source', 'reputation', 'size', 'articles']
        reader = csv.DictReader(csvfile, fieldnames = fieldnames)
        for row in reader:
#            print(row['source'])
            globals.sources[row['source']] = source(row['source'], row['reputation'], 100, [])
    writeToDisk(FILEPATH)

def writeToDisk(filepath):
    with open(filepath, 'w') as csvfile:
        # fieldnames = ['source', 'reputation', 'size', 'articles']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()	   
        for k in globals.sources.keys():	  
            writer.writerow({'source': k, 'reputation':
                globals.sources.get(k).reputation, 'size': globals.sources.get(k).size, 'articles': globals.sources.get(k).articles })


def dumpRepTable():
     with open('rep_model/reputationDict.csv') as csvfile:
        # fieldnames = ['source', 'reputation', 'size', 'articles']
        reader = csv.DictReader(csvfile, fieldnames = fieldnames)
        
        # for row in reader:
        #    print(type(row['source']))
            # globals.sources[row['source']] = source(row['source'], row['reputation'], 100, [])
        # print(len(reader))
# dumpRepTable()
loadDefaultRepsFromDisk(DEFAULT_FILEPATH)