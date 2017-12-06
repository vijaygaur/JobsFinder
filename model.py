import gensim
import numpy
import pickle
import json
from gensim import corpora, similarities, models

try:
    dictionary = corpora.Dictionary.load('/tmp/dictionary.dict')
    corpus = corpora.MmCorpus('/tmp/corpus.mm')
    lsi=models.LsiModel(corpus,id2word=dictionary, num_topics=10)
    mapping = pickle.load(open('/tmp/mapping.dict','rb'))
except:
    pass

class Model:
    def train(self,data):
        # Create sentences from documents
        docs=json.loads(data)

        sentences = [doc["description"].lower().split() for doc in docs]

        # Store Ids of documents
        mapping = {}
        index=0
        for doc in docs:
            mapping[index]=doc["id"]
            index=index+1

        output = open("/tmp/mapping.dict","wb")
        pickle.dump(mapping,output)
        output.close()

        # Stop Words
        stopwords = set('for a an the of and to in'.split())

        # Tokenize
        words = [[word for word in doc["description"].lower().split() if word not in stopwords] for doc in docs]

        dictionary = corpora.Dictionary(words)
        corpus = [dictionary.doc2bow(word) for word in words]
        corpora.MmCorpus.serialize('/tmp/corpus.mm',corpus)
        dictionary.save('/tmp/dictionary.dict')

        # Model
        lsi = gensim.models.LsiModel(corpus,id2word=dictionary,num_topics=10)

        return data

    def predict(self,doc):
        bow = dictionary.doc2bow(doc.lower().split())
        lsivec = lsi[bow]
        index = similarities.MatrixSimilarity(lsi[corpus])

        similarlist = index[lsivec]
        listdump = json.dumps(similarlist.tolist())

        results=[]
        index=0
        for i in similarlist.tolist():
            result={}
            result["id"]=mapping[index]
            result["score"]=i
            results.append(result)
            index+=1
        retvalue = sorted(results, key=lambda x: x["score"] , reverse=True)
        print(type(retvalue))
        print (retvalue)

        return retvalue

