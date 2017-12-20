"""
bm25 function calculate score between word_list1 and word_list2 which is drawn from texts
Please note that the above formula for IDF shows potentially major drawbacks when using it for terms appearing in more than half 
of the corpus documents. These terms' IDF is negative, so for any two almost-identical documents, one which contains the term and 
one which does not contain it, the latter will possibly get a larger score. This means that terms appearing in more than half of the corpus will provide negative contributions to the final document score.
"""
def bm25(word_list1, word_list2, documents, K1=2.0, b=0.75):	#这里两个参数都可以更改，会影响结果。但b一般是0.75
    def calc_df(documents):		 # number of document contains w
        dfs = {}
        for d in documents:
            for w in set(d):
                if w not in dfs:
                    dfs[w] = 1
                else:
                    dfs[w] += 1
        return dfs
    
    def idf(word):
        df = (word in dfs) and dfs[word] or 0.0
        return math.log((len(documents) - df + 0.5) / (df + 0.5))	#bm25算法里面，判断一个词与整个句子库的相关性的权重，即IDF
    
    totdl = 0.0
    for d in documents:
        totdl += len(d)
    avgdl = totdl / len(documents)		#avgdl=所有词的数量/所有句子的数量
    
    tf = {}
    for w in word_list1:
        tf[w] = 0.0
    for w in word_list2:
        if w in tf:
            tf[w] += 1.0		#统计输入的句子的每个词在已有词库中是否存在，出现多少次。
    bm25 = 0.0
    for k in tf.keys():
        bm25 += idf(k) * (tf[k] * (K1 + 1)) / (tf[k] + K1 * (1 - b + b * len(word_list2) / avgdl))		#bm25算法核心（一般的推导公式），得到的是每个词相关性得分，然后加在一起得到句子的相关性
        #print(k,idf(k))    
    return bm25
