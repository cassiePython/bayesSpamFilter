from __future__ import division, print_function
from collections import defaultdict
"""
令A表示邮件是垃圾邮件，令B表示邮件的内容。
如果P(A|B) > P(¬A|B)，我们就认为该邮件是垃圾邮件。
因为对于上面的比较，贝叶斯公式都要除以P(B)，
因此我们只需要比较P(A)P(B|A) > P(¬A)P(B|¬A)。
"""
class BayesClassifier:
    
    def __init__(self):

        self.total = 0 #训练集邮件总数
        self.numSpam = 0 #训练集垃圾邮件数量
        self.numHam = 0 #训练集非垃圾邮件数量

        self.pA = 0 #P(A)
        self.pNotA = 0 #P(¬A)

        self.trainPositive = defaultdict(int) #训练集垃圾邮件中的单词集合
        self.trainNegative = defaultdict(int) #训练集非垃圾邮件中的单词集合

        self.positiveTotal = 0 #训练集垃圾邮件中的单词数
        self.negativeTotal = 0 #训练集非垃圾邮件中的单词数
    
    def train(self, trainDatas, trainLabels):
        """训练贝叶斯

        给定训练数据和对应的标签，计算P(A)和(¬A),
        统计垃圾邮件和非垃圾邮件中的单词集合及每种单词的数目

        Args:
            trainDatas: 训练集数据，List列表，内容为包含邮件word的List
            trainLabels: 训练集标签, List列表，0表示SPAM，1表示HAM
        Returns:
            pass
        Raises:
            pass
        """

        self.total = len(trainDatas) #训练集邮件总数

        for index, email in enumerate(trainDatas):
            if trainLabels[index] == 0: #SPAM
                self.numSpam += 1 #统计训练集垃圾邮件数量
            self.__processEmail(email, trainLabels[index])
            
        self.numHam = self.total - self.numSpam #训练集非垃圾邮件数量

        self.pA = self.numSpam / float(self.total) #P(A)
        self.pNoA = 1 - self.pA #P(¬A)

    def __processEmail(self, email, label):
        """统计邮件单词

        给定一封训练集的邮件，计算该邮件中
        的单词集合及相应的单词的数目

        Args:
            email: 训练样本，List列表，内容为邮件的word
            label: 训练样本标签, int，0表示SPAM，1表示HAM
        Returns:
            pass
        Raises:
            pass
        """
        for word in email:
            if label == 0: #SPAM
                self.trainPositive[word] += 1
                self.positiveTotal += 1
            else:
                self.trainNegative[word] += 1
                self.negativeTotal += 1

    def __conditionWord(self, word, label):
        """统计测试集一封邮件中的某单词在训练集中出现的概率

        给定一封训练集的邮件，计算该邮件中
        的单词集合及相应的单词的数目

        Args:
            email: 测试样本，List列表，内容为邮件的word
            label: 测试样本标签, int，0表示SPAM，1表示HAM
        Returns:
            测试集一封邮件中的某单词在训练集中出现的概率
            label=0,则返回该单词在SPAM训练集中出现的概率
            label=1,则返回该单词在HAM训练集中出现的概率
        Raises:
            pass
        """
        alpha = 1.0
        if label == 0: #SPAM
            return (self.trainPositive[word] + alpha) / float(self.positiveTotal)
        return (self.trainNegative[word] + +alpha ) / float(self.negativeTotal)

    def __conditionalEmail(self, email, label):
        """计算P(B|A)或者P(B|¬A)

        给定一封测试集的邮件，计算对于该邮件的P(B|A)或者P(B|¬A)

        Args:
            email: 测试样本，List列表，内容为邮件的word
            label: 测试样本标签, int，0表示SPAM，1表示HAM
        Returns:
            label=0,返回P(B|A)
            label=1,返回P(B|¬A)
        Raises:
            pass
        """
        result = 1.0
        for word in email:
            result *= self.__conditionWord(word, label)
        return result
                
    def classify(self, email):
        """进行分类

        给定一封测试集的邮件，判断该邮件是否是垃圾邮件

        Args:
            email: 测试样本，List列表，内容为邮件的word
            label: 测试样本标签, int，0表示SPAM，1表示HAM
        Returns:
            0：SPAM
            1：HAM
        Raises:
            pass
        """
        isSpam = self.pA * self.__conditionalEmail(email, 0) # P(A)*P(B|A)的值
        notSpam = self.pNotA * self.__conditionalEmail(email, 1) # P(¬A)*P(B|¬A)的值
        #print (isSpam, notSpam)
        if isSpam > notSpam:
            return 0  #SPAM
        return 1  #HAM
        #return isSpam > notSpam
