基于贝叶斯的垃圾邮件分类

##理论基础

令A表示邮件是垃圾邮件，令B表示邮件的内容。
如果P(A|B) > P(¬A|B)，即给定一封邮件的内容，
它是垃圾邮件的概率大于是非垃圾邮件的概率，
我们就认为该邮件是垃圾邮件。
因为对于上面的比较，贝叶斯公式都要除以P(B)，
因此我们只需要比较P(A)P(B|A) > P(¬A)P(B|¬A)。

##代码结构

- bayesSpamFilter：主目录
- training_dataset: 训练集目录，包含两种邮件
- test_dataset: 测试集目录，包含两种邮件
- classifier.py：贝叶斯分类器
- train.py：训练测试过程

##代码解析

首先我们来定义一个BayesClassifier类，及其初始化的方法：

```
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
```

在初始化方法中，我们定义变量以记录训练集的一些信息：训练集邮件的数量；训练集中垃圾邮件的数量，非垃圾邮件的数量；记录从训练集中得到的先验P(A)和P(¬A); 记录训练集的单词信息，为计算P(B|A)和P(B|A)和P(B|¬A)奠定基础。

接下来我们先来求先验P(A)和P(¬A)。P(A)表示的是所有样本中出现垃圾邮件的概率，P(¬A)表示所有样本中出现非垃圾邮件的概率。我们只需要求出P(A)，也就求出了P(¬A)。对于P(A)，我们只需要统计下邮件的总数和其中垃圾邮件的数目，做个除法就可以了。我们在上面的BayesClassifier类中定义一个函数train来进行贝叶斯的训练过程，在其中求出P(A)和P(¬A)。

```
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
```

这样我们就求出来了P(A)和P(¬A)，那么如果求出P(B|A)和P(B|¬A)呢？P(B|A)的含义是：给你一封待判断的邮件，我可以假定它是垃圾邮件，在此前提下，它的内容确实像是垃圾邮件内容的概率。我们可以比对该邮件的内容与训练集中垃圾邮件的内容来确定这个概率。那么怎么衡量呢？

首先我们从训练集中构造两个字典：训练集中垃圾邮件中的单词集合及每个单词对应的的数目；训练集中非垃圾邮件中的单词集合及每个单词对应的的数目。然后对于一封测试邮件，我们先假定它是垃圾邮件，然后统计该邮件的每个单词在训练集垃圾邮件字典中出现的概率，这些概率相乘之后作为P(B|A)的值；同理也可求出P(B|¬A)。

如果有不理解的地方可以先看代码的实现，再回头来看就容易理解了。

上面代码中的processEmail函数就是用来从训练集中构造字典的，我们先来实现这个函数：

```
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
```

这里我们就构造完成了对于训练集的两个字典。接下来我们实现一个函数统计测试集一封邮件中的某单词在训练集中某种字典中出现的概率。

```
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
```

注意上面函数中的alpha值，我们引入该值的目的是为了解决单词不存在的情况。如果没有alpha，那么对于一个在训练集中不存在的来自于测试邮件的单词，trainPositive[word]或者trainNegative[word]为为0，导致最终的概率为0（其实就是拉普拉斯平滑）。

有了这个conditionWord函数，我们就可以定义另一个函数conditionalEmail来求解P(B|A)或者P(B|¬A)了：

```
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
```

最后我们来实现对于一封测试集邮件的分类函数：

```
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
        if isSpam > notSpam:
            return 0  #SPAM
        return 1  #HAM
```

至此我们的整个BayesClassifier分类器就完成了。值得注意的是计算P(B|A)和P(B|¬A)的策略有很多种，有的方法会考虑一些关键性的词语，比如sex，budget，sale等等；有的会考虑一些词组，短语，甚至句子。

最后我们考虑如何使用这个贝叶斯分类器。思路还是比较简单的：对于每封邮件，读取邮件的内容，构造出单词列表，构造相应的标签。然后训练贝叶斯分类器，最后进行预测。这里要注意的是预处理的过程，因为邮件的内容中很可能包含各种html的标签啊，等等等等。

```
import classifier
import os
import bs4
import re

training_dataset_spam = "training_dataset/spam"
training_dataset_ham = "training_dataset/ham"
    
def readDatas(path, label):
    """读取邮件，构造标签"""

    trainDatas = []
    emails = os.listdir(path)
  
    for email in emails:
        email_path = os.path.join(path, email)
        #以二进制读取可以避免读取时遇到的编码问题
        #也可以尝试直接使用utf-8等等格式读取
        email_text = open(email_path,"rb").read() 
        try:
            email_text = bs4.UnicodeDammit.detwingle(
                        email_text).decode('utf-8') #使用utf-8解码
        except:
            continue
        regEx = re.compile(r'[^a-zA-Z]|\d') #正则表达式
        words = regEx.split(email_text)
        #因为上面提取出来可能出现空的word，len(word)>0是为了排除这种情况
        words = [word.lower() for word in words if len(word) > 0]
        trainDatas.append(words)

    counts = len(trainDatas)
    labels = [label for i in range(counts)]

    return trainDatas, labels
        
trainDatas_spam, labels_spam = readDatas(training_dataset_spam, 0)
trainDatas_ham, labels_ham = readDatas(training_dataset_spam, 1)
trainDatas, labels = trainDatas_spam + trainDatas_ham, labels_spam + labels_ham

model = classifier.BayesClassifier()
model.train(trainDatas, labels)

test_dataset_spam = "test_dataset/spam"
test_dataset_ham = "test_dataset/ham"


def evaluate(test_path, label):
    """进行测试"""
    testDatas, labels = readDatas(test_path, label)
    count = 0
    for email in testDatas:
        result = model.classify(email)
        if result == label:
            count += 1
    return count / len(testDatas)
        

print("Accuracy of SPAM:", evaluate(test_dataset_spam, 0))
print("Accuracy of HAM:", evaluate(test_dataset_ham, 1))
```

现在我们就完成了整个demo。但是我们思考下有哪些可以改进的地方呢？

- 我们发现现在的方法中在求解P(B|A)和P(B|¬A)时依赖于大量的浮点数乘法，我们可以使用log函数将其转化为加法操作；
- 实际上我们使用的求解P(B|A)和P(B|¬A)的策率是bag-of-word模型，这个效果不太好，可以通过使用TF-IDF等进行改进；
- N-Grams技术：考虑N个连续单词的集合，并用该集合来计算概率。比如：bad和not bad表示的其实是不同的含义；

