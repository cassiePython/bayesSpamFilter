# -*- coding:utf-8 -*-
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





