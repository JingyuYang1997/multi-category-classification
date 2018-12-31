import re
import random
from collections import Counter

'''创建表示级与级之间标签从属关系的字典'''
class1_class2={}
class1_class3={}
class2_class3={}
pattern_class='.*?\t.*?\t.*?\t.*?\t.*?\t(.*?)\t(.*?)\t(.*?)\n'
with open('./data/train_b.txt') as f:
    data=f.read()
    class_all=re.findall(pattern_class,data)[1:]
    class1_all=[item[0] for item in class_all]
    class1_all=set(class1_all)
    class2_all=[item[1] for item in class_all]
    class2_all=set(class2_all)
    for class1_item in class1_all:
        class1_class2[class1_item]=[]
        class1_class3[class1_item]=[]
    for class2_item in class2_all:
        class2_class3[class2_item]=[]
    for class_tuple in class_all:
        if class_tuple[1] not in class1_class2[class_tuple[0]]:
            class1_class2[class_tuple[0]].append(class_tuple[1])
        if class_tuple[2] not in class1_class3[class_tuple[0]]:
            class1_class3[class_tuple[0]].append(class_tuple[2])
        if class_tuple[2] not in class2_class3[class_tuple[1]]:
            class2_class3[class_tuple[1]].append(class_tuple[2])

'''多个ans参与决策树投票'''
pattern='(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
source_files=['./ans/ans_1.txt','./ans/ans_2.txt','./ans/ans_3.txt','./ans/ans_4.txt','./ans/ans_5.txt']
class1=[]
class2=[]
class3=[]

with open(source_files[0]) as f:
    data = f.read()
    data = re.findall(pattern, data)[1:]
    item_id=[data_item[0] for data_item in data]

for source_file in source_files:
    with open(source_file) as f:
        data = f.read()
        data = re.findall(pattern,data)[1:]
        cate1 = [data_item[1] for data_item in data]
        cate2 = [data_item[2] for data_item in data]
        cate3 = [data_item[3] for data_item in data]
        class1.append(cate1)
        class2.append(cate2)
        class3.append(cate3)

results1=[]
results2=[]
results3=[]
for index in range(len(data)):
    '''vote1,vote2,vote3为三级预测最初的票行情况'''
    '''对第一级和第二级进行不弃票处理，投票得到结果'''
    vote1 = [class1[0][index], class1[1][index], class1[2][index], class1[3][index], class1[4][index]]
    vote2 = [class2[0][index], class2[1][index], class2[2][index], class2[3][index], class2[4][index]]
    vote3 = [class3[0][index], class3[1][index], class3[2][index], class3[3][index], class3[4][index]]
    vote1_counter=Counter(vote1)
    vote1_top=vote1_counter.most_common(1)
    result1=vote1_top[0][0]
    if vote1_top[0][1]==1:
        result1=vote1[0]

    vote2_counter = Counter(vote2)
    vote2_top = vote2_counter.most_common(1)
    result2 = vote2_top[0][0]
    if vote2_top[0][1] == 1:
        result2 = vote2[0]

    '''第三级预测按照决策树进行弃票处理，最终投票得到结果'''
    del_idx=[]
    votes3_copy=vote3.copy()
    for index in range(5):
        if vote3[index] not in class2_class3[result2]:
           del_idx.append(index)
    if len(del_idx)!=0:
        for idx in del_idx:
            vote3.remove(votes3_copy[idx])

    '''若全部弃票则从上一级附属级中随机选择可能正确的结果'''
    if len(vote3)==0:
        if result2 in class1_class2[result1]:
            random_index=random.randint(0,len(class2_class3[result2])-1)   # class1-class2从属关系正确，从class2_class3中随机取
            result3=class2_class3[result2][random_index]
        else:
            random_index=random.randint(0,len(class1_class3[result1])-1)   # class1-class2从属关系错误，从class1_class3中随机取
            result3=class1_class3[result1][random_index]
    else:
        vote3_counter = Counter(vote3)
        vote3_top = vote3_counter.most_common(1)
        if vote3_top[0][1] == 1:                                           # 如果弃票之后可能的结果都是一票则随机取
            ran_idx=random.randint(0,len(vote3_top)-1)
            result3=vote3_top[ran_idx][0]
        else:
            result3 = vote3_top[0][0]
    results1.append(result1)
    results2.append(result2)
    results3.append(result3)

with open('./ans/ans_ensemble.txt','w') as f:
    f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
    for index in range(len(item_id)):
        answer=item_id[index]+'\t'+results1[index]+'\t'+results2[index]+'\t'+results3[index]+'\n'
        f.write(answer)