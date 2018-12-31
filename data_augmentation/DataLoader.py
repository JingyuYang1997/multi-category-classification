import re

def MyDataSet_BalanceCategories(source_file):
    pattern = '(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
    with open(source_file, 'r') as f:
        data = f.read()
    data = re.findall(pattern, data)[1:]
    # random.shuffle(data)
    return data

