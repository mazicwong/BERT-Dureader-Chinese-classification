import json

def cut(path, save_path):
    fw = open(save_path, 'w', encoding='utf-8')
    fw.write('total_id' + '\t' + 'text' + '\t' + 'label' + '\t' + 'fact_or_opinion' + '\n')
    total_id = 0
    label2idx = {"Yes":'0', "No":'1', "Depends":'2', "No_Opinion":'3'}
    cntLabel = [0]*4
    for line in open(path, 'r', encoding='utf-8').readlines():
        line = json.loads(line)
        answers = line['answers']
        labels = line['yesno_answers']
        fact_or_opinion = line["fact_or_opinion"]
        if len(answers)==0 or len(labels)==0 or len(fact_or_opinion)==0 or len(answers)!=len(labels):
            continue
            raise KeyError("lengths problem")
        for answer, label in zip(answers, labels):
            answer = ''.join(answer.split())
            fw.write(str(total_id) + '\t' + answer + '\t' + label2idx[label] + '\t' + fact_or_opinion + '\n')
            cntLabel[int(label2idx[label])] += 1
            total_id += 1
    print(cntLabel)   # train = [8486, 5222, 3149, 203]    |||   dev = [242, 208, 0, 0]
    fw.close()

if __name__ == '__main__':
    cut(path = './train.json', save_path = './train.tsv')
    cut(path = './dev.json', save_path = './dev.tsv')
