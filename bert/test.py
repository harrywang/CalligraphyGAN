from bert_serving.client import BertClient



def main():
    prefix_q = '##### **Q:** '
    with open('./client/README.md') as fp:
    questions = [v.replace(prefix_q, '').strip() for v in fp if v.strip() and v.startswith(prefix_q)]
    print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))



    #bc = BertClient()
    #v = bc.encode(['红烧肉卤鸡蛋', '酱香排骨', '面筋塞肉','好','鼎'])
    #print(v)
    #print(v.shape)  #


if __name__ == '__main__':
    main()
