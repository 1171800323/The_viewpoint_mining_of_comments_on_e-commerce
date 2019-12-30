import model
import parse_data
import numpy as np
import pre_process
import re
import result_element

model, (vocab, chunk_tags) = model.create_model(train=False)
model.load_weights('model/crf.h5')
result_all = []


def get_result(predict_text):
    ele = result_element.Element()
    string, length = parse_data.process_data(predict_text, vocab)
    raw = model.predict(string)[0][-length:]
    result = [np.argmax(row) for row in raw]
    print(result)
    ele.result_list = result
    result_tags = [chunk_tags[i] for i in result]
    asp, opi = '', ''
    flag = -1
    for idx, ch in enumerate(predict_text):
        tag = result_tags[idx]
        tag_next = 'O'
        if idx < len(predict_text) - 1:
            tag_next = result_tags[idx + 1]
        if tag in ('B-ASP', 'I-ASP'):
            asp += ch
            if tag_next != 'I-ASP':
                print('asp:', asp)
                ele.asp.append(asp)
                asp = ''
        if idx <= flag:
            pass
        else:
            tag = result[idx]
            if tag == 3:
                opi += ch
                while idx + 1 < len(predict_text):
                    if result[idx + 1] == 4:
                        idx += 1
                        opi += predict_text[idx]
                    else:
                        break
                print('opi:', opi)
                ele.opi.append(opi)
                opi = ''
                flag = idx
            if tag == 4:
                if idx > 0:
                    if predict_text[idx - 1] == '很':
                        opi += '很'
                        opi += predict_text[idx]
                    while idx + 1 < len(predict_text):
                        if result[idx + 1] == 4:
                            idx += 1
                            opi += predict_text[idx]
                        else:
                            break
                    print('opi:', opi)
                    ele.opi.append(opi)
                    opi = ''
                    flag = idx
                if idx == 0:
                    opi += predict_text[idx]
                    while idx + 1 < len(predict_text):

                        if result[idx + 1] == 4:
                            idx += 1
                            opi += predict_text[idx]
                        else:
                            break
                    print('opi:', opi)
                    ele.opi.append(opi)
                    opi = ''
                    flag = idx
        # if tag in ('B-OPI', 'I-OPI'):
        #     opi += ch
        #     if tag_next != 'I-OPI':
        #         length_opi = len(opi)
        #         single = ''
        #         start = -1
        #         if idx > length_opi:
        #             start = idx - length_opi - 1
        #             if idx == len(predict_text) - 1:
        #                 start += 1
        #             single = predict_text[start]
        #         elif idx > length_opi - 1:
        #             start = idx - length_opi
        #             if idx == len(predict_text) - 1:
        #                 start += 1
        #             single = predict_text[start]
        #         flag = result_tags[idx - length_opi + 1][0] == 'I'
        #         if flag and start != -1 and parse_data.search_begin(single):
        #             # print('opi:', predict_text[start:idx + 1])
        #             ele.opi.append(str(predict_text[start:idx + 1]))
        #         else:
        #             # print('opi:', opi)
        #             ele.opi.append(opi)
        #         opi = ''
    # result_all.append(ele)
    return ele


def filter_n(x):
    if isinstance(x, str):
        x = re.sub('\n', '', str(x))
    return x


def get_all():
    result_all.clear()
    df_review = pre_process.read_review("Test")
    # df_review.sort_values(by="id", ascending=True)
    # df_review.to_csv("test2.csv", sep=",", header=True, index=False)
    df_review.applymap(filter_n)
    print(df_review)
    ans_list = []
    for index, row in df_review.iterrows():
        tmp = re.split('\n', str(row))
        print(index + 1)
        # print(str(row))
        # print(tmp[1])
        text_raw = re.sub(r"Review\s+", '', tmp[1])
        if re.search(r'\s+', text_raw):
            text_tmp = re.split(r"\s+", text_raw)
            for i in text_tmp:
                ele_get_all = get_result(i)
                ans_list.extend(match(index, ele_get_all, i))
        else:
        # print(text_raw)
            ele_get_all = get_result(text_raw)
            ans_list.extend(match(index, ele_get_all, text_raw))
    file1 = open('Test/task1_answer.csv', 'w', encoding='utf-8')
    for i in ans_list:
        file1.write(i + '\n')
    file1.close()
    print(ans_list)


def match(index_match: int, ele_match: result_element.Element, text_match):
    if index_match == 32:
        print('caution')
        pass
    tmp_match = re.split(r"[，。！？?!,.\s+]", text_match)
    ans = []
    counter_o = []
    counter_a = []
    tmp_o = []
    tmp_a = []
    prefix = []
    for i in tmp_match:

        if i != '':
            counter_o.clear()
            counter_a.clear()
            tmp_o.clear()
            tmp_a.clear()
            # counter_o = []
            for j in ele_match.opi:
                if str(i).find(j) != -1:
                    counter_o.append(str(i).find(j))
                    tmp_o.append(j)
            # counter_a = []
            for k in ele_match.asp:
                if str(i).find(k) != -1:
                    counter_a.append(str(i).find(k))
                    tmp_a.append(k)
        # if len(counter_o) == 2:
        #     if tmp_o[0].find(tmp_o[1]) != -1:
        #         counter_o.pop(1)
        #         tmp_o.pop(1)
        #     elif tmp_o[1].find(tmp_o[0]) != -1:
        #         counter_o.pop(0)
        #         tmp_o.pop(0)
        if len(counter_a) >= 2:                      #这里为了解决前缀导致错误匹配的问题
            prefix.clear()
            flag = 0
            for j in range(len(tmp_a)):
                for k in range(len(tmp_a)):
                    if tmp_a[j].find(tmp_a[k]) != -1 and j != k:
                        if tmp_a[k].find(tmp_a[j]) == -1:
                            prefix.append(k)
                        else:
                            flag = 1
                            break
            if flag == 1:
                prefix.clear()
                # tmp_a = list(set(tmp_a))
                # counter_a = list(set(counter_a))
                prefix.clear()
                # tmp_o = list(set(tmp_o))
                tmp_set = set()
                tmp_set.clear()
                counter_pop = 0
                for x in range(len(tmp_a)):
                    if tmp_a[x] not in tmp_set:
                        tmp_set.add(tmp_a[x])
                    else:
                        counter_a.pop(x - counter_pop)
                        counter_pop += 1
                tmp_a = list(tmp_set)

                for j in range(len(tmp_a)):
                    for k in range(len(tmp_a)):
                        if tmp_a[j].find(tmp_a[k]) != -1 and j != k:
                            if tmp_a[k].find(tmp_a[j]) == -1:
                                prefix.append(k)
                            else:
                                flag = 1

            prefix.sort()
            count_filter = 0
            for p in prefix:  #
                counter_a.pop(p - count_filter)
                tmp_a.pop(p - count_filter)
                count_filter += 1

        if len(counter_o) >= 2:
            prefix.clear()
            flag = 0
            for j in range(len(tmp_o)):
                for k in range(len(tmp_o)):
                    if tmp_o[j].find(tmp_o[k]) != -1 and j != k:
                        if tmp_o[k].find(tmp_o[j]) == -1:
                            prefix.append(k)
                        else:
                            flag = 1
                            break
            if flag == 1:
                prefix.clear()
                # tmp_o = list(set(tmp_o))
                tmp_set = set()
                tmp_set.clear()
                counter_pop = 0
                for x in range(len(tmp_o)):
                    if tmp_o[x] not in tmp_set:
                        tmp_set.add(tmp_o[x])
                    else:
                        counter_o.pop(x - counter_pop)
                        counter_pop += 1
                tmp_o = list(tmp_set)
                # counter_o = list(set(counter_o))
                for j in range(len(tmp_o)):
                    for k in range(len(tmp_o)):
                        if tmp_o[j].find(tmp_o[k]) != -1 and j != k:
                            if tmp_o[k].find(tmp_o[j]) == -1:
                                prefix.append(k)
                            else:
                                flag = 1

            prefix.sort()
            count_filter = 0
            for p in prefix:                                 #
                counter_o.pop(p - count_filter)
                tmp_o.pop(p - count_filter)
                count_filter += 1

        if len(counter_o) == 0 and len(counter_a) == 0:
            pass   #没有匹配到，直接pass
        elif len(counter_o) != 0 and len(counter_a) == 0:   #有opinion但是没有asp
            for j in tmp_o:
                ans.append(str(index_match + 1) + ',_,' + j)
            # return ans
        elif len(counter_o) == len(counter_a):           #有多对asp和opinion，则一一对应
            for k in range(len(counter_o)):
                ans.append(str(index_match + 1) + ',' + tmp_a[k] + ',' + tmp_o[k])
        elif len(counter_o) == 0 and len(counter_a) != 0:     #有asp没有opinion
            ans.append(str(index_match + 1) + ',' + tmp_a[0] + ',_')
        elif len(counter_o) == 1 and len(counter_a) > 1:      #一个opinion但是有多个asp
            for k in tmp_a:
                ans.append(str(index_match + 1) + ',' + k + ',' + tmp_o[0])
        elif len(counter_o) > len(counter_a) > 0:
            #多个opinion和多个asp同时出现的时候，我们就把距离asp最近的opi作为这个asp的opi，剩余的opi直接输出
            printed_set = set()
            printed_set.clear()
            # 继续这里，一个asp和多个opinion同时出现
            for x in range(len(counter_a)):
                tmp_close = -1
                for k in counter_o:
                    if abs(k - counter_a[x]) <= tmp_close:
                        tmp_close = k
                for m in tmp_o:
                    if str(i).find(m) == tmp_close:
                        ans.append((str(index_match + 1) + ',' + tmp_a[0] + ',' + m))
                        printed_set.update(m)

            for m in tmp_o:
                if m not in printed_set:
                    ans.append((str(index_match + 1) + ',_,' + m))
            printed_set.clear()


        else:
            ans.append(str(index_match + 1) + 'caution_combin')
    return list(set(ans))


# text = '很好，超值，很好用'
# get_result(text)

get_all()

