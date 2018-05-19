import json
import numpy as np
class data:
    def __init__(self):
        self.data_list = []
        self.f = open('tr_data_20170128','r', encoding='utf8')
        for line in self.f:
            self.data_list.append(json.loads(line))
        self.tag_frequency = [] # 52 tags
        self.info_list = [] #{'user_id': 'user_gp_frequency'}, ordered
        self.result = [] #training data input, 52 tags array, ordered
        for i in self.data_list:
            temp = {}
            if 'user_gp_frequency' in i.keys() and 'article_l1_categories' in i.keys():
                temp['user_id'] = i['user_id']
                temp['user_gp_frequency'] = i['user_gp_frequency']
                for j in i['user_gp_frequency'].keys():
                    self.tag_frequency.append(j)
            if temp:
                self.info_list.append(temp)
        self.tag_frequency = list(set(self.tag_frequency))
        #------------enlarge the info_list------------------
        for item in self.tag_frequency:
            for box in self.info_list:
                if item not in box['user_gp_frequency'].keys():
                    box['user_gp_frequency'][item] = 0
        #----------clean repeated users-------------------
        self.info_list = self.clean_users()
        #----------create ordered input-------------------
        self.result = self.sorted_input(self.info_list)
        #----------write into file------------------------
        self.write_file()

    def sorted_input(self, list):
        result = []
        for dict in list:
            result.append([dict['user_gp_frequency'][item] for item in sorted(dict['user_gp_frequency'].keys())])
        return result

    def write_file(self):
        write = open('training_input', 'w')
        write.writelines(json.dumps(i) + '\n' for i in self.result)
        write.close()

    def clean_users(self):
        temp = [json.dumps(i) for i in self.info_list]
        temp_after = []
        for ele in temp:
            if ele not in temp_after:
                temp_after.append(ele)
        return [json.loads(j) for j in temp_after]

    def __del__(self):
        self.f.close()

class read:
    def __init__(self):
        self.reader = open('training_input','r')
        self.list = []
        for line in self.reader:
            self.list.append(json.loads(line))
        self.list = np.array(self.list)
        for i in self.list:
            print (i)
        print (len(self.list))

    def __del__(self):
        self.reader.close()

#ob = data()
ob_r = read()


