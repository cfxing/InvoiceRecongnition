import re

def get_rec(txtPath):
    file = open(txtPath,'r',encoding='utf-8')
    res = {'发票编号':None,'税号':None,'购买方':None,'销售方':None,'价格':None,'服务名称':None}
    for line in file:
        # print(line)
        words = line.strip().split('\t')
        print(words)
        for i,word in enumerate(words):
            num12 = re.match('\d{12}',word)
            num8 = re.match('\d{8}',word)
            buyer = re.match('[称:]',word)
            price = re.match(r'[（]小写[）￥]',word)
            service_name = re.match(r'^[*+]\D+[*+]\D',word)
            # print(cc)
            # print(cc2)
            # print(buyer)
            # print(price)
            # print(service_name)
            if num12 != None:
                if res['发票编号'] == None:
                    res['发票编号'] = num12.string
            elif num8 != None:
                if res['税号'] == None:
                    res['税号'] = num8.string
            elif buyer != None:
                if res['购买方'] == None:
                    res['购买方'] = words[i+1]
                    continue
                if res['销售方'] == None:
                    res['销售方'] = words[i+1]
            elif price != None:
                if res['价格'] == None:
                    prices = re.findall(r'\d+', price.string)
                    pri = prices[0] + '.' + prices[1]
                    res['价格'] = pri
            elif service_name != None:
                if res['服务名称'] == None:
                    res['服务名称'] = service_name.string
    print(res)



# if __name__ == '__main__':
#     get_rec('./test_result/result/3test.txt')
