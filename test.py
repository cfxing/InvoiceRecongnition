import re

num11 = re.compile(r'\d{12}')
num8 = re.compile(r'\d{8}')
next_pattern = re.compile(r'称[:]')
service_name = re.compile(r'[\*\+]\D[\*\+]]')
price = re.compile(r'[（]小写[）￥]')
next_address = re.compile(r'[地址、电话：]')

re_num = [num11,num8,service_name,price]
re_next = [next_pattern,next_address]

res = {}

glob_cc = None
rec = re.compile(r'\d+')
def get_num(rec,str):
    return rec.sub(rec,' ',str).strip()
def get_num11(txtPath):
    file = open(txtPath, 'r', encoding='utf-8')
    next_0_message = False
    next_1_message = False
    for line in file:
        # print(line)
        # pattern = re.compile(r'^称[$称:]')
        if next_0_message:
            if res.get('buyer') == None:
                res['buyer'] = line.strip()
            next_0_message = False
            continue
        if next_1_message :
            if res.get('sallers') == None:
                res['sallers'] = ''
            next_1_message = False
            continue

        for index,i in enumerate(re_next):
            cc = i.match(line)
            if cc != None:
                if index == 0:
                    next_0_message = True
                elif index == 1:
                    next_1_message = True
                continue


        for index,pattern in enumerate(re_num):
            cc = pattern.match(line)
            if cc != None:
                if index == 0:
                    if res.get('num12')==None:
                        res['num12'] = cc.string.strip()
                        continue

                if  index == 1:
                    if res.get('num8') == None:
                        res['num8'] = cc.string.strip()
                        continue

                if   index == 2:
                    if res.get('service_name')==None:
                        res['service_name'] = cc.string.strip()
                        continue
                if   index == 3:
                    if res.get('price') == None:
                        res['price'] = cc.string.strip()
                        continue
            # if cc is not None:
            #     print('cc', cc)
            #     print(line)
    print(res)






# file = open('./test_result/result/cropImgCC24.txt','r',encoding='utf-8')
# isMessage = False
# for line in file:
#     # print(line)
#     # pattern = re.compile(r'^称[$称:]')
#     pattern = re.compile('称')
#     cc = pattern.match(line)
#
#
#
#     if cc is not None:
#         print('cc',cc)
#         isMessage = True
#         continue
#     if isMessage:
#         print(line)
#         break



# file2 =  open('./test_result/result/cropImgCC25.txt','r',encoding='utf-8')
# for line in file2:
#     pattern = re.compile(r'[\+\*]')
#     cc = pattern.match(line)
#
#     if cc is not None:
#         print('cc', cc)
#         print(line)
#
# file3 =  open('./test_result/result/cropImgCC27.txt','r',encoding='utf-8')
# isMessage2 = False
# for line in file3:
#     pattern = re.compile(r'称[:]')
#     cc = pattern.match(line)
#
#     if cc is not None:
#         print('cc', cc)
#         isMessage2 = True
#         continue
#     if isMessage2:
#         print(line)
#         break

if __name__ == '__main__':
    get_num11('test_result/result/test.txt')