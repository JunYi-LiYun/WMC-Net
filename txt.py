# -*-coding:utf-8-*-


import os
import os.path


def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def get_files_list(dir):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    files_list = [] #写入文件的数据
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            print("parent is: " + parent)
            print("filename is: " + filename)
            print(os.path.join(parent, filename).replace('\\','/'))  # 输出rootdir路径下所有文件（包含子文件）信息
            curr_file = parent.split(os.sep)[-1]	#获取正在遍历的文件夹名（也就是类名）
            curr_filea= curr_file.split('.')[0]
            #根据class名确定labels
            if curr_filea == "001":
                labels = 0
            elif curr_filea == "002":
                labels = 1
            elif curr_filea == "003":
                labels = 2
            elif curr_filea == "004":
                labels = 3
            elif curr_filea == "005":
                labels = 4
            elif curr_filea == "006":
                labels = 5
            elif curr_filea == "007":
                labels = 6
            elif curr_filea == "008":
                labels = 7
            elif curr_filea == "009":
                labels = 8
            elif curr_filea == "010":
                labels = 9
            elif curr_filea == "011":
                labels = 10
            elif curr_filea == "012":
                labels = 11
            elif curr_filea == "013":
                labels = 12
            elif curr_filea == "014":
                labels = 13
            elif curr_filea == "015":
                labels = 14
            elif curr_filea == "016":
                labels = 15
            elif curr_filea == "017":
                labels = 16
            elif curr_filea == "018":
                labels = 17
            elif curr_filea == "019":
                labels = 18
            elif curr_filea == "020":
                labels = 19
            elif curr_filea == "021":
                labels = 20
            elif curr_filea == "022":
                labels = 21
            elif curr_filea == "023":
                labels = 22
            elif curr_filea == "024":
                labels = 23
            elif curr_filea == "025":
                labels = 24
            elif curr_filea == "026":
                labels = 25
            elif curr_filea == "027":
                labels = 26
            elif curr_filea == "028":
                labels = 27
            elif curr_filea == "029":
                labels = 28
            elif curr_filea == "030":
                labels = 29
            elif curr_filea == "031":
                labels = 30
            elif curr_filea == "032":
                labels = 31
            elif curr_filea == "033":
                labels = 32
            elif curr_filea == "034":
                labels = 33
            elif curr_filea == "035":
                labels = 34
            elif curr_filea == "036":
                labels = 35
            elif curr_filea == "037":
                labels = 36
            elif curr_filea == "038":
                labels = 37
            elif curr_filea == "039":
                labels = 38
            elif curr_filea == "040":
                labels = 39
            elif curr_filea == "041":
                labels = 40
            elif curr_filea == "042":
                labels = 41
            elif curr_filea == "043":
                labels = 42
            elif curr_filea == "044":
                labels = 43
            elif curr_filea == "045":
                labels = 44
            elif curr_filea == "046":
                labels = 45
            elif curr_filea == "047":
                labels = 46
            elif curr_filea == "048":
                labels = 47
            elif curr_filea == "049":
                labels = 48
            elif curr_filea == "050":
                labels = 49
            elif curr_filea == "051":
                labels = 50
            elif curr_filea == "052":
                labels = 51
            elif curr_filea == "053":
                labels = 52
            elif curr_filea == "054":
                labels = 53
            elif curr_filea == "055":
                labels = 54
            elif curr_filea == "056":
                labels = 55
            elif curr_filea == "057":
                labels = 56
            elif curr_filea == "058":
                labels = 57
            elif curr_filea == "059":
                labels = 58
            elif curr_filea == "060":
                labels = 59
            elif curr_filea == "061":
                labels = 60
            elif curr_filea == "062":
                labels = 61
            elif curr_filea == "063":
                labels = 62
            elif curr_filea == "064":
                labels = 63
            elif curr_filea == "065":
                labels = 64
            elif curr_filea == "066":
                labels = 65
            elif curr_filea == "067":
                labels = 66
            elif curr_filea == "068":
                labels = 67
            elif curr_filea == "069":
                labels = 68
            elif curr_filea == "070":
                labels = 69
            elif curr_filea == "071":
                labels = 70
            elif curr_filea == "072":
                labels = 71
            elif curr_filea == "073":
                labels = 72
            elif curr_filea == "074":
                labels = 73
            elif curr_filea == "075":
                labels = 74
            elif curr_filea == "076":
                labels = 75
            elif curr_filea == "077":
                labels = 76
            elif curr_filea == "078":
                labels = 77
            elif curr_filea == "079":
                labels = 78
            elif curr_filea == "080":
                labels = 79
            elif curr_filea == "081":
                labels = 80
            elif curr_filea == "082":
                labels = 81
            elif curr_filea == "083":
                labels = 82
            elif curr_filea == "084":
                labels = 83
            elif curr_filea == "085":
                labels = 84
            elif curr_filea == "086":
                labels = 85
            elif curr_filea == "087":
                labels = 86
            elif curr_filea == "088":
                labels = 87
            elif curr_filea == "089":
                labels = 88
            elif curr_filea == "090":
                labels = 89
            elif curr_filea == "091":
                labels = 90
            elif curr_filea == "092":
                labels = 91
            elif curr_filea == "093":
                labels = 92
            elif curr_filea == "094":
                labels = 93
            elif curr_filea == "095":
                labels = 94
            elif curr_filea == "096":
                labels = 95
            elif curr_filea == "097":
                labels = 96
            elif curr_filea == "098":
                labels = 97
            elif curr_filea == "099":
                labels = 98
            elif curr_filea == "100":
                labels = 99
            elif curr_filea == "101":
                labels = 10
            elif curr_filea == "102":
                labels = 101
            elif curr_filea == "103":
                labels = 102
            elif curr_filea == "104":
                labels = 103
            elif curr_filea == "105":
                labels = 104
            elif curr_filea == "106":
                labels = 105
            elif curr_filea == "107":
                labels = 106
            elif curr_filea == "108":
                labels = 107
            elif curr_filea == "109":
                labels = 108
            elif curr_filea == "110":
                labels = 109
            elif curr_filea == "111":
                labels = 110
            elif curr_filea == "112":
                labels = 111
            elif curr_filea == "113":
                labels = 112
            elif curr_filea == "114":
                labels = 113
            elif curr_filea == "115":
                labels = 114
            elif curr_filea == "116":
                labels = 115
            elif curr_filea == "117":
                labels = 116
            elif curr_filea == "118":
                labels = 117
            elif curr_filea == "119":
                labels = 118
            elif curr_filea == "120":
                labels = 119
            elif curr_filea == "121":
                labels = 120
            elif curr_filea == "122":
                labels = 121
            elif curr_filea == "123":
                labels = 122
            elif curr_filea == "124":
                labels = 123
            elif curr_filea == "125":
                labels = 124
            elif curr_filea == "126":
                labels = 125
            elif curr_filea == "127":
                labels = 126
            elif curr_filea == "128":
                labels = 127
            elif curr_filea == "129":
                labels = 128
            elif curr_filea == "130":
                labels = 129
            elif curr_filea == "131":
                labels = 130
            elif curr_filea == "132":
                labels = 131
            elif curr_filea == "133":
                labels = 132
            elif curr_filea == "134":
                labels = 133
            elif curr_filea == "135":
                labels = 134
            elif curr_filea == "136":
                labels = 135
            elif curr_filea == "137":
                labels = 136
            elif curr_filea == "138":
                labels = 137
            elif curr_filea == "139":
                labels = 138
            elif curr_filea == "140":
                labels = 139
            elif curr_filea == "141":
                labels = 140
            elif curr_filea == "142":
                labels = 141
            elif curr_filea == "143":
                labels = 142
            elif curr_filea == "144":
                labels = 143
            elif curr_filea == "145":
                labels = 144
            elif curr_filea == "146":
                labels = 145
            elif curr_filea == "147":
                labels = 146
            elif curr_filea == "148":
                labels = 147
            elif curr_filea == "149":
                labels = 148
            elif curr_filea == "150":
                labels = 149
            elif curr_filea == "151":
                labels = 150
            elif curr_filea == "152":
                labels = 151
            elif curr_filea == "153":
                labels = 152
            elif curr_filea == "154":
                labels = 153
            elif curr_filea == "155":
                labels = 154
            elif curr_filea == "156":
                labels = 155
            elif curr_filea == "157":
                labels = 156
            elif curr_filea == "158":
                labels = 157
            elif curr_filea == "159":
                labels = 158
            elif curr_filea == "160":
                labels = 159
            elif curr_filea == "161":
                labels = 160
            elif curr_filea == "162":
                labels = 161
            elif curr_filea == "163":
                labels = 162
            elif curr_filea == "164":
                labels = 163
            elif curr_filea == "165":
                labels = 164
            elif curr_filea == "166":
                labels = 165
            elif curr_filea == "167":
                labels = 166
            elif curr_filea == "168":
                labels = 167
            elif curr_filea == "169":
                labels = 168
            elif curr_filea == "170":
                labels = 169
            elif curr_filea == "171":
                labels = 170
            elif curr_filea == "172":
                labels = 171
            elif curr_filea == "173":
                labels = 172
            elif curr_filea == "174":
                labels = 173
            elif curr_filea == "175":
                labels = 174
            elif curr_filea == "176":
                labels = 175
            elif curr_filea == "177":
                labels = 176
            elif curr_filea == "178":
                labels = 177
            elif curr_filea == "179":
                labels = 178
            elif curr_filea == "180":
                labels = 179
            elif curr_filea == "181":
                labels = 180
            elif curr_filea == "182":
                labels = 181
            elif curr_filea == "183":
                labels = 182
            elif curr_filea == "184":
                labels = 183
            elif curr_filea == "185":
                labels = 184
            elif curr_filea == "186":
                labels = 185
            elif curr_filea == "187":
                labels = 186
            elif curr_filea == "188":
                labels = 187
            elif curr_filea == "189":
                labels = 188
            elif curr_filea == "190":
                labels = 189
            elif curr_filea == "191":
                labels = 190
            elif curr_filea == "192":
                labels = 191
            elif curr_filea == "193":
                labels = 192
            elif curr_filea == "194":
                labels = 193
            elif curr_filea == "195":
                labels = 194
            elif curr_filea == "196":
                labels = 195
            elif curr_filea == "197":
                labels = 196
            elif curr_filea == "198":
                labels = 197
            elif curr_filea == "199":
                labels = 198
            elif curr_filea == "200":
                labels = 199



            dir_path = parent.replace('\\', '/').split('/')[-2]  # train?val?test?
            curr_file = os.path.join(dir_path, curr_file)  # 相对路径

            files_list.append([os.path.join(curr_file, filename).replace('\\', '/') ,',',labels])  # 相对路径+label

        # 写入csv文件
            path = "%s" % os.path.join(curr_file, filename).replace('\\', '/')
            label = "%d" % labels
            list = [path, label]
            data = pd.DataFrame([list])
            if dir == r'J:\ljy\flower\train':
              data.to_csv(r"J:\ljy\train.csv", mode='a', header=False, index=False)
            elif dir == r'J:\ljy\flower\test':
              data.to_csv(r'J:\ljy\test.csv', mode='a', header=False, index=False)

    return files_list




if __name__ == '__main__':

    import pandas as pd
    # 先生成两个csv文件夹
    df = pd.DataFrame(columns=['img_path', 'target'])
    df.to_csv(r"/media/y1408/H/ljy/CUB/train.csv", index=False)

    df2 = pd.DataFrame(columns=['img_path', 'target'])
    df2.to_csv(r'/media/y1408/H/ljy/CUB/test.csv', index=False)

    #写入txt文件
    train_dir = r'/media/y1408/H/ljy/CUB/train'
    train_txt = r'/media/y1408/H/ljy/CUB/train.csv'
    train_data = get_files_list(train_dir)
    write_txt(train_data, train_txt, mode='w')

    val_dir = r'/media/y1408/H/ljy/CUB/test'
    val_txt = r'/media/y1408/H/ljy/CUB/test.csv'
    val_data = get_files_list(val_dir)
    write_txt(val_data, val_txt, mode='w')


