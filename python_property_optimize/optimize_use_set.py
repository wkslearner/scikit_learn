


# 去过普吉岛的人员数据
users_visited_phuket = [
    {"first_name": "Sirena", "last_name": "Gross", "phone_number": "650-568-0388", "date_visited": "2018-03-14"},
    {"first_name": "James", "last_name": "Ashcraft", "phone_number": "412-334-4380", "date_visited": "2014-09-16"},
]

# 去过新西兰的人员数据
users_visited_nz = [
    {"first_name": "Justin", "last_name": "Malcom", "phone_number": "267-282-1964", "date_visited": "2011-03-13"},
    {"first_name": "Albert", "last_name": "Potter", "phone_number": "702-249-3714", "date_visited": "2013-09-11"},
]



'''使用for循环 暴力查询  复杂度n*m'''
def find_potential_customers_v1():
    """找到去过普吉岛但是没去过新西兰的人
    """
    for phuket_record in users_visited_phuket:
        is_potential = True
        for nz_record in users_visited_nz:
            if phuket_record['first_name'] == nz_record['first_name'] and \
                    phuket_record['last_name'] == nz_record['last_name'] and \
                    phuket_record['phone_number'] == nz_record['phone_number']:
                is_potential = False
                break

        if is_potential:
            yield phuket_record


'''
基于集合的匹配模式 复杂度n+m
(原理：里的字典和集合对象都是基于 哈希表（Hash Table） 实现的。
判断一个东西是不是在集合里的平均时间复杂度是 O(1)，非常快 ）  
'''
def find_potential_customers_v2():
    """找到去过普吉岛但是没去过新西兰的人，性能改进版
    """
    # 首先，遍历所有新西兰访问记录，创建查找索引
    nz_records_idx = {
        (rec['first_name'], rec['last_name'], rec['phone_number'])
        for rec in users_visited_nz
    }

    for rec in users_visited_phuket:
        key = (rec['first_name'], rec['last_name'], rec['phone_number'])
        if key not in nz_records_idx:
            yield rec





