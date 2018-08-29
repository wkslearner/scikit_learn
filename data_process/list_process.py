#!/usr/bin/python
# encoding=utf-8



'''移除列表中部分元素，生成新列表'''

def remove_list(all_list,remove_element):
    end_list = []
    for element in all_list:
        if element in remove_element:
            continue
        else:
            end_list.append(element)

    return end_list

