

def earning_caculate(in_point,out_point,num_point):
    earn_value=(out_point-in_point)*num_point

    return earn_value

def extreme_caculate(in_point,max_limit,min_limit,num_point):
    get_risk_rate=(max_limit-in_point)/(in_point-min_limit)
    abs_earning=get_risk_rate*num_point

    return get_risk_rate,abs_earning


def except_caculate(in_point,prob_list,except_value):
    diff_list=[]
    for prob_value in prob_list:
        diff=prob_value-in_point
        diff_list.append(diff)
    
    return diff_list


if __name__=='__main__':
    pass







