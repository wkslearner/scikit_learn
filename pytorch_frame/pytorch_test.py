
import torch

letter='wang'
n_letters=3
tensor = torch.zeros(4,2,n_letters)

# print(tensor)

a='sss'


def convert_float(strs):
    try:
        strs = float(strs)
    except:
        strs = strs

    return strs


print(convert_float(a))









