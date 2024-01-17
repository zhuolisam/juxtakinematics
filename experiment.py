import collections
from io import BytesIO, StringIO

import numpy as np

# num_frames=16

# hi = np.array([     1178.8,      1181.4 ,     1179.9    ,  1182.4  ,    1181.6    ,  1182.5     , 1182.3    ,  1184.7])
# print(len(hi))
# print(np.cumsum(hi))

# test_nan = np.array([np.nan, np.nan, 14])
# print(np.diff(test_nan))


# def find_non_null_indices(data):
#     # print('but first we check full length:', len(data))
#     # Find the first non-null data point from the beginning
#     start_index = np.where(~np.isnan(data))[0]
#     start_index = start_index[0] if start_index.size > 0 else None

#     # Find the first non-null data point from the end
#     end_index = np.where(~np.isnan(data))[0]
#     end_index = end_index[-1] if end_index.size > 0 else None
#     return start_index, end_index


# start_index, end_index = find_non_null_indices(test_nan)
# print(start_index, end_index)


# import numpy as np

# arr = np.array([1, 2, 3, 4, 5])
# arr_diff = np.diff(arr, prepend=arr[0])
# print(arr_diff)

# arr_diff_2 = np.diff(arr_diff, prepend=arr_diff[0])
# print(arr_diff_2)

# test_dict = {
#     "a": 1,
#     "b": 2,
#     "c": 3
# }


# class Test:
#     def __init__(self):
#         self.a = 1
#         self.b = 2
#         self.c = 3

#     # def __getattr__(self, __name: str):
#     #     return self.__name

#     def test(self):
#         return self.a

#     def check_attr(self, what):
#         return hasattr(self, what)


# testArr = np.array([np.nan, np.nan])
# print(np.isnan(testArr).all())

hello = "abc"
hello = collections.Counter(hello)
print(hello["t"])

hello2 = collections.defaultdict(int)
hello2["a"] == 0
print(hello2["b"])

hello3 = {
    "a": 1,
}

print(hello3.get("b", 0))
