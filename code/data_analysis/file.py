import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
def test_function(x):
    y = (x-3)*(x-112)*(x-56)*(x-78)*(x-125)*(x-183)*(x-205)*(x-235)*(x-277)*(x-305)/100000000000000000
    return y
def get_function(minimum, maximum, step):
    x = np.arange(minimum, maximum+step, step)
    y = [test_function(i) for i in x]
    return x, y
def randomize_function(x, y):
    y = [i + np.random.randint(0, abs(i/8)+40) - i/6  for i in y]
    return x, y
x, y = get_function(0, 305, 1)
x, y = randomize_function(x, y)
plt.plot(x, y)
y = gaussian_filter(y, sigma=1)
plt.plot(x, y)
plt.show()
# smoothed_cases = []
# for y_val in y:
#     gaussian_kernel_values = np.exp(
#         -(((y - y_val).apply(lambda x: x)) ** 2) / (2 * (2 ** 2))
#     )
#     gaussian_kernel_values /= gaussian_kernel_values.sum()
#     smoothed_cases.append(round(npl['new_cases'] * gaussian_kernel_values).sum())
# npl['smoothed_new_cases'] = smoothed_cases
# get_function(0, 305, 0.01)
# smoothed_cases = []
# for date in sorted(npl['date']):
#     npl['gkv'] = np.exp(
#         -(((npl['date'] - date).apply(lambda x: x.days)) ** 2) / (2 * (2 ** 2))
#     )
#     npl['gkv'] /= npl['gkv'].sum()
#     smoothed_cases.append(round(npl['new_cases'] * npl['gkv']).sum())

# npl['smoothed_new_cases'] = smoothed_cases