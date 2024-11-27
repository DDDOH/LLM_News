import numpy

val_ls = [82.43, 81.67, 82.09]

mean = numpy.mean(val_ls)
std = numpy.std(val_ls)
ub = mean + 1.96 * std / numpy.sqrt(len(val_ls))
lb = mean - 1.96 * std / numpy.sqrt(len(val_ls))

print(f'[{lb:.4f}%, {ub:.4f}%]')
