def sum_of_all_elements(array):
    sum = 0
    for i in array:
        sum = sum + i
    return sum

def mean_value_of_all_elements(array):
    sum = sum_of_all_elements(array)
    count = len(array)

    mean = sum/count
    return mean

array = [1, 3, 1, 3]

sum = sum_of_all_elements(array)
mean = mean_value_of_all_elements(array)

print('Sum of all elements equals {0}'.format(sum))
print('Mean of all elements equals {0}'.format(mean))