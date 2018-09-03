numbers = [1, 2, 3, 4, 5, 6]

print('Numbers before update')
print(numbers)

next_number = 7

numbers.append(next_number)

print('Numbers after update')
print(numbers)

other_numbers = [100, 200, 300]

collection_of_numbers = numbers + other_numbers

print("A concatenated collection of numbers looks like:")
print(collection_of_numbers)