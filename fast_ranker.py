def calculate_percentile_and_average(filename, target_number):
    with open(filename, 'r') as file:
        numbers = [int(line.strip()) for line in file]

    numbers.sort()

    try:
        index = numbers.index(target_number)
        percentile = (index + 1) / len(numbers) * 100
        smaller_percentage = index / len(numbers) * 100
        average = sum(numbers) / len(numbers)

        print(f"The percentile of {target_number} is: {percentile:.2f}%")
        print(f"{smaller_percentage:.2f}% of numbers in the file are smaller than {target_number}.")
        print(f"The average of all numbers in the file is: {average:.2f}")
    except ValueError:
        print(f"{target_number} is not in the list.")

# 用法示例

def main(x):
    filename = 'random_baseline.log'  # 替换为实际的文件名
    target_number = x
    calculate_percentile_and_average(filename, target_number)
