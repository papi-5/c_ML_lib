import csv

label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

with open('Iris.csv', 'r') as f_in, open('iris_clean.csv', 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    next(reader)  # skip header row
    for row in reader:
        row = row[1:]  # remove first column (id)
        row[-1] = label_map[row[-1]]  # map label string to number
        writer.writerow(row)