import csv

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Проверка, чтобы игнорировать пустые строки
                data.append(row[0])  # Добавляем значение первой колонки в список
    return data

# Замените 'file.csv' на путь к вашему файлу CSV
csv_filename = 'folder_items_202306131027.csv'
csv_data = read_csv(csv_filename)

print(csv_data)