import csv
from app.config import Config

class StudentDB:
    def __init__(self, file_path):
        self.file_path = file_path

    def add_student(self, student_id, name, year, class_):
        with open(self.file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([student_id, name, year, class_])

    def get_students(self):
        students = []
        with open(self.file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                students.append({'id': row[0], 'name': row[1], 'year': row[2], 'class': row[3]})
        return students

    def delete_student(self, student_id):
        students = self.get_students()
        with open(self.file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'name', 'year', 'class'])
            for student in students:
                if student['id'] != student_id:
                    writer.writerow([student['id'], student['name'], student['year'], student['class']])