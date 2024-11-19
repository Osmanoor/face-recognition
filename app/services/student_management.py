import os
import sys
import uuid
from app.models import StudentDB
from app.utils.file_operations import save_image
from app.utils.error_handling import handle_error
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import add_persons



class StudentManagement:
    def __init__(self, student_db_path):
        self.student_db = StudentDB(student_db_path)
        self.add_person = add_persons
        
    def add_student(self, name, year, class_, images):
        try:
            student_id = str(uuid.uuid4())
            image_paths = []
            for i, image in enumerate(images):
                image_path = save_image(image, student_id, i)
                image_paths.append(image_path)
            self.add_person.add()
            self.student_db.add_student(student_id, name, year, class_)
            return {'success': True, 'message': 'الطالب تم إضافته بنجاح'}
        except Exception as e:
            handle_error(e)
            return {'success': False, 'message': 'حدث خطأ أثناء إضافة الطالب'}

    def get_students(self):
        try:
            students = self.student_db.get_students()
            return {'success': True, 'students': students}
        except Exception as e:
            handle_error(e)
            return {'success': False, 'message': 'حدث خطأ أثناء جلب قائمة الطلاب'}

    def delete_student(self, student_id):
        try:
            self.add_person.delete(student_id)
            self.student_db.delete_student(student_id)
            return {'success': True, 'message': 'الطالب تم حذفه بنجاح'}
        except Exception as e:
            handle_error(e)
            return {'success': False, 'message': 'حدث خطأ أثناء حذف الطالب'}