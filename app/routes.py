import sys
import os
from flask import request, jsonify, Response, send_from_directory
from app.services.student_management import StudentManagement
# from app.services.face_recognition import FaceTrackerRecognizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from recognize import FaceTrackerRecognizer


student_db_path = 'Students.csv'
student_management = StudentManagement(student_db_path)
face_tracker = FaceTrackerRecognizer(0)

def register_routes(app):
    @app.route("/")
    def serve_frontend():
        return send_from_directory(app.static_folder, "index.html")
    
    @app.errorhandler(404)
    def not_found(e):
        return send_from_directory(app.static_folder, "index.html")

    @app.route('/add_student', methods=['POST'])
    def add_student():
        data = request.form
        images = request.files.getlist('images')
        name = data.get('name')
        year = data.get('year')
        class_ = data.get('class')
        response = student_management.add_student(name, year, class_, images)
        return jsonify(response)

    @app.route('/display_students', methods=['GET'])
    def display_students():
        response = student_management.get_students()
        return jsonify(response)

    @app.route('/delete_student', methods=['POST'])
    def delete_student():
        student_id = request.form.get('id')
        response = student_management.delete_student(student_id)
        return jsonify(response)

    @app.route('/start', methods=['POST'])
    def start():
        camera_ip = request.form.get('camera_ip')
        if not camera_ip is None:
            face_tracker.camera_ip = camera_ip
        if not face_tracker.thread_track or not face_tracker.thread_track.is_alive():
            face_tracker.start()
        return jsonify({'success': True, 'message': 'بدء التصوير'})
    
    @app.route('/stop', methods=['POST'])
    def stop_tracking():
        global tracking_active
        if face_tracker.thread_track and face_tracker.thread_track.is_alive():
            face_tracker.stop()
            face_tracker.thread_track.join()
            return jsonify({'success': True, 'message': 'Tracking stopped successfully'})
        else:
            return jsonify({'success': False, 'message': 'Tracking is not active'})


    @app.route('/display_frames')
    def display_frames():
        return Response(face_tracker.get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')