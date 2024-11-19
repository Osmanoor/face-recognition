import os
import pandas as pd
from datetime import datetime, date
import threading

class AttendanceManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.datetoday = date.today().strftime("%m_%d_%y")
        self.attendance_file = f'Attendance/Attendance-{self.datetoday}.csv'
        self._initialize_attendance_file()

    def _initialize_attendance_file(self):
        if not os.path.isdir('Attendance'):
            os.makedirs('Attendance')
        if not os.path.isfile(self.attendance_file):
            with open(self.attendance_file, 'w') as f:
                f.write('Name,Enter_Time,Exit_Time')

    def _read_attendance_df(self):
        return pd.read_csv(self.attendance_file)

    def _write_attendance_df(self, df):
        df.to_csv(self.attendance_file, index=False)

    def add_entrance(self, name):
        with self.lock:
            current_time = datetime.now().strftime("%H:%M:%S")
            df = self._read_attendance_df()
            if name in df['Name'].values:
                # Check if enter_time is already set
                enter_time = df.loc[df['Name'] == name, 'Enter_Time'].values[0]
                if pd.isna(enter_time):
                    df.loc[df['Name'] == name, 'Enter_Time'] = current_time
            else:
                # Add new entry with enter_time
                new_entry = pd.DataFrame({'Name': [name], 'Enter_Time': [current_time], 'Exit_Time': [None]})
                df = pd.concat([df, new_entry], ignore_index=True)
            self._write_attendance_df(df)

    def add_exit(self, name):
        with self.lock:
            current_time = datetime.now().strftime("%H:%M:%S")
            df = self._read_attendance_df()
            if name in df['Name'].values:
                # Check if exit_time is already set
                exit_time = df.loc[df['Name'] == name, 'Exit_Time'].values[0]
                if pd.isna(exit_time):
                    df.loc[df['Name'] == name, 'Exit_Time'] = current_time
            else:
                # Add new entry with exit_time
                new_entry = pd.DataFrame({'Name': [name], 'Enter_Time': [None], 'Exit_Time': [current_time]})
                df = pd.concat([df, new_entry], ignore_index=True)
            self._write_attendance_df(df)

    def extract_attendance(self):
        with self.lock:
            df = self._read_attendance_df()
            names = df['Name']
            enter_times = df['Enter_Time']
            exit_times = df['Exit_Time']
            l = len(df)
            return names, enter_times, exit_times, l

# Example usage:
if __name__ == "__main__":
    manager = AttendanceManager()

    # Adding entrance attendance
    manager.add_entrance("John_123")

    # Adding exit attendance
    manager.add_exit("John_123")

    # Extracting attendance
    names, enter_times, exit_times, count = manager.extract_attendance()
    print("Names:", names)
    print("Enter Times:", enter_times)
    print("Exit Times:", exit_times)
    print("Count:", count)