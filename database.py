import sqlite3
import os
from datetime import datetime

class LicensePlateDB:
    def __init__(self, db_path='license_plates.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create registered vehicles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registered_vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT UNIQUE NOT NULL,
                owner_name TEXT,
                vehicle_type TEXT,
                status TEXT DEFAULT 'active',
                registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # Create detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT NOT NULL,
                detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                video_name TEXT,
                frame_number INTEGER,
                confidence_score REAL,
                is_registered BOOLEAN,
                match_type TEXT
            )
        ''')
        
        # Create alerts table for flagged vehicles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT NOT NULL,
                alert_type TEXT,
                alert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_registered_vehicle(self, license_plate, owner_name=None, 
                              vehicle_type=None, status='active', notes=None):
        """Add a vehicle to the registered vehicles database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO registered_vehicles 
                (license_plate, owner_name, vehicle_type, status, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (license_plate.upper(), owner_name, vehicle_type, status, notes))
            
            conn.commit()
            return True, "Vehicle registered successfully"
        except sqlite3.IntegrityError:
            return False, "License plate already exists"
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            conn.close()
    
    def add_multiple_vehicles(self, vehicles_data):
        """Add multiple vehicles at once
        vehicles_data: list of tuples (license_plate, owner_name, vehicle_type, status, notes)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        added = 0
        skipped = 0
        
        for vehicle in vehicles_data:
            try:
                cursor.execute('''
                    INSERT INTO registered_vehicles 
                    (license_plate, owner_name, vehicle_type, status, notes)
                    VALUES (?, ?, ?, ?, ?)
                ''', vehicle)
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1
                continue
        
        conn.commit()
        conn.close()
        
        return added, skipped
    
    def check_license_plate(self, license_plate):
        """Check if a license plate is in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM registered_vehicles 
            WHERE license_plate = ?
        ''', (license_plate.upper(),))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'found': True,
                'id': result[0],
                'license_plate': result[1],
                'owner_name': result[2],
                'vehicle_type': result[3],
                'status': result[4],
                'registered_date': result[5],
                'notes': result[6]
            }
        else:
            return {'found': False}
    
    def fuzzy_match(self, detected_plate, threshold=0.8):
        """Find similar license plates using fuzzy matching"""
        from difflib import SequenceMatcher
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT license_plate FROM registered_vehicles')
        all_plates = cursor.fetchall()
        conn.close()
        
        matches = []
        detected_plate = detected_plate.upper()
        
        for (plate,) in all_plates:
            similarity = SequenceMatcher(None, detected_plate, plate).ratio()
            if similarity >= threshold:
                matches.append({
                    'plate': plate,
                    'similarity': similarity,
                    'match_type': 'exact' if similarity == 1.0 else 'fuzzy'
                })
        
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
    def log_detection(self, license_plate, video_name=None, frame_number=None, 
                     confidence_score=None, is_registered=False, match_type='exact'):
        """Log a detection event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections 
            (license_plate, video_name, frame_number, confidence_score, 
             is_registered, match_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (license_plate.upper(), video_name, frame_number, 
              confidence_score, is_registered, match_type))
        
        conn.commit()
        conn.close()
    
    def create_alert(self, license_plate, alert_type='unauthorized', description=None):
        """Create an alert for a detected vehicle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (license_plate, alert_type, description)
            VALUES (?, ?, ?)
        ''', (license_plate.upper(), alert_type, description))
        
        conn.commit()
        conn.close()
    
    def get_all_registered_vehicles(self):
        """Get all registered vehicles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, license_plate, owner_name, vehicle_type, status, 
                   registered_date, notes
            FROM registered_vehicles
            ORDER BY registered_date DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_detection_history(self, limit=100):
        """Get recent detection history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, license_plate, detection_time, video_name, 
                   frame_number, confidence_score, is_registered, match_type
            FROM detections
            ORDER BY detection_time DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_active_alerts(self):
        """Get all unresolved alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, license_plate, alert_type, alert_time, description
            FROM alerts
            WHERE resolved = 0
            ORDER BY alert_time DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def delete_vehicle(self, license_plate):
        """Remove a vehicle from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM registered_vehicles 
            WHERE license_plate = ?
        ''', (license_plate.upper(),))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted > 0
    
    def update_vehicle_status(self, license_plate, new_status):
        """Update vehicle status (active, suspended, stolen, etc.)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE registered_vehicles 
            SET status = ?
            WHERE license_plate = ?
        ''', (new_status, license_plate.upper()))
        
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        
        return updated > 0


def initialize_sample_data():
    """Initialize database with sample vehicles"""
    db = LicensePlateDB()
    
    sample_vehicles = [
        ('MV51VSU', 'John Smith', 'Car', 'active', 'Regular visitor'),
        ('OA13NRU', 'Sarah Johnson', 'SUV', 'active', 'Building resident'),
        ('GX15OCJ', 'Mike Brown', 'Van', 'active', 'Delivery service'),
        ('AP05JED', 'Emma Wilson', 'Car', 'active', 'Staff member'),
        ('RH05ZZK', 'David Lee', 'Truck', 'active', 'Maintenance crew'),
    ]
    
    added, skipped = db.add_multiple_vehicles(sample_vehicles)
    print(f"Added {added} vehicles, skipped {skipped} duplicates")
    
    return db


if __name__ == "__main__":
    # Initialize database with sample data
    db = initialize_sample_data()
    
    # Test queries
    print("\n=== All Registered Vehicles ===")
    vehicles = db.get_all_registered_vehicles()
    for v in vehicles:
        print(f"{v[1]} - {v[2]} ({v[3]}) - Status: {v[4]}")
    
    print("\n=== Testing License Plate Check ===")
    test_plate = "MV51VSU"
    result = db.check_license_plate(test_plate)
    print(f"Checking {test_plate}: {result}")
    
    print("\n=== Testing Fuzzy Match ===")
    test_plate = "MV51VSV"  # Slightly wrong
    matches = db.fuzzy_match(test_plate, threshold=0.7)
    print(f"Similar to {test_plate}:")
    for match in matches:
        print(f"  {match['plate']} - {match['similarity']:.2%} match")