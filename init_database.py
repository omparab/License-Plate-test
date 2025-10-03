"""
Script to initialize the license plate database with sample data
Run this before starting the Streamlit app
"""

from database import initialize_sample_data, LicensePlateDB

def main():
    print("=" * 60)
    print("License Plate Database Initialization")
    print("=" * 60)
    
    # Initialize database with sample data
    db = initialize_sample_data()
    
    print("\n✅ Database initialized successfully!")
    print("\n" + "=" * 60)
    print("Registered Vehicles:")
    print("=" * 60)
    
    vehicles = db.get_all_registered_vehicles()
    
    for v in vehicles:
        print(f"\n📋 {v[1]}")
        print(f"   Owner: {v[2]}")
        print(f"   Type: {v[3]}")
        print(f"   Status: {v[4]}")
        print(f"   Registered: {v[5]}")
        if v[6]:
            print(f"   Notes: {v[6]}")
    
    print("\n" + "=" * 60)
    print(f"Total vehicles in database: {len(vehicles)}")
    print("=" * 60)
    
    # Test database queries
    print("\n🔍 Testing Database Queries...")
    
    # Test exact match
    test_plate = "MV51VSU"
    result = db.check_license_plate(test_plate)
    if result['found']:
        print(f"✅ Exact match test passed: {test_plate} found")
    else:
        print(f"❌ Exact match test failed: {test_plate} not found")
    
    # Test fuzzy match
    test_plate_fuzzy = "MV51VSV"  # Wrong last character
    matches = db.fuzzy_match(test_plate_fuzzy, threshold=0.7)
    if matches:
        print(f"✅ Fuzzy match test passed: {test_plate_fuzzy} matched to {matches[0]['plate']}")
    else:
        print(f"❌ Fuzzy match test failed: No matches for {test_plate_fuzzy}")
    
    print("\n✨ Database is ready to use!")
    print("Run 'streamlit run app.py' to start the application\n")

if __name__ == "__main__":
    main()