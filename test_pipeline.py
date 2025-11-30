"""
Test script สำหรับทดสอบ data preparation pipeline
รันไฟล์นี้เพื่อดู DEBUG logs และตรวจสอบว่าระบบทำงานถูกต้อง
"""

from data_prep import get_prepared_data

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Data Preparation Pipeline")
    print("="*70 + "\n")
    
    # รัน pipeline
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = get_prepared_data()
    
    print("\n" + "="*70)
    print("Testing Complete! Summary:")
    print("="*70)
    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ X_val shape:   {X_val.shape}")
    print(f"✓ X_test shape:  {X_test.shape}")
    print(f"✓ Preprocessor:  {type(preprocessor).__name__}")
    print("="*70 + "\n")
