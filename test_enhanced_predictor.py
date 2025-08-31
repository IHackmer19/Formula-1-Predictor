#!/usr/bin/env python3
"""
Test script for enhanced circuit predictor
"""

from enhanced_circuit_predictor import EnhancedCircuitPredictor

def test_enhanced_predictor():
    """Test the enhanced predictor with Dutch GP"""
    print("🧪 TESTING ENHANCED CIRCUIT PREDICTOR")
    print("="*50)
    
    predictor = EnhancedCircuitPredictor()
    predictor.load_data()
    
    # Test with Dutch GP (Round 15)
    race_info = predictor.get_race_info(15, 2025)
    if not race_info:
        print("❌ Could not get race info for Dutch GP")
        return
    
    print(f"✅ Race info: {race_info['grand_prix_name']}")
    
    # Check if it's race weekend
    is_weekend = predictor.check_if_race_weekend(race_info)
    print(f"🕐 Race weekend: {is_weekend}")
    
    # Get drivers from previous race
    drivers_list = predictor.get_drivers_from_previous_race(race_info)
    print(f"✅ Found {len(drivers_list)} drivers")
    
    # Test feature preparation
    X, feature_labels = predictor.prepare_enhanced_features(drivers_list, race_info)
    print(f"✅ Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    print("🎉 Test completed successfully!")

if __name__ == "__main__":
    test_enhanced_predictor()