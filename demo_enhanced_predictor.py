#!/usr/bin/env python3
"""
Demo script for enhanced circuit predictor
Shows the capabilities without requiring user input
"""

from enhanced_circuit_predictor import EnhancedCircuitPredictor

def demo_enhanced_predictor():
    """Demo the enhanced predictor with Dutch GP"""
    print("🎯 ENHANCED CIRCUIT PREDICTOR DEMO")
    print("="*60)
    
    predictor = EnhancedCircuitPredictor()
    predictor.load_data()
    
    # Display available races
    print("\n🏁 AVAILABLE RACES FOR 2025")
    print("="*50)
    year_races = predictor.display_available_races(2025)
    
    # Demo with Dutch GP (Round 15)
    print("\n🎯 DEMO: Dutch Grand Prix (Round 15)")
    print("="*50)
    
    race_info = predictor.get_race_info(15, 2025)
    if not race_info:
        print("❌ Could not get race info for Dutch GP")
        return
    
    print(f"🏁 Race: {race_info['grand_prix_name']}")
    print(f"📅 Date: {race_info['date']}")
    print(f"🏟️  Circuit: {race_info['circuit_name']}")
    print(f"🌐 F1.com Race Number: {race_info['f1_race_number']}")
    
    # Check if it's race weekend
    is_weekend = predictor.check_if_race_weekend(race_info)
    print(f"🕐 Race weekend detected: {is_weekend}")
    
    if is_weekend:
        print("🟡 Would fetch live session data from:")
        print(f"   FP1: https://www.formula1.com/en/results/2025/races/{race_info['f1_race_number']}/{race_info['grand_prix']}/practice/1")
        print(f"   FP2: https://www.formula1.com/en/results/2025/races/{race_info['f1_race_number']}/{race_info['grand_prix']}/practice/2")
        print(f"   FP3: https://www.formula1.com/en/results/2025/races/{race_info['f1_race_number']}/{race_info['grand_prix']}/practice/3")
        print(f"   Q:   https://www.formula1.com/en/results/2025/races/{race_info['f1_race_number']}/{race_info['grand_prix']}/qualifying")
    
    # Get drivers from previous race
    drivers_list = predictor.get_drivers_from_previous_race(race_info)
    
    # Prepare features
    X, feature_labels = predictor.prepare_enhanced_features(drivers_list, race_info)
    
    print(f"\n📊 FEATURE ANALYSIS COMPLETE")
    print(f"✅ Drivers analyzed: {len(drivers_list)}")
    print(f"✅ Features per driver: {X.shape[1]}")
    print(f"✅ Total feature dimensions: {X.shape[0]} x {X.shape[1]}")
    
    # Show feature breakdown
    print(f"\n🔧 FEATURE BREAKDOWN:")
    print(f"   • Driver circuit performance: 5 features")
    print(f"   • Constructor circuit performance: 5 features")
    print(f"   • Season trends: 3 features")
    print(f"   • Previous race performance: 2 features")
    print(f"   • Current weekend performance: 5 features")
    print(f"   • Circuit characteristics: 4 features")
    print(f"   • Total: {X.shape[1]} features")
    
    # Show some example features for first driver
    if len(drivers_list) > 0:
        driver = drivers_list[0]
        print(f"\n📊 EXAMPLE FEATURES FOR {driver['driver_name']}:")
        
        # Circuit performance
        driver_circuit = predictor.analyze_driver_circuit_performance(driver['driver_id'], race_info['circuit'])
        print(f"   Circuit races: {driver_circuit['races']}")
        print(f"   Avg position: {driver_circuit['avg_position']:.1f}")
        print(f"   Best position: {driver_circuit['best_position']}")
        print(f"   Consistency: {driver_circuit['consistency']:.2f}")
        
        # Season trends
        season_trends = predictor.get_season_trends(driver['driver_id'], driver['constructor_id'])
        print(f"   Season trend: {season_trends['driver_trend']:.2f}")
        print(f"   Races completed: {season_trends['races_completed']}")
        
        # Weekend performance (if available)
        weekend_performance = predictor.get_current_weekend_performance(driver['driver_name'])
        if is_weekend:
            print(f"   FP1 position: {weekend_performance['fp1_position']}")
            print(f"   FP2 position: {weekend_performance['fp2_position']}")
            print(f"   FP3 position: {weekend_performance['fp3_position']}")
            print(f"   Qualifying: {weekend_performance['qualifying_position']}")
            print(f"   Weekend trend: {weekend_performance['weekend_trend']:.2f}")
    
    print(f"\n🎉 ENHANCED PREDICTOR DEMO COMPLETE!")
    print(f"✅ Ready for predictions with live weekend data integration")
    print(f"✅ User can select any race from 1-24")
    print(f"✅ Automatic detection of race weekends")
    print(f"✅ Live data fetching from Formula1.com")

if __name__ == "__main__":
    demo_enhanced_predictor()