def generate_policy_recommendation(weights, current_pm25, target_pm25=50.0):
    """
    Analyzes causal weights and generates a smart policy recommendation.
    
    Args:
        weights (BetaWeights): The causal weights from the Double ML model.
        current_pm25 (float): The current PM2.5 level.
        target_pm25 (float): The safe target level (default 50.0 for WHO/local standards).
    """
    coeffs = weights.coefficients
    total_attributable = sum(coeffs.values())
    
    if total_attributable == 0:
        return "Not enough data to attribute sources. Wait for the model to update."

    # Convert to percentages
    percents = {k: (v / total_attributable) * 100 for k, v in coeffs.items()}
    
    # Identify the primary driver
    primary_source = max(percents, key=percents.get)
    primary_pct = percents[primary_source]
    
    # Format names nicely
    names = {
        'traffic_ratio': 'Vehicular Traffic',
        'industry_ratio': 'Industrial Emissions',
        'dust_ratio': 'Construction Dust',
        'biomass_ratio': 'Biomass Burning'
    }
    
    advisor_text = f"**AI Policy Advisor:**\n\n"
    
    if current_pm25 <= target_pm25:
        advisor_text += f"✅ Air quality is currently within safe limits ({current_pm25:.1f} µg/m³). "
        advisor_text += f"Maintenance of existing policies is recommended. "
    else:
        excess = current_pm25 - target_pm25
        advisor_text += f"⚠️ Air quality is hazardous (exceeding target by {excess:.1f} µg/m³). "
        
    advisor_text += f"Currently, **{names[primary_source]}** is the dominant driver, accounting for **{primary_pct:.1f}%** of the local attributable pollution. "
    
    if primary_pct > 40:
        advisor_text += f"To achieve the highest immediate impact, emergency suppression policies should heavily target {names[primary_source]}."
    else:
        advisor_text += f"Since pollution is distributed across multiple sources, a broad-spectrum reduction across {names[primary_source]} and other sectors is required."
        
    return advisor_text
