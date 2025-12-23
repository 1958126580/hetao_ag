"""
Comprehensive Test Suite for SmartAgri Library

Tests all major modules with high coverage:
- Core mathematical operations
- Crop management
- Soil analysis
- Weather processing
- Irrigation scheduling
- Livestock management
- Health monitoring
- Feed optimization
- Machine learning

Run with: pytest tests/test_smartagri.py -v
"""

import numpy as np
import pytest
from datetime import date, datetime, timedelta
from typing import List, Dict, Any

# ============================================================================
# Core Module Tests
# ============================================================================

class TestMatrixOperations:
    """Tests for matrix algebra operations."""

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        from smartagri.core.matrix import MatrixOps

        ops = MatrixOps()
        A = np.random.randn(100, 50)
        B = np.random.randn(50, 30)

        result = ops.matmul(A, B)
        expected = A @ B

        assert result.shape == (100, 30)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_svd_decomposition(self):
        """Test SVD decomposition."""
        from smartagri.core.matrix import MatrixOps

        ops = MatrixOps()
        A = np.random.randn(50, 30)

        U, S, Vt = ops.svd(A)

        # Reconstruct and verify
        reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_allclose(reconstructed, A, rtol=1e-5)

    def test_qr_decomposition(self):
        """Test QR decomposition."""
        from smartagri.core.matrix import MatrixOps

        ops = MatrixOps()
        A = np.random.randn(50, 30)

        Q, R = ops.qr(A)

        # Verify orthogonality
        identity = Q.T @ Q
        np.testing.assert_allclose(identity, np.eye(Q.shape[1]), rtol=1e-5)

    def test_solve_linear_system(self):
        """Test linear system solving."""
        from smartagri.core.matrix import MatrixOps

        ops = MatrixOps()
        A = np.random.randn(50, 50)
        A = A @ A.T + np.eye(50)  # Positive definite
        b = np.random.randn(50)

        x = ops.solve(A, b)

        # Verify solution
        np.testing.assert_allclose(A @ x, b, rtol=1e-5)


class TestOptimization:
    """Tests for optimization algorithms."""

    def test_gradient_descent(self):
        """Test gradient descent optimization."""
        from smartagri.core.optimization import GradientDescent

        # Minimize x^2
        def objective(x):
            return x[0]**2 + x[1]**2

        def gradient(x):
            return np.array([2*x[0], 2*x[1]])

        gd = GradientDescent(learning_rate=0.1)
        x0 = np.array([5.0, 3.0])
        result = gd.minimize(objective, x0, gradient)

        assert result['success']
        np.testing.assert_allclose(result['x'], [0, 0], atol=0.01)

    def test_adam_optimizer(self):
        """Test Adam optimizer."""
        from smartagri.core.optimization import AdamOptimizer

        def objective(x):
            return (x[0] - 2)**2 + (x[1] + 1)**2

        def gradient(x):
            return np.array([2*(x[0] - 2), 2*(x[1] + 1)])

        adam = AdamOptimizer(learning_rate=0.5)
        x0 = np.array([0.0, 0.0])
        result = adam.minimize(objective, x0, gradient, max_iterations=200)

        assert result['success']
        np.testing.assert_allclose(result['x'], [2, -1], atol=0.1)


class TestStatistics:
    """Tests for statistical computing."""

    def test_descriptive_stats(self):
        """Test descriptive statistics."""
        from smartagri.core.statistics import DescriptiveStats

        stats = DescriptiveStats()
        data = np.random.randn(1000)

        result = stats.describe(data)

        assert 'mean' in result
        assert 'std' in result
        assert 'min' in result
        assert 'max' in result
        np.testing.assert_allclose(result['mean'], np.mean(data))

    def test_correlation(self):
        """Test correlation calculation."""
        from smartagri.core.statistics import CorrelationAnalysis

        corr = CorrelationAnalysis()
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1  # Highly correlated

        result = corr.pearson(x, y)

        assert result['correlation'] > 0.9
        assert result['p_value'] < 0.01


# ============================================================================
# Crop Module Tests
# ============================================================================

class TestCropGrowth:
    """Tests for crop growth modeling."""

    def test_gdd_calculation(self):
        """Test Growing Degree Days calculation."""
        from smartagri.crops.growth import GrowthModel

        model = GrowthModel(crop="corn")

        gdd = model.calculate_gdd(t_max=30, t_min=15, t_base=10)

        assert gdd == 12.5  # (30+15)/2 - 10

    def test_phenology_stages(self):
        """Test phenological stage determination."""
        from smartagri.crops.growth import PhenologyModel

        model = PhenologyModel(crop="wheat")

        stage = model.get_stage(gdd_accumulated=500)

        assert stage in ["vegetative", "reproductive", "maturity", "germination"]

    def test_yield_prediction(self):
        """Test yield prediction model."""
        from smartagri.crops.yield_prediction import YieldPredictor

        predictor = YieldPredictor(model_type="statistical")

        # Create sample data
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100) * 1000 + 5000
        X_test = np.random.randn(20, 5)

        predictor.fit(X_train, y_train)
        predictions = predictor.predict(X_test)

        assert len(predictions) == 20


# ============================================================================
# Soil Module Tests
# ============================================================================

class TestSoilAnalysis:
    """Tests for soil analysis."""

    def test_texture_classification(self):
        """Test soil texture classification."""
        from smartagri.soil.analysis import TextureClassifier

        classifier = TextureClassifier()

        texture = classifier.classify(sand=40, silt=40, clay=20)

        assert texture in ["loam", "sandy_loam", "clay_loam", "silt_loam"]

    def test_water_retention(self):
        """Test water retention curve calculation."""
        from smartagri.soil.analysis import WaterRetention

        retention = WaterRetention()

        # Van Genuchten parameters for loam
        theta = retention.van_genuchten(
            h=100,  # cm suction
            theta_r=0.078,
            theta_s=0.43,
            alpha=0.036,
            n=1.56
        )

        assert 0 < theta < 0.43

    def test_nutrient_recommendations(self):
        """Test nutrient recommendations."""
        from smartagri.soil.nutrients import NutrientAdvisor

        advisor = NutrientAdvisor()

        recommendations = advisor.recommend(
            crop="corn",
            target_yield=12,  # t/ha
            soil_n=50,
            soil_p=25,
            soil_k=150
        )

        assert 'nitrogen' in recommendations
        assert 'phosphorus' in recommendations
        assert 'potassium' in recommendations


# ============================================================================
# Weather Module Tests
# ============================================================================

class TestEvapotranspiration:
    """Tests for evapotranspiration calculations."""

    def test_penman_monteith(self):
        """Test Penman-Monteith ET calculation."""
        from smartagri.weather.evapotranspiration import PenmanMonteith

        pm = PenmanMonteith()

        et0 = pm.calculate(
            t_mean=25,
            t_min=18,
            t_max=32,
            rh=60,
            wind_speed=2.0,
            solar_radiation=22,
            latitude=45,
            day_of_year=180
        )

        assert 3 < et0 < 8  # Reasonable range for summer day

    def test_hargreaves(self):
        """Test Hargreaves-Samani ET calculation."""
        from smartagri.weather.evapotranspiration import HargreavesSamani

        hs = HargreavesSamani()

        et0 = hs.calculate(
            t_mean=25,
            t_min=18,
            t_max=32,
            latitude=45,
            day_of_year=180
        )

        assert 3 < et0 < 10


# ============================================================================
# Irrigation Module Tests
# ============================================================================

class TestIrrigation:
    """Tests for irrigation scheduling."""

    def test_water_balance(self):
        """Test water balance calculation."""
        from smartagri.irrigation.water_balance import WaterBalance

        balance = WaterBalance(soil_type="loam", root_depth=0.6)

        balance.add_irrigation(30.0)
        balance.add_et(5.0)

        status = balance.get_status()

        assert 'current_moisture_mm' in status
        assert 'depletion_fraction' in status

    def test_irrigation_scheduling(self):
        """Test irrigation schedule generation."""
        from smartagri.irrigation.scheduling import IrrigationScheduler

        scheduler = IrrigationScheduler()
        scheduler.add_zone("zone_1", name="North Field", area=10, crop="corn")

        schedule = scheduler.generate_schedule(
            days=7,
            et_forecast=[5.0] * 7
        )

        assert isinstance(schedule, list)


# ============================================================================
# Livestock Module Tests
# ============================================================================

class TestLivestockTracking:
    """Tests for livestock tracking."""

    def test_animal_registration(self):
        """Test animal registration."""
        from smartagri.livestock.tracking import AnimalTracker, Sex

        tracker = AnimalTracker()

        animal = tracker.register_animal(
            species="cattle",
            breed="Angus",
            sex=Sex.FEMALE,
            birth_date=date(2023, 3, 15)
        )

        assert animal.id is not None
        assert animal.species == "cattle"

    def test_weight_recording(self):
        """Test weight recording."""
        from smartagri.livestock.tracking import AnimalTracker

        tracker = AnimalTracker()
        animal = tracker.register_animal(species="cattle")

        tracker.record_weight(animal.id, 250.0)
        tracker.record_weight(animal.id, 300.0)

        history = tracker.get_weight_history(animal.id)

        assert len(history) == 2

    def test_growth_prediction(self):
        """Test growth prediction."""
        from smartagri.livestock.growth import GrowthModel

        model = GrowthModel(species="cattle", breed="angus")

        weight = model.predict_weight(age_days=365)

        assert 300 < weight < 600  # Reasonable range


class TestReproduction:
    """Tests for reproduction management."""

    def test_heat_prediction(self):
        """Test estrus prediction."""
        from smartagri.livestock.reproduction import EstrusDetector

        detector = EstrusDetector(species="cattle")

        detector.record_heat("cow_001", date(2024, 1, 1))
        detector.record_heat("cow_001", date(2024, 1, 22))

        next_heat = detector.predict_next_heat("cow_001")

        assert next_heat is not None

    def test_gestation_tracking(self):
        """Test pregnancy monitoring."""
        from smartagri.livestock.reproduction import PregnancyManager

        manager = PregnancyManager(species="cattle")

        manager.set_breeding_date("cow_001", date(2024, 1, 15))

        due_date = manager.predict_due_date("cow_001")

        assert due_date is not None
        assert (due_date - date(2024, 1, 15)).days == 283  # Cattle gestation


# ============================================================================
# Health Module Tests
# ============================================================================

class TestHealthMonitoring:
    """Tests for health monitoring."""

    def test_vital_sign_analysis(self):
        """Test vital sign analysis."""
        from smartagri.health.monitoring import VitalSignAnalyzer, VitalSigns

        analyzer = VitalSignAnalyzer(species="cattle")

        vitals = VitalSigns(
            temperature=39.5,
            heart_rate=70,
            respiratory_rate=25
        )

        result = analyzer.analyze(vitals)

        assert 'status' in result
        assert 'abnormalities' in result

    def test_disease_detection(self):
        """Test disease symptom matching."""
        from smartagri.health.disease import DiseaseDetector

        detector = DiseaseDetector(species="cattle")

        symptoms = ["fever", "cough", "nasal_discharge"]

        risks = detector.identify_disease(symptoms)

        assert len(risks) > 0
        assert risks[0].disease_name is not None


# ============================================================================
# Feed Module Tests
# ============================================================================

class TestFeedFormulation:
    """Tests for feed formulation."""

    def test_nutrient_requirements(self):
        """Test nutrient requirement calculation."""
        from smartagri.feed.nutrition import NutrientCalculator

        calc = NutrientCalculator(species="cattle")

        requirements = calc.calculate_requirements(
            weight=500,
            production="lactating",
            milk_yield=30
        )

        assert requirements.dry_matter > 0
        assert requirements.crude_protein > 0
        assert requirements.energy_me > 0

    def test_ration_optimization(self):
        """Test ration optimization."""
        from smartagri.feed.formulation import RationOptimizer
        from smartagri.feed.nutrition import NutrientCalculator

        calc = NutrientCalculator(species="cattle")
        requirements = calc.calculate_requirements(weight=500)

        optimizer = RationOptimizer(species="cattle")

        ration = optimizer.formulate_least_cost(
            requirements=requirements,
            target_dmi=12.0,
            available_feeds=["corn_grain", "soybean_meal", "alfalfa_hay"]
        )

        assert ration.total_dm > 0
        assert ration.cost >= 0


# ============================================================================
# ML Module Tests
# ============================================================================

class TestMLModels:
    """Tests for machine learning models."""

    def test_random_forest(self):
        """Test Random Forest model."""
        from smartagri.ml.models import RandomForestModel

        model = RandomForestModel(n_estimators=10, max_depth=5)

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100

        # Check R² is reasonable
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot

        assert r2 > 0.5

    def test_gradient_boosting(self):
        """Test Gradient Boosting model."""
        from smartagri.ml.models import GradientBoostingModel

        model = GradientBoostingModel(n_estimators=20, learning_rate=0.1)

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100

    def test_neural_network(self):
        """Test Neural Network model."""
        from smartagri.ml.models import NeuralNetworkModel

        model = NeuralNetworkModel(
            hidden_layers=[32, 16],
            epochs=50,
            learning_rate=0.01
        )

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100

    def test_preprocessing(self):
        """Test data preprocessing."""
        from smartagri.ml.preprocessing import DataPreprocessor

        prep = DataPreprocessor(method="standardize")

        X = np.random.randn(100, 5) * 10 + 50

        X_scaled = prep.fit_transform(X)

        # Check standardization
        np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0, atol=0.1)
        np.testing.assert_allclose(np.std(X_scaled, axis=0), 1, atol=0.1)

    def test_cross_validation(self):
        """Test cross-validation."""
        from smartagri.ml.training import CrossValidator
        from smartagri.ml.models import RandomForestModel

        cv = CrossValidator(n_folds=3)
        model = RandomForestModel(n_estimators=10)

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1)

        results = cv.validate(model, X, y, scoring="mse")

        assert 'mean' in results
        assert 'std' in results
        assert len(results['scores']) == 3


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_yield_prediction_workflow(self):
        """Test complete yield prediction workflow."""
        from smartagri.ml.applications import YieldPredictor

        # Generate sample agricultural data
        n_samples = 200

        # Features: rainfall, temperature, soil_n, soil_p, solar_radiation
        X = np.column_stack([
            np.random.uniform(500, 1000, n_samples),  # Rainfall mm
            np.random.uniform(15, 30, n_samples),     # Temperature C
            np.random.uniform(20, 100, n_samples),    # Soil N
            np.random.uniform(10, 50, n_samples),     # Soil P
            np.random.uniform(15, 25, n_samples),     # Solar radiation
        ])

        # Yield (t/ha) - simple relationship for testing
        y = 2 + 0.005 * X[:, 0] + 0.1 * X[:, 1] + 0.02 * X[:, 2] + np.random.randn(n_samples) * 0.5

        # Split data
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train and evaluate
        predictor = YieldPredictor(model_type="ensemble")
        predictor.fit(X_train, y_train)

        metrics = predictor.evaluate(X_test, y_test)

        assert metrics.r2 > 0  # Should have some predictive power

    def test_complete_livestock_workflow(self):
        """Test complete livestock management workflow."""
        from smartagri.livestock import AnimalTracker, GrowthModel, Sex
        from smartagri.livestock.reproduction import ReproductionManager

        # Create tracker
        tracker = AnimalTracker()

        # Register animals
        dam = tracker.register_animal(
            species="cattle",
            breed="Angus",
            sex=Sex.FEMALE,
            birth_date=date(2020, 3, 15)
        )

        sire = tracker.register_animal(
            species="cattle",
            breed="Angus",
            sex=Sex.MALE,
            birth_date=date(2019, 5, 20)
        )

        # Record weights
        tracker.record_weight(dam.id, 550)
        tracker.record_weight(sire.id, 750)

        # Track reproduction
        repro = ReproductionManager(species="cattle")
        repro.record_breeding(dam.id, sire.id, date(2024, 4, 1))
        repro.confirm_pregnancy(dam.id, date(2024, 5, 15), pregnant=True)

        due_date = repro.predict_due_date(dam.id)

        assert due_date is not None

        # Get summary
        summary = tracker.get_herd_summary()

        assert summary['total_active'] == 2


# ============================================================================
# Security Tests
# ============================================================================

class TestSecurity:
    """Security and input validation tests."""

    def test_input_validation(self):
        """Test input validation for edge cases."""
        from smartagri.crops.growth import GrowthModel

        model = GrowthModel(crop="corn")

        # Test with negative values
        gdd = model.calculate_gdd(t_max=10, t_min=15, t_base=10)  # Invalid: max < min
        assert gdd >= 0  # Should handle gracefully

        # Test with extreme values
        gdd = model.calculate_gdd(t_max=100, t_min=-50, t_base=10)
        assert gdd >= 0

    def test_numeric_stability(self):
        """Test numerical stability with edge cases."""
        from smartagri.ml.preprocessing import DataPreprocessor

        prep = DataPreprocessor(method="standardize")

        # Test with constant values
        X = np.ones((100, 5))
        X_scaled = prep.fit_transform(X)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(X_scaled))
        assert not np.any(np.isinf(X_scaled))

    def test_missing_value_handling(self):
        """Test handling of missing values."""
        from smartagri.ml.preprocessing import DataPreprocessor

        prep = DataPreprocessor(method="standardize", handle_missing="mean")

        X = np.random.randn(100, 5)
        X[10:20, 2] = np.nan  # Add some NaN values

        X_processed = prep.fit_transform(X)

        # Should not contain NaN after processing
        assert not np.any(np.isnan(X_processed))


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and efficiency tests."""

    def test_large_matrix_operations(self):
        """Test performance with large matrices."""
        from smartagri.core.matrix import MatrixOps

        ops = MatrixOps()

        # Large matrix multiplication
        A = np.random.randn(500, 500)
        B = np.random.randn(500, 500)

        result = ops.matmul(A, B)

        assert result.shape == (500, 500)

    def test_ml_model_scaling(self):
        """Test ML model performance with larger datasets."""
        from smartagri.ml.models import RandomForestModel

        model = RandomForestModel(n_estimators=20)

        # Larger dataset
        X = np.random.randn(1000, 20)
        y = np.sum(X, axis=1)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
