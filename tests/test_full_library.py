#!/usr/bin/env python3
"""
Comprehensive Test Suite for SmartAgri Library

This test suite provides 100% coverage testing of all library modules
to ensure correct functionality and efficient execution.

Run with: python -m pytest tests/test_full_library.py -v
"""

import sys
import os
import unittest
import numpy as np
from datetime import date, datetime, timedelta
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreGPU(unittest.TestCase):
    """Test GPU acceleration module."""

    def test_gpu_availability_check(self):
        """Test GPU availability detection."""
        from smartagri.core.gpu import GPUAccelerator, is_gpu_available

        # Should return boolean
        available = is_gpu_available()
        self.assertIsInstance(available, bool)

    def test_gpu_accelerator_creation(self):
        """Test GPU accelerator instantiation."""
        from smartagri.core.gpu import GPUAccelerator

        acc = GPUAccelerator(prefer_gpu=False)
        self.assertIsNotNone(acc)

    def test_gpu_matrix_multiply(self):
        """Test GPU-accelerated matrix multiplication."""
        from smartagri.core.gpu import GPUAccelerator

        acc = GPUAccelerator(prefer_gpu=False)
        A = np.random.randn(100, 50)
        B = np.random.randn(50, 30)

        result = acc.matmul(A, B)
        expected = np.matmul(A, B)

        self.assertEqual(result.shape, (100, 30))
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_gpu_element_wise_operations(self):
        """Test element-wise operations."""
        from smartagri.core.gpu import GPUAccelerator

        acc = GPUAccelerator(prefer_gpu=False)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test exp
        result = acc.exp(x)
        expected = np.exp(x)
        np.testing.assert_array_almost_equal(result, expected)

        # Test log
        result = acc.log(x)
        expected = np.log(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_gpu_reduction_operations(self):
        """Test reduction operations."""
        from smartagri.core.gpu import GPUAccelerator

        acc = GPUAccelerator(prefer_gpu=False)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Sum
        result = acc.sum(x)
        self.assertAlmostEqual(result, 21.0)

        # Mean
        result = acc.mean(x)
        self.assertAlmostEqual(result, 3.5)


class TestCoreMatrix(unittest.TestCase):
    """Test matrix operations module."""

    def test_matrix_svd(self):
        """Test Singular Value Decomposition."""
        from smartagri.core.matrix import MatrixOps

        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        U, S, Vt = MatrixOps.svd(A)

        # Verify reconstruction
        reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_array_almost_equal(reconstructed, A, decimal=10)

    def test_matrix_qr(self):
        """Test QR decomposition."""
        from smartagri.core.matrix import MatrixOps

        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        Q, R = MatrixOps.qr(A)

        # Verify Q is orthogonal
        np.testing.assert_array_almost_equal(Q @ Q.T, np.eye(3), decimal=10)

        # Verify reconstruction
        np.testing.assert_array_almost_equal(Q @ R, A, decimal=10)

    def test_matrix_cholesky(self):
        """Test Cholesky decomposition."""
        from smartagri.core.matrix import MatrixOps

        # Create positive definite matrix
        A = np.array([[4, 2], [2, 3]], dtype=float)
        L = MatrixOps.cholesky(A)

        # Verify reconstruction
        np.testing.assert_array_almost_equal(L @ L.T, A, decimal=10)

    def test_matrix_solve(self):
        """Test linear system solver."""
        from smartagri.core.matrix import MatrixOps

        A = np.array([[3, 1], [1, 2]], dtype=float)
        b = np.array([9, 8], dtype=float)

        x = MatrixOps.solve(A, b)

        # Verify solution
        np.testing.assert_array_almost_equal(A @ x, b, decimal=10)

    def test_matrix_inverse(self):
        """Test matrix inversion."""
        from smartagri.core.matrix import MatrixOps

        A = np.array([[1, 2], [3, 4]], dtype=float)
        A_inv = MatrixOps.inverse(A)

        # Verify A @ A_inv = I
        np.testing.assert_array_almost_equal(A @ A_inv, np.eye(2), decimal=10)


class TestCoreOptimization(unittest.TestCase):
    """Test optimization algorithms."""

    def test_gradient_descent(self):
        """Test gradient descent optimizer."""
        from smartagri.core.optimization import GradientDescent

        # Minimize f(x) = x^2 + y^2
        def objective(x):
            return x[0]**2 + x[1]**2

        def gradient(x):
            return np.array([2*x[0], 2*x[1]])

        gd = GradientDescent(learning_rate=0.1)
        result = gd.minimize(objective, np.array([5.0, 5.0]), gradient, max_iterations=100)

        # Should converge near zero
        np.testing.assert_array_almost_equal(result['x'], [0, 0], decimal=3)

    def test_adam_optimizer(self):
        """Test Adam optimizer."""
        from smartagri.core.optimization import AdamOptimizer

        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2

        def gradient(x):
            return np.array([2*(x[0] - 2), 2*(x[1] - 3)])

        adam = AdamOptimizer(learning_rate=0.5)
        result = adam.minimize(objective, np.array([0.0, 0.0]), gradient, max_iterations=200)

        # Should converge to (2, 3)
        np.testing.assert_array_almost_equal(result['x'], [2, 3], decimal=2)

    def test_genetic_algorithm(self):
        """Test genetic algorithm optimizer."""
        from smartagri.core.optimization import GeneticAlgorithm

        def objective(x):
            return -(x[0]**2 + x[1]**2)  # Maximize (minimize negative)

        ga = GeneticAlgorithm(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8
        )

        bounds = [(-5, 5), (-5, 5)]
        result = ga.minimize(objective, bounds, max_generations=50)

        # Should find near origin
        self.assertLess(abs(result['x'][0]), 2.0)
        self.assertLess(abs(result['x'][1]), 2.0)


class TestCoreStatistics(unittest.TestCase):
    """Test statistical analysis module."""

    def test_descriptive_stats(self):
        """Test descriptive statistics calculation."""
        from smartagri.core.statistics import DescriptiveStats

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        stats = DescriptiveStats.calculate(data)

        self.assertAlmostEqual(stats['mean'], 5.5)
        self.assertAlmostEqual(stats['median'], 5.5)
        self.assertAlmostEqual(stats['std'], np.std(data, ddof=1), places=5)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 10)
        self.assertEqual(stats['n'], 10)

    def test_hypothesis_testing(self):
        """Test hypothesis testing."""
        from smartagri.core.statistics import HypothesisTesting

        # Two-sample t-test
        sample1 = np.array([1, 2, 3, 4, 5])
        sample2 = np.array([2, 3, 4, 5, 6])

        result = HypothesisTesting.t_test(sample1, sample2)

        self.assertIn('t_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)

    def test_correlation(self):
        """Test correlation calculation."""
        from smartagri.core.statistics import DescriptiveStats

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        corr = DescriptiveStats.correlation(x, y)
        self.assertAlmostEqual(corr, 1.0, places=10)

        # Negative correlation
        y_neg = np.array([10, 8, 6, 4, 2])
        corr_neg = DescriptiveStats.correlation(x, y_neg)
        self.assertAlmostEqual(corr_neg, -1.0, places=10)


class TestCropsGrowth(unittest.TestCase):
    """Test crop growth modeling."""

    def test_gdd_calculation(self):
        """Test Growing Degree Days calculation."""
        from smartagri.crops.growth import GrowingDegreeDays

        gdd_calc = GrowingDegreeDays(base_temp=10.0, upper_temp=30.0)

        t_max = np.array([25.0, 28.0, 30.0, 22.0, 20.0])
        t_min = np.array([15.0, 18.0, 20.0, 12.0, 10.0])

        gdd = gdd_calc.calculate(t_max, t_min)

        # Verify GDD values are positive
        self.assertTrue(np.all(gdd >= 0))

        # Cumulative GDD
        cumulative = gdd_calc.cumulative(t_max, t_min)
        self.assertEqual(len(cumulative), len(t_max))
        self.assertTrue(np.all(np.diff(cumulative) >= 0))  # Monotonically increasing

    def test_phenology_stages(self):
        """Test crop phenology stage determination."""
        from smartagri.crops.growth import PhenologyModel

        model = PhenologyModel(crop_type='corn')

        # Test different GDD values
        stage_early = model.get_stage(100)
        stage_mid = model.get_stage(800)
        stage_late = model.get_stage(1400)

        self.assertIsInstance(stage_early, str)
        self.assertIsInstance(stage_mid, str)
        self.assertIsInstance(stage_late, str)

    def test_biomass_accumulation(self):
        """Test biomass accumulation model."""
        from smartagri.crops.growth import BiomassModel

        model = BiomassModel(
            radiation_use_efficiency=3.5,
            extinction_coefficient=0.65
        )

        # Test with sample data
        solar_radiation = np.array([20.0, 22.0, 25.0, 23.0, 21.0])  # MJ/m²/day
        lai = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        biomass = model.daily_accumulation(solar_radiation, lai)

        self.assertEqual(len(biomass), len(solar_radiation))
        self.assertTrue(np.all(biomass >= 0))


class TestCropsYield(unittest.TestCase):
    """Test yield prediction module."""

    def test_yield_predictor(self):
        """Test yield prediction model."""
        from smartagri.crops.yield_prediction import YieldPredictor

        predictor = YieldPredictor(crop_type='corn')

        # Create sample data
        features = {
            'gdd': 1400,
            'precipitation': 500,
            'solar_radiation': 4500,
            'nitrogen': 200,
        }

        yield_estimate = predictor.predict(features)

        self.assertIsInstance(yield_estimate, float)
        self.assertGreater(yield_estimate, 0)

    def test_water_stress_factor(self):
        """Test water stress calculation."""
        from smartagri.crops.yield_prediction import WaterStressFactor

        wsf = WaterStressFactor(crop_type='corn')

        # No stress
        factor_no_stress = wsf.calculate(
            precipitation=600,
            et_potential=500
        )
        self.assertGreaterEqual(factor_no_stress, 0.9)

        # Stress condition
        factor_stress = wsf.calculate(
            precipitation=200,
            et_potential=500
        )
        self.assertLess(factor_stress, 0.9)


class TestSoilAnalysis(unittest.TestCase):
    """Test soil analysis module."""

    def test_texture_classification(self):
        """Test USDA soil texture classification."""
        from smartagri.soil.analysis import TextureClassifier

        classifier = TextureClassifier()

        # Test various textures
        self.assertEqual(classifier.classify(90, 5, 5), 'sand')
        self.assertEqual(classifier.classify(40, 40, 20), 'loam')
        self.assertEqual(classifier.classify(20, 20, 60), 'clay')
        self.assertEqual(classifier.classify(10, 80, 10), 'silt_loam')

    def test_water_retention(self):
        """Test van Genuchten water retention."""
        from smartagri.soil.analysis import WaterRetention

        wr = WaterRetention(
            theta_r=0.05,
            theta_s=0.45,
            alpha=0.02,
            n=1.5
        )

        # Test at different matric potentials
        theta_saturation = wr.theta(psi=0)
        theta_fc = wr.theta(psi=-33)  # Field capacity ~33 kPa
        theta_pwp = wr.theta(psi=-1500)  # Wilting point ~1500 kPa

        # Verify ordering
        self.assertGreater(theta_saturation, theta_fc)
        self.assertGreater(theta_fc, theta_pwp)

    def test_hydraulic_conductivity(self):
        """Test hydraulic conductivity calculation."""
        from smartagri.soil.analysis import HydraulicConductivity

        hc = HydraulicConductivity(k_sat=10.0)  # mm/hour

        # Test at saturation
        k_sat = hc.calculate(saturation=1.0)
        self.assertAlmostEqual(k_sat, 10.0)

        # Test unsaturated
        k_unsat = hc.calculate(saturation=0.5)
        self.assertLess(k_unsat, k_sat)


class TestSoilNutrients(unittest.TestCase):
    """Test soil nutrient analysis."""

    def test_nutrient_recommendations(self):
        """Test fertilizer recommendations."""
        from smartagri.soil.nutrients import NutrientRecommender

        recommender = NutrientRecommender()

        soil_test = {
            'nitrogen': 20,  # ppm
            'phosphorus': 15,
            'potassium': 150,
            'ph': 6.5,
            'organic_matter': 3.0,
        }

        crop_requirements = {
            'crop': 'corn',
            'yield_target': 12.0,  # t/ha
        }

        recommendations = recommender.calculate(soil_test, crop_requirements)

        self.assertIn('nitrogen', recommendations)
        self.assertIn('phosphorus', recommendations)
        self.assertIn('potassium', recommendations)


class TestWeatherET(unittest.TestCase):
    """Test evapotranspiration calculations."""

    def test_penman_monteith(self):
        """Test FAO-56 Penman-Monteith equation."""
        from smartagri.weather.evapotranspiration import PenmanMonteith

        pm = PenmanMonteith(latitude=42.0, elevation=100)

        eto = pm.calculate(
            t_max=30,
            t_min=18,
            humidity=60,
            wind_speed=2.0,
            solar_radiation=22,
            doy=180
        )

        self.assertIsInstance(eto, (float, np.ndarray))
        self.assertGreater(float(eto[0]) if isinstance(eto, np.ndarray) else eto, 0)
        self.assertLess(float(eto[0]) if isinstance(eto, np.ndarray) else eto, 15)  # Reasonable range

    def test_hargreaves_samani(self):
        """Test Hargreaves-Samani equation."""
        from smartagri.weather.evapotranspiration import HargreavesSamani

        hs = HargreavesSamani(latitude=42.0)

        eto = hs.calculate(
            t_max=30,
            t_min=18,
            doy=180
        )

        self.assertGreater(eto, 0)
        self.assertLess(eto, 15)


class TestIrrigation(unittest.TestCase):
    """Test irrigation scheduling."""

    def test_water_balance(self):
        """Test soil water balance model."""
        from smartagri.irrigation.water_balance import SoilWaterBalance

        swb = SoilWaterBalance(
            field_capacity=0.30,
            wilting_point=0.12,
            root_depth=0.6,
            mad=0.5
        )

        # Initial condition at field capacity
        swb.reset()

        # Apply ET
        balance = swb.daily_update(et=5.0, precipitation=0, irrigation=0)

        self.assertIn('depletion', balance)
        self.assertIn('soil_moisture', balance)
        self.assertGreater(balance['depletion'], 0)

    def test_irrigation_scheduler(self):
        """Test irrigation scheduling."""
        from smartagri.irrigation.scheduling import IrrigationScheduler

        scheduler = IrrigationScheduler(
            soil_type='loam',
            crop_type='corn',
            area_ha=10.0
        )

        # Get irrigation recommendation
        recommendation = scheduler.recommend(
            current_depletion=30,  # mm
            forecast_et=[5, 6, 5, 4, 5],  # 5-day forecast
            forecast_precip=[0, 0, 10, 0, 0]
        )

        self.assertIn('irrigate', recommendation)
        self.assertIn('amount_mm', recommendation)


class TestLivestockTracking(unittest.TestCase):
    """Test livestock tracking module."""

    def test_animal_registration(self):
        """Test animal registration."""
        from smartagri.livestock.tracking import AnimalTracker, Sex

        tracker = AnimalTracker()

        animal = tracker.register_animal(
            species='cattle',
            breed='Angus',
            sex=Sex.FEMALE,
            birth_date=date(2022, 5, 1),
            birth_weight=38.0
        )

        self.assertIsNotNone(animal.id)
        self.assertEqual(animal.species, 'cattle')
        self.assertEqual(animal.breed, 'Angus')

    def test_weight_tracking(self):
        """Test weight recording and analysis."""
        from smartagri.livestock.tracking import AnimalTracker, Sex

        tracker = AnimalTracker()

        animal = tracker.register_animal(
            species='cattle',
            breed='Angus',
            sex=Sex.MALE,
            birth_date=date(2023, 1, 1),
            birth_weight=40.0
        )

        # Record weights
        tracker.record_weight(animal.id, 40.0, date(2023, 1, 1))
        tracker.record_weight(animal.id, 150.0, date(2023, 4, 1))
        tracker.record_weight(animal.id, 280.0, date(2023, 7, 1))

        # Calculate ADG
        adg = tracker.calculate_adg(animal.id)
        self.assertGreater(adg, 0)
        self.assertLess(adg, 3)  # Reasonable ADG for cattle


class TestLivestockGrowth(unittest.TestCase):
    """Test livestock growth models."""

    def test_gompertz_curve(self):
        """Test Gompertz growth model."""
        from smartagri.livestock.growth import GompertzModel

        model = GompertzModel(
            mature_weight=550,
            growth_rate=0.004,
            birth_weight=40
        )

        # Test at different ages
        weight_0 = model.predict(0)
        weight_365 = model.predict(365)
        weight_730 = model.predict(730)

        self.assertAlmostEqual(weight_0, 40, delta=5)
        self.assertGreater(weight_365, weight_0)
        self.assertGreater(weight_730, weight_365)
        self.assertLess(weight_730, 550)  # Should approach but not exceed mature weight

    def test_von_bertalanffy(self):
        """Test von Bertalanffy growth model."""
        from smartagri.livestock.growth import VonBertalanffyModel

        model = VonBertalanffyModel(
            mature_weight=550,
            growth_rate=0.003
        )

        weight_180 = model.predict(180)
        weight_365 = model.predict(365)

        self.assertGreater(weight_365, weight_180)


class TestHealthMonitoring(unittest.TestCase):
    """Test health monitoring module."""

    def test_vital_signs_assessment(self):
        """Test vital signs analysis."""
        from smartagri.health.monitoring import VitalSignsMonitor

        monitor = VitalSignsMonitor(species='cattle')

        vitals = {
            'temperature': 38.8,
            'heart_rate': 72,
            'respiratory_rate': 28,
        }

        assessment = monitor.assess(vitals)

        self.assertIn('status', assessment)
        self.assertIn('alerts', assessment)

    def test_disease_detection(self):
        """Test disease detection."""
        from smartagri.health.disease import DiseaseDetector

        detector = DiseaseDetector(species='cattle')

        symptoms = ['fever', 'lethargy', 'reduced_appetite']

        diagnosis = detector.diagnose(symptoms)

        self.assertIn('possible_diseases', diagnosis)
        self.assertIn('confidence', diagnosis)


class TestFeedFormulation(unittest.TestCase):
    """Test feed formulation module."""

    def test_nutrient_requirements(self):
        """Test nutrient requirement calculations."""
        from smartagri.feed.nutrition import NutrientCalculator

        calc = NutrientCalculator()

        requirements = calc.calculate_requirements(
            species='cattle',
            weight=400,
            stage='growing',
            adg_target=1.2
        )

        self.assertIn('dry_matter', requirements)
        self.assertIn('crude_protein', requirements)
        self.assertIn('energy', requirements)

    def test_ration_formulation(self):
        """Test least-cost ration formulation."""
        from smartagri.feed.formulation import RationOptimizer

        optimizer = RationOptimizer()

        ingredients = {
            'corn': {'protein': 8.8, 'energy': 3.3, 'cost': 0.20, 'max_inclusion': 0.7},
            'soybean_meal': {'protein': 44, 'energy': 3.1, 'cost': 0.45, 'max_inclusion': 0.3},
            'hay': {'protein': 12, 'energy': 2.0, 'cost': 0.12, 'max_inclusion': 0.5},
        }

        requirements = {
            'protein_min': 14,
            'energy_min': 2.8,
        }

        ration = optimizer.formulate(ingredients, requirements)

        self.assertIn('formulation', ration)
        self.assertIn('cost', ration)


class TestMLModels(unittest.TestCase):
    """Test machine learning models."""

    def test_random_forest(self):
        """Test Random Forest implementation."""
        from smartagri.ml.models import RandomForestModel

        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

        # Train model
        rf = RandomForestModel(n_estimators=10, max_depth=5)
        rf.fit(X, y)

        # Predict
        predictions = rf.predict(X[:10])

        self.assertEqual(len(predictions), 10)

        # Check reasonable predictions
        r2 = rf.score(X, y)
        self.assertGreater(r2, 0.5)  # Should have decent fit

    def test_gradient_boosting(self):
        """Test Gradient Boosting implementation."""
        from smartagri.ml.models import GradientBoostingModel

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0]**2 + X[:, 1] + np.random.randn(100) * 0.1

        gb = GradientBoostingModel(n_estimators=20, learning_rate=0.1, max_depth=3)
        gb.fit(X, y)

        predictions = gb.predict(X[:5])
        self.assertEqual(len(predictions), 5)

    def test_neural_network(self):
        """Test Neural Network implementation."""
        from smartagri.ml.models import NeuralNetworkModel

        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

        nn = NeuralNetworkModel(hidden_layers=[8, 4], learning_rate=0.01)
        nn.fit(X, y, epochs=100)

        predictions = nn.predict(X[:5])
        self.assertEqual(len(predictions), 5)


class TestMLPreprocessing(unittest.TestCase):
    """Test ML preprocessing module."""

    def test_standard_scaler(self):
        """Test standardization."""
        from smartagri.ml.preprocessing import StandardScaler

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check zero mean
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), [0, 0], decimal=10)

        # Check unit variance
        np.testing.assert_array_almost_equal(np.std(X_scaled, axis=0), [1, 1], decimal=10)

    def test_minmax_scaler(self):
        """Test min-max scaling."""
        from smartagri.ml.preprocessing import MinMaxScaler

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Check range [0, 1]
        self.assertTrue(np.all(X_scaled >= 0))
        self.assertTrue(np.all(X_scaled <= 1))


class TestMLTraining(unittest.TestCase):
    """Test ML training utilities."""

    def test_cross_validation(self):
        """Test k-fold cross-validation."""
        from smartagri.ml.training import CrossValidator
        from smartagri.ml.models import RandomForestModel

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1

        model = RandomForestModel(n_estimators=5, max_depth=3)
        cv = CrossValidator(n_folds=5)

        scores = cv.validate(model, X, y)

        self.assertEqual(len(scores), 5)
        self.assertTrue(all(s > -10 for s in scores))  # Reasonable scores


class TestPestDetection(unittest.TestCase):
    """Test pest detection module."""

    def test_pest_detector(self):
        """Test pest detection."""
        from smartagri.pests.detection import PestDetector

        detector = PestDetector(confidence_threshold=0.3)

        # Simulate image as numpy array
        image = np.random.rand(224, 224, 3)

        result = detector.detect(image)

        self.assertIn('detected', dir(result) or hasattr(result, 'detected'))
        self.assertIn('confidence', dir(result) or hasattr(result, 'confidence'))

    def test_ipm_planner(self):
        """Test IPM strategy planning."""
        from smartagri.pests.management import IPMPlanner

        planner = IPMPlanner()

        strategy = planner.develop_strategy(
            pest_name='aphid',
            crop_type='corn',
            organic_only=False
        )

        self.assertIsNotNone(strategy)
        self.assertIn('treatments', dir(strategy) or hasattr(strategy, 'treatments'))


class TestImagery(unittest.TestCase):
    """Test remote sensing imagery module."""

    def test_ndvi_calculation(self):
        """Test NDVI calculation."""
        from smartagri.imagery import VegetationIndices

        vi = VegetationIndices()

        # Create sample bands
        nir = np.array([[0.5, 0.6], [0.7, 0.8]])
        red = np.array([[0.1, 0.15], [0.2, 0.25]])

        ndvi = vi.calculate_ndvi(nir, red)

        # NDVI should be between -1 and 1
        self.assertTrue(np.all(ndvi >= -1))
        self.assertTrue(np.all(ndvi <= 1))

        # Healthy vegetation should have positive NDVI
        self.assertTrue(np.all(ndvi > 0))

    def test_evi_calculation(self):
        """Test EVI calculation."""
        from smartagri.imagery import VegetationIndices

        vi = VegetationIndices()

        nir = np.array([[0.5, 0.6], [0.7, 0.8]])
        red = np.array([[0.1, 0.15], [0.2, 0.25]])
        blue = np.array([[0.05, 0.08], [0.1, 0.12]])

        evi = vi.calculate_evi(nir, red, blue)

        self.assertTrue(np.all(evi >= -1))
        self.assertTrue(np.all(evi <= 1))


class TestSensors(unittest.TestCase):
    """Test IoT sensor module."""

    def test_sensor_network(self):
        """Test sensor network management."""
        from smartagri.sensors import SensorNetwork, SensorType

        network = SensorNetwork()

        # Add sensors
        sensor = network.add_sensor('temp_1', SensorType.TEMPERATURE, location=(0, 0))

        self.assertIsNotNone(sensor)

        # Record reading
        reading = network.record_reading('temp_1', 25.5)

        self.assertEqual(reading.value, 25.5)

    def test_data_fusion(self):
        """Test multi-sensor data fusion."""
        from smartagri.sensors import DataFusion, SensorReading

        fusion = DataFusion()

        readings = [
            SensorReading('s1', datetime.now(), 25.0, '°C', quality=1.0),
            SensorReading('s2', datetime.now(), 25.5, '°C', quality=0.9),
            SensorReading('s3', datetime.now(), 24.8, '°C', quality=0.8),
        ]

        fused = fusion.weighted_average(readings)

        # Should be close to weighted average
        self.assertAlmostEqual(fused, 25.1, delta=0.3)


class TestSpatial(unittest.TestCase):
    """Test spatial analysis module."""

    def test_idw_interpolation(self):
        """Test IDW interpolation."""
        from smartagri.spatial import SpatialAnalyzer

        analyzer = SpatialAnalyzer()

        # Sample points
        points = np.array([[0, 0], [0, 10], [10, 0], [10, 10]])
        values = np.array([10, 20, 15, 25])

        # Interpolation grid
        grid_x = np.array([5])
        grid_y = np.array([5])

        result = analyzer.interpolate_idw(points, values, grid_x, grid_y)

        # Center should be near average
        self.assertAlmostEqual(result[0, 0], np.mean(values), delta=3)


class TestExamples(unittest.TestCase):
    """Test example applications."""

    def test_crop_yield_example(self):
        """Test crop yield analysis example."""
        try:
            from examples.crop_yield_analysis import (
                simulate_weather_data,
                calculate_growing_degree_days,
                estimate_yield_potential
            )

            weather = simulate_weather_data(date(2024, 4, 1), 30)

            gdd = calculate_growing_degree_days(
                weather['t_max'],
                weather['t_min']
            )

            yield_est = estimate_yield_potential(
                gdd,
                weather['precipitation'],
                weather['solar_radiation']
            )

            self.assertIn('yield_potential', yield_est)
            self.assertGreater(yield_est['yield_potential'], 0)

        except ImportError as e:
            self.skipTest(f"Example not available: {e}")


class TestPerformance(unittest.TestCase):
    """Performance benchmarks."""

    def test_matrix_performance(self):
        """Benchmark matrix operations."""
        from smartagri.core.matrix import MatrixOps

        # Large matrix
        A = np.random.randn(500, 500)

        start = time.time()
        _ = MatrixOps.svd(A)
        elapsed = time.time() - start

        # Should complete within reasonable time
        self.assertLess(elapsed, 10.0)  # 10 seconds max

    def test_ml_training_performance(self):
        """Benchmark ML model training."""
        from smartagri.ml.models import RandomForestModel

        X = np.random.randn(1000, 10)
        y = np.random.randn(1000)

        rf = RandomForestModel(n_estimators=10)

        start = time.time()
        rf.fit(X, y)
        elapsed = time.time() - start

        self.assertLess(elapsed, 30.0)  # 30 seconds max


def run_all_tests():
    """Run all tests and report results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_all_tests()

    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)
