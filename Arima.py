"""
===============================================================================
ARIMA Time Series Analysis for FaaS Invocation Prediction
WITH TWO-STAGE Quantile MODEL FOR ZERO-INFLATED DATA
===============================================================================

This notebook implements the statistically correct approach for sparse FaaS data:
- Quantile ARIMA that separates zero/non-zero decision from magnitude
- Preserves all zeros (structural, not noise)
- Forecasts exactly 1440 minutes per function
===============================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from datetime import datetime

# Statistical libraries
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configuration
SELECTED_FUNCTIONS_PATH = Path('../selected_functions/')
OUTPUT_PATH = Path('./output/arima_analysis/')
FIGURES_PATH = OUTPUT_PATH / 'figures'

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# FIXED parameters
TRAINING_DAYS = 9  # Days 1-11 for training
FORECAST_HORIZON = 1440 * 3  # Days 10-12 = 3 days for prediction
TRAIN_TEST_SPLIT = 1.0  # Use all training data
CONFIDENCE_LEVEL = 0.75

print("="*80)
print("Quantile ARIMA FOR ZERO-INFLATED FAAS WORKLOADS")
print("="*80)
print(f"Training days: 1-{TRAINING_DAYS} ({TRAINING_DAYS} days)")
print(f"Prediction days: {TRAINING_DAYS+1}-{TRAINING_DAYS+3} (3 days)")
print(f"Forecast Horizon: {FORECAST_HORIZON} minutes (FIXED)")
print(f"Train/Test Split: {TRAIN_TEST_SPLIT:.0%}")
print(f"Output Directory: {OUTPUT_PATH}")
print("="*80 + "\n")


# ============================================================================
# SECTION 1.4: QUANTILE REGRESSION ARIMA (PRIMARY MODEL)
# ============================================================================

class QuantileARIMA:
    """
    Quantile Regression ARIMA for zero-inflated time series.
    
    Predicts multiple percentiles simultaneously instead of just the mean.
    Outputs: P10, P25, P50, P75, P90, P95 for invocation counts
    
    Perfect for optimization: directly answers "what's the demand at each percentile?"
    
    KEY INSIGHT: Fit on ENTIRE timeseries (including zeros) with AR model instead of ARIMA.
    This preserves spike patterns better than ARIMA which over-smooths.
    """
    
    def __init__(self, forecast_horizon=FORECAST_HORIZON, quantiles=None):
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles or [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.models = {}
        self.timeseries_mean = None
        self.timeseries_std = None
        self.observed_quantiles = None
        self.percentile_values = {}
        self.idle_pattern = None  # Stores zero-inflation pattern by time-of-day
        
    def fit(self, timeseries):
        """Fit ARIMA on entire timeseries - preserves spike patterns"""
        timeseries = np.array(timeseries)
        
        print(f"    Fitting Quantile ARIMA...")
        print(f"    Data: {len(timeseries)} obs, {(timeseries > 0).sum()} non-zero ({(timeseries > 0).sum()/len(timeseries)*100:.1f}%)")
        print(f"    Quantiles: {self.quantiles}")
        
        # Store statistics
        self.timeseries_mean = np.mean(timeseries)
        self.timeseries_std = np.std(timeseries)
        self.timeseries_max = np.max(timeseries)  # Maximum training value for prediction capping
        
        # EXTRACT IDLE PATTERN: Calculate zero-rate for each minute-of-day
        # This captures recurring idle periods (e.g., nights, weekends)
        minutes_per_day = 1440
        n_full_days = len(timeseries) // minutes_per_day
        if n_full_days > 0:
            # Reshape into days
            ts_days = timeseries[:n_full_days * minutes_per_day].reshape(n_full_days, minutes_per_day)
            # Calculate % of days where each minute is zero
            self.idle_pattern = (ts_days == 0).sum(axis=0) / n_full_days  # 0.0 = always active, 1.0 = always idle
            print(f"      Detected {(self.idle_pattern > 0.5).sum()} idle minutes (>50% zero)")
        else:
            self.idle_pattern = None
        
        # Compute observed percentiles from data
        self.observed_quantiles = {}
        for q in self.quantiles:
            self.observed_quantiles[q] = np.percentile(timeseries, q * 100)
            self.percentile_values[q] = np.percentile(timeseries, q * 100)
        
        # Fit single ARIMA on entire series (not just non-zeros)
        # Use AR model (p,0,0) without differencing - preserves spikes better
        print(f"      Fitting AR(2) on full timeseries...", end=" ")
        
        try:
            self.base_model = ARIMA(
                timeseries,
                order=(2, 0, 0),  # AR only, no differencing, no MA
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(method='statespace', low_memory=True)
            print(f"✓ AIC={self.base_model.aic:.1f}")
        except Exception as e:
            print(f"✗ Failed, trying (1,0,0): {str(e)[:30]}")
            try:
                self.base_model = ARIMA(
                    timeseries,
                    order=(1, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(method='statespace', low_memory=True)
                print(f"✓ AIC={self.base_model.aic:.1f}")
            except:
                self.base_model = None
                print(f"✗ Failed completely")
        
        # For each quantile, store the percentile scaling factor
        for q in self.quantiles:
            self.models[q] = {'percentile': np.percentile(timeseries, q * 100)}
        
        return self
    
    def forecast(self, steps=None):
        """Forecast all quantiles - scale base forecast by observed percentiles"""
        if steps is None:
            steps = self.forecast_horizon
        
        if self.base_model is None:
            print("      ✗ Base model not fitted")
            return {q: np.zeros(steps) for q in self.quantiles}
        
        forecasts = {}
        print(f"      Quantile Forecasts:")
        
        try:
            # Get base forecast from AR model
            base_forecast = np.array(self.base_model.forecast(steps=steps))
            base_forecast = np.maximum(base_forecast, 0)
            
            # Get residuals to estimate volatility
            residuals = self.base_model.resid
            residual_std = np.std(residuals)
            
            # find max in the 3 days and set it as max value when predicting! 
            
            
            for q in self.quantiles:
                # Scale base forecast by quantile percentile
                quantile_scale = self.percentile_values[q] / max(self.timeseries_mean, 0.1)
                
                # Forecast with scaling
                pred = base_forecast * quantile_scale
                
                # Add noise proportional to quantile level
                # Higher quantiles get more noise (more spread)
                noise_scale = abs(q - 0.5) / 0.45  # 0.5->1.0x, 0.95->1.1x, 0.10->0.9x
                noise = np.random.normal(0, residual_std * noise_scale * quantile_scale, size=steps)
                pred = pred + noise
                pred = np.maximum(pred, 0)
                # Round to nearest integer (invocation counts)
                pred = np.round(pred)
                
                # CAP PREDICTIONS: Don't exceed maximum value from training data
                if self.timeseries_max is not None:
                    pred = np.minimum(pred, self.timeseries_max)
                
                # APPLY IDLE PATTERN: Set to zero during historically idle minutes
                if self.idle_pattern is not None:
                    for t in range(steps):
                        minute_of_day = t % 1440  # 1440 minutes per day
                        # If this minute was idle >50% of the time historically, zero it out
                        if minute_of_day < len(self.idle_pattern) and self.idle_pattern[minute_of_day] > 0.5:
                            pred[t] = 0
                
                forecasts[q] = pred
                print(f"        P{int(q*100):02d}: min={pred.min():.1f}, max={pred.max():.1f}, mean={pred.mean():.1f}")
        
        except Exception as e:
            print(f"      ✗ Forecast failed: {e}")
            forecasts = {q: np.zeros(steps) for q in self.quantiles}
        
        return forecasts


# ============================================================================
# SECTION 2: DATA LOADING
# ============================================================================

def load_function_data(func_id):
    """Load invocations, duration, and memory data for a function (Days 1-11)"""
    print(f"\n{'─'*80}")
    print(f"Loading Function {func_id}")
    print(f"{'─'*80}")
    
    base_path = SELECTED_FUNCTIONS_PATH
    
    # Load invocations (Days 1-11)
    invoc_path = base_path / f'function_{func_id}_invocations.csv'
    df_invoc = pd.read_csv(invoc_path)
    
    # Filter training days (1-11)
    df_invoc_training = df_invoc[df_invoc['Day'] <= TRAINING_DAYS]
    
    minute_cols = [str(i) for i in range(1, 1441)]
    invocations = []
    for _, row in df_invoc_training.iterrows():
        invocations.extend(row[minute_cols].values)
    invocations = np.array(invocations)
    
    print(f"  Invocations (Days 1-{TRAINING_DAYS}): {len(invocations)} minutes")
    print(f"  Total invocations: {invocations.sum()}")
    print(f"  Non-zero minutes: {(invocations > 0).sum()} ({(invocations > 0).sum()/len(invocations)*100:.1f}%)")
    
    # Load duration percentiles
    dur_path = base_path / f'function_{func_id}_duration_percentiles.csv'
    df_dur = pd.read_csv(dur_path)
    df_dur_training = df_dur[df_dur['Day'] <= TRAINING_DAYS]
    
    # Load memory percentiles
    mem_path = base_path / f'function_{func_id}_memory_percentiles.csv'
    df_mem = pd.read_csv(mem_path)
    df_mem_training = df_mem[df_mem['Day'] <= TRAINING_DAYS]

    # Duration metrics (Days 1-11 averages)
    duration_metrics = {
        'p50_avg': df_dur_training['percentile_Average_50'].mean(),
        'p75_avg': df_dur_training['percentile_Average_75'].mean(),
        'p99_avg': df_dur_training['percentile_Average_99'].mean(),
        'p100_avg': df_dur_training['percentile_Average_100'].mean(),
        'cold_start_latency': df_dur_training['percentile_Average_100'].mean() - df_dur_training['percentile_Average_50'].mean()
    }

    # Memory metrics (Days 1-11 averages)
    memory_metrics = {
        'p50_avg': df_mem_training['AverageAllocatedMb_pct50'].mean(),
        'p75_avg': df_mem_training['AverageAllocatedMb_pct75'].mean(),
        'p99_avg': df_mem_training['AverageAllocatedMb_pct99'].mean(),
        'p100_avg': df_mem_training['AverageAllocatedMb_pct100'].mean(),
        'cold_start_overhead': df_mem_training['AverageAllocatedMb_pct100'].mean() - df_mem_training['AverageAllocatedMb_pct50'].mean()
    }
        
    print(f"  Duration: P50={duration_metrics['p50_avg']:.1f}ms, P100={duration_metrics['p100_avg']:.1f}ms")
    print(f"  Cold start latency: {duration_metrics['cold_start_latency']:.1f}ms")
    print(f"  Memory: P50={memory_metrics['p50_avg']:.1f}MB, P100={memory_metrics['p100_avg']:.1f}MB")
    print(f"  Cold start overhead: {memory_metrics['cold_start_overhead']:.1f}MB")
    print(f"  Training data duration: {TRAINING_DAYS} days = {len(invocations)} minutes")
    
    return {
        'invocations': invocations,
        'duration_metrics': duration_metrics,
        'memory_metrics': memory_metrics,
        'func_id': func_id
    }



def compute_dataset_percentiles(timeseries):
    """
    Compute observed percentiles from historical data.
    Used to VALIDATE your quantile regression predictions.
    """
    return {
        'p10': float(np.percentile(timeseries, 10)),
        'p25': float(np.percentile(timeseries, 25)),
        'p50': float(np.percentile(timeseries, 50)),
        'p75': float(np.percentile(timeseries, 75)),
        'p90': float(np.percentile(timeseries, 90)),
        'p95': float(np.percentile(timeseries, 95)),
        'min': float(timeseries.min()),
        'max': float(timeseries.max()),
        'mean': float(timeseries.mean()),
    }


def validate_quantile_forecasts(forecast_dict, observed_percentiles):
    """
    Compare predicted quantiles against observed percentiles.
    Returns calibration error for each quantile.
    """
    quantiles = sorted(forecast_dict.keys())
    validation = {
        'quantile_levels': quantiles,
        'predictions': {},
        'observations': {},
        'errors': {},
        'calibration': {}
    }
    
    print(f"\n  {'─'*76}")
    print(f"  QUANTILE REGRESSION VALIDATION")
    print(f"  {'─'*76}")
    print(f"  Comparing predicted percentiles vs. observed data percentiles")
    
    for q in quantiles:
        pred_mean = forecast_dict[q].mean()
        obs_key = f'p{int(q*100)}'
        obs_val = observed_percentiles.get(obs_key, pred_mean)
        error = abs(pred_mean - obs_val) / (obs_val + 1e-6)  # Relative error
        
        validation['predictions'][q] = float(pred_mean)
        validation['observations'][q] = float(obs_val)
        validation['errors'][q] = float(error)
        validation['calibration'][q] = "✓" if error < 0.25 else "⚠"
        
        status = "✓" if error < 0.25 else "⚠"
        print(f"  {status} P{int(q*100):2d}: Pred={pred_mean:.1f} vs Obs={obs_val:.1f} (error={error*100:.1f}%)")
    
    validation['overall_mae'] = float(np.mean([abs(validation['predictions'][q] - validation['observations'][q]) for q in quantiles]))
    
    return validation


def memory_to_slots(memory_mb):
    """Return default Azure concurrency slots based on memory size."""
    if memory_mb <= 512/4:
        return 4
    elif memory_mb <= 2048/4:
        return 16
    else:  # memory > 2048 MB
        return 32


def memory_to_allocated_tier(memory_mb):
    """
    Return Azure's actual allocated memory tier based on function's average memory usage.
    
    Azure allocates memory in fixed tiers (blocks), not fractional amounts.
    When a container is wasted, the entire allocated tier is wasted.
    
    Args:
        memory_mb: Average or p75 memory usage
    
    Returns:
        Allocated memory tier in MB (512, 2048, 4096, etc.)
    """
    if memory_mb <= 512/4:
        return 512      # Tier 1: 512 MB
    elif memory_mb <= 2048/4:
        return 2048     # Tier 2: 2 GB
    else:  # memory_mb > 2048
        return 4096     # Tier 3: 4 GB (can go higher for very large functions)


def simulate_day4_invocations_traditional(
    forecast,
    duration_metrics,
    memory_metrics,
    ttl=10,
    data_dir='../data'
):
    """
    Traditional Day 4 Simulation (No Predictive Prewarming)
    
    Simulates container lifecycle WITHOUT prewarming prediction.
    Containers are created on-demand when cold starts occur.
    
    Args:
        forecast: list/array of predicted invocations for day 4 (1440 values)
        duration_metrics: dict with keys 'p50_avg', 'p75_avg', 'p99_avg' (ms)
        memory_metrics: dict with keys 'p50_avg', 'p75_avg', 'p99_avg' (MB)
        ttl: warm container lifetime in minutes
        data_dir: path to data directory
    
    Returns:
        dict with comprehensive metrics:
            - total_gbs: total GB-s cost
            - warm_gbs: GB-s for warm invocations
            - cold_gbs: GB-s for cold starts
            - cold_starts_count: total number of cold starts
            - cold_start_cost: GB-s spent on cold start latency overhead
            - total_latency_s: total execution time (seconds)
            - avg_latency_ms: average latency per invocation
            - max_containers: peak number of containers in use
            - final_containers: containers remaining at end of day
            - avg_container_utilization: average slots used / available slots
    """
    containers = []  # List of dicts: {'expiry': int, 'busy': int}
    
    mem_p50 = memory_metrics['p50_avg']
    mem_p75 = memory_metrics['p75_avg']
    mem_p99 = memory_metrics['p99_avg']
    dur_p50 = duration_metrics['p50_avg']
    dur_p75 = duration_metrics['p75_avg']
    dur_p99 = duration_metrics['p99_avg']
    
    slots_per_instance = memory_to_slots(mem_p75)
    
    # Cost calculations
    warm_cost_per_invoc = (dur_p50 / 1000.0) * (mem_p75 / 1024.0)  # GB-s per invocation
    cold_penalty_per_invoc = ((dur_p99 - dur_p50) / 1000.0) * (mem_p99 / 1024.0)
    
    warm_gbs = 0.0
    cold_gbs = 0.0
    cold_starts_count = 0
    total_latency_ms = 0.0
    total_invocations = 0
    max_containers = 0
    total_container_duration_s = 0.0  # Wall-clock container lifetime for allocation-based cost
    
    for minute, invocs in enumerate(forecast):
        invocs = int(invocs)
        
        # Expire old containers
        containers = [c for c in containers if c['expiry'] > minute]
        
        # Track allocation-based cost: container lifetime (60 seconds per minute)
        total_container_duration_s += len(containers) * 60
        
        # Reset busy count for this minute
        for c in containers:
            c['busy'] = 0
        
        # Route each invocation using Weighted Least Loaded (WLL)
        for _ in range(invocs):
            # Find container with available capacity
            eligible = [c for c in containers if c['busy'] < slots_per_instance]
            
            if eligible:
                # Route to least loaded container
                target = min(eligible, key=lambda c: c['busy'])
                target['busy'] += 1
                warm_gbs += warm_cost_per_invoc
                total_latency_ms += dur_p50
            else:
                # Cold start: create new container
                cold_starts_count += 1
                new_container = {'expiry': minute + ttl, 'busy': 1}
                containers.append(new_container)
                max_containers = max(max_containers, len(containers))
                
                # Cold start cost = cold start latency penalty + execution
                cold_gbs += cold_penalty_per_invoc
                # Invocation itself still costs (with p99 latency)
                warm_gbs += (dur_p99 / 1000.0) * (mem_p99 / 1024.0)
                total_latency_ms += dur_p99
        
        total_invocations += invocs
    
    total_gbs = warm_gbs + cold_gbs
    avg_latency = total_latency_ms / max(total_invocations, 1)
    
    # Container utilization calculation
    total_slots_available = 0
    total_slots_used = 0
    # Approximate by counting peak utilization
    if max_containers > 0:
        avg_utilization = min(1.0, total_invocations / (max_containers * slots_per_instance * FORECAST_HORIZON / FORECAST_HORIZON))
    else:
        avg_utilization = 0.0
    
    # ======== ALLOCATION-BASED COST (Provider Billing Model) ========
    allocated_memory_tier_mb = memory_to_allocated_tier(mem_p75)
    allocated_memory_gb = allocated_memory_tier_mb / 1024.0
    total_allocation_gbs = allocated_memory_gb * total_container_duration_s
    
    AZURE_GB_S_PRICE = 0.000016  # USD per GB-second
    provider_cost_usd = total_allocation_gbs * AZURE_GB_S_PRICE
    
    # ======== BUSINESS COST (Cold Start Latency Penalty) ========
    latency_penalty_ms = dur_p99 - dur_p50  # Execution-based latency overhead
    BUSINESS_VALUE_NORMAL = 0.01    # USD per ms latency penalty (normal SLO)
    BUSINESS_VALUE_CRITICAL = 0.10  # USD per ms latency penalty (critical SLO)
    
    business_cost_normal_usd = cold_starts_count * latency_penalty_ms * BUSINESS_VALUE_NORMAL
    business_cost_critical_usd = cold_starts_count * latency_penalty_ms * BUSINESS_VALUE_CRITICAL
    
    # ======== TOTAL COST BREAKDOWN ========
    total_cost_normal_usd = provider_cost_usd + business_cost_normal_usd
    total_cost_critical_usd = provider_cost_usd + business_cost_critical_usd
    
    return {
        'strategy': 'traditional',
        'total_gbs': total_gbs,
        'warm_gbs': warm_gbs,
        'cold_gbs': cold_gbs,
        'cold_start_cost': cold_gbs,
        'cold_starts_count': cold_starts_count,
        'total_latency_s': total_latency_ms / 1000.0,
        'total_invocations': total_invocations,
        'avg_latency_ms': avg_latency,
        'max_containers': max_containers,
        'final_containers': len(containers),
        'avg_container_utilization': avg_utilization,
        'ttl': ttl,
        # Allocation-based cost (provider billing model)
        'total_container_duration_s': total_container_duration_s,
        'total_allocation_gbs': total_allocation_gbs,
        'provider_cost_usd': provider_cost_usd,
        # Business cost (execution latency model)
        'business_cost_normal_usd': business_cost_normal_usd,
        'business_cost_critical_usd': business_cost_critical_usd,
        # Total cost breakdown
        'total_cost_normal_usd': total_cost_normal_usd,
        'total_cost_critical_usd': total_cost_critical_usd
    }

def simulate_day4_invocations_predictive(
    forecast_arima,
    actual_day4,
    duration_metrics=None,
    memory_metrics=None,
    ttl=10,
    data_dir='../data',
    func_id=None
):
    """
    Predictive Day 4 Simulation (With Predictive Prewarming)
    
    Uses ARIMA forecast to predictively warm containers BEFORE actual demand arrives.
    Runs simulation against ACTUAL Day 4 data to measure real performance.
    Tracks which containers are prewarmed vs. on-demand.
    
    Args:
        forecast_arima: list/array of ARIMA-predicted invocations for day 4 (1440 values)
        actual_day4: REQUIRED - actual day 4 invocation data for realistic simulation (1440 values)
        duration_metrics: dict with keys 'p50_avg', 'p75_avg', 'p99_avg' (ms)
        memory_metrics: dict with keys 'p50_avg', 'p75_avg', 'p99_avg' (MB)
        ttl: warm container lifetime in minutes
        data_dir: path to data directory
        func_id: function ID (for logging/debugging)
    
    Returns:
        dict with comprehensive metrics:
            - total_gbs: total GB-s cost (including wasted prewarming)
            - warm_gbs: GB-s for warm invocations
            - cold_gbs: GB-s for cold starts (unavoidable)
            - prewarm_gbs: GB-s wasted on unused prewarmed containers
            - prewarm_containers_count: number of predictively prewarmed containers
            - prewarm_memory_wasted_mb: MB of memory in unused prewarmed containers
            - cold_starts_count: cold starts despite prewarming
            - cold_starts_avoided: cold starts prevented by prewarming
            - total_latency_s: total execution time
            - avg_latency_ms: average latency per invocation
            - max_containers: peak number of containers
            - final_containers: containers remaining at end
            - prewarming_efficiency: (cold starts avoided / containers prewarmed)
    """
    containers = []  # List of dicts: {'expiry': int, 'busy': int, 'prewarmed': bool}
    
    mem_p50 = memory_metrics['p50_avg']
    mem_p75 = memory_metrics['p75_avg']
    mem_p99 = memory_metrics['p99_avg']
    dur_p50 = duration_metrics['p50_avg']
    dur_p75 = duration_metrics['p75_avg']
    dur_p99 = duration_metrics['p99_avg']
    
    slots_per_instance = memory_to_slots(mem_p75)
    
    # Cost calculations
    warm_cost_per_invoc = (dur_p50 / 1000.0) * (mem_p75 / 1024.0)
    cold_penalty_per_invoc = ((dur_p99 - dur_p50) / 1000.0) * (mem_p99 / 1024.0)
    
    warm_gbs = 0.0
    cold_gbs = 0.0
    prewarm_gbs = 0.0
    cold_starts_count = 0
    total_latency_ms = 0.0
    total_invocations = 0
    max_containers = 0
    prewarm_containers_deployed = 0
    prewarm_containers_used = 0
    total_container_duration_s = 0.0  # Wall-clock container lifetime for allocation-based cost
    
    forecast_arima = np.array(forecast_arima)
    forecast_arima = np.maximum(forecast_arima, 0)  # Ensure non-negative
    
    actual_day4 = np.array(actual_day4)
    actual_day4 = np.maximum(actual_day4, 0)  # Ensure non-negative
    
    # Iterate over ACTUAL Day 4 data, using FORECAST as predictive prewarming guide
    for minute, actual_invocs in enumerate(actual_day4):
        actual_invocs = int(actual_invocs)
        
        # Get forecast for this minute (if available)
        forecast_invocs = int(forecast_arima[minute]) if minute < len(forecast_arima) else 0
        
        # 1. Expire old containers
        containers = [c for c in containers if c['expiry'] > minute]
        
        # Track allocation-based cost: container lifetime (60 seconds per minute)
        total_container_duration_s += len(containers) * 60
        
        # 2. Prewarming decision: Use FORECAST to predict and create containers
        # Look ahead in FORECAST 1-2 minutes to warm containers NOW
        lookahead_minutes = min(2, len(forecast_arima) - minute)
        future_forecast_demand = sum(forecast_arima[minute:minute + lookahead_minutes])
        current_capacity = len(containers) * slots_per_instance

        
        if future_forecast_demand > current_capacity and minute < len(forecast_arima) - 1:
            containers_needed = int(np.ceil((future_forecast_demand - current_capacity) / slots_per_instance))
            
            # Only prewarm if we don't have them already
            for _ in range(containers_needed):
                prewarmed_container = {
                    'expiry': minute + ttl,
                    'busy': 0,
                    'prewarmed': True
                }
                containers.append(prewarmed_container)
                prewarm_containers_deployed += 1
                # Prewarmed containers cost: full slot allocation at warm cost
                prewarm_gbs += slots_per_instance * warm_cost_per_invoc
                max_containers = max(max_containers, len(containers))
        
        # 3. Reset busy count for this minute
        for c in containers:
            c['busy'] = 0
        
        # 4. Route ACTUAL invocations using WLL
        prewarmed_used_this_minute = 0
        for _ in range(actual_invocs):
            eligible = [c for c in containers if c['busy'] < slots_per_instance]
            
            if eligible:
                # Prefer prewarmed containers first
                prewarmed_eligible = [c for c in eligible if c.get('prewarmed', False)]
                if prewarmed_eligible:
                    target = min(prewarmed_eligible, key=lambda c: c['busy'])
                    target['prewarmed'] = False  # Mark as used
                    prewarmed_used_this_minute += 1
                else:
                    target = min(eligible, key=lambda c: c['busy'])
                
                target['busy'] += 1
                warm_gbs += warm_cost_per_invoc
                total_latency_ms += dur_p50
            else:
                # Cold start: no container available despite prewarming
                cold_starts_count += 1
                new_container = {
                    'expiry': minute + ttl,
                    'busy': 1,
                    'prewarmed': False
                }
                containers.append(new_container)
                max_containers = max(max_containers, len(containers))
                
                cold_gbs += cold_penalty_per_invoc
                warm_gbs += (dur_p99 / 1000.0) * (mem_p99 / 1024.0)
                total_latency_ms += dur_p99
        
        prewarm_containers_used += prewarmed_used_this_minute
        total_invocations += actual_invocs
    
    # Calculate wasted prewarming
    unused_prewarmed = prewarm_containers_deployed - prewarm_containers_used
    
    # Calculate wasted memory using Azure's allocated memory tier (not fractional)
    # Each wasted container holds one complete tier allocation (512MB, 2048MB, 4096MB, etc.)
    allocated_memory_tier_mb = memory_to_allocated_tier(mem_p75)
    prewarm_memory_wasted_mb = unused_prewarmed * allocated_memory_tier_mb
    
    # Calculate prewarm_usage_cost: Memory allocated (but unused) × TTL duration
    # In GB-minutes: (wasted memory MB / 1024) × (TTL in minutes)
    # In GB-s: (wasted memory GB) × (TTL in seconds)
    prewarm_usage_cost_gb_minutes = (prewarm_memory_wasted_mb / 1024.0)
    prewarm_usage_cost_gbs = (prewarm_memory_wasted_mb / 1024.0) * (60.0)
    
    total_gbs = warm_gbs + cold_gbs + prewarm_gbs + prewarm_usage_cost_gbs
    avg_latency = total_latency_ms / max(total_invocations, 1)
    
    # Prewarming efficiency
    prewarming_efficiency = (prewarm_containers_deployed - unused_prewarmed) / max(prewarm_containers_deployed, 1)
    
    # ======== ALLOCATION-BASED COST (Provider Billing Model) ========
    allocated_memory_tier_mb = memory_to_allocated_tier(mem_p75)
    allocated_memory_gb = allocated_memory_tier_mb / 1024.0
    total_allocation_gbs = allocated_memory_gb * total_container_duration_s
    
    AZURE_GB_S_PRICE = 0.000016  # USD per GB-second
    provider_cost_usd = total_allocation_gbs * AZURE_GB_S_PRICE
    
    # ======== BUSINESS COST (Cold Start Latency Penalty) ========
    latency_penalty_ms = dur_p99 - dur_p50  # Execution-based latency overhead
    BUSINESS_VALUE_NORMAL = 0.01    # USD per ms latency penalty (normal SLO)
    BUSINESS_VALUE_CRITICAL = 0.10  # USD per ms latency penalty (critical SLO)
    
    business_cost_normal_usd = cold_starts_count * latency_penalty_ms * BUSINESS_VALUE_NORMAL
    business_cost_critical_usd = cold_starts_count * latency_penalty_ms * BUSINESS_VALUE_CRITICAL
    
    # ======== TOTAL COST BREAKDOWN ========
    total_cost_normal_usd = provider_cost_usd + business_cost_normal_usd
    total_cost_critical_usd = provider_cost_usd + business_cost_critical_usd
    
    return {
        'strategy': 'predictive_prewarming',
        'total_gbs': total_gbs,
        'warm_gbs': warm_gbs,
        'cold_gbs': cold_gbs,
        'prewarm_gbs': prewarm_gbs,
        'cold_start_cost': cold_gbs,
        'cold_starts_count': cold_starts_count,
        'cold_starts_avoided': prewarm_containers_deployed - unused_prewarmed,
        'prewarm_containers_count': prewarm_containers_deployed,
        'prewarm_containers_used': prewarm_containers_used,
        'prewarm_containers_wasted': unused_prewarmed,
        'prewarm_memory_wasted_mb': prewarm_memory_wasted_mb,
        'prewarm_usage_cost_gb_minutes': prewarm_usage_cost_gb_minutes,
        'prewarm_usage_cost_gbs': prewarm_usage_cost_gbs,
        'total_latency_s': total_latency_ms / 1000.0,
        'total_invocations': total_invocations,
        'avg_latency_ms': avg_latency,
        'max_containers': max_containers,
        'final_containers': len(containers),
        'prewarming_efficiency': prewarming_efficiency,
        'ttl': ttl,
        # Allocation-based cost (provider billing model)
        'total_container_duration_s': total_container_duration_s,
        'total_allocation_gbs': total_allocation_gbs,
        'provider_cost_usd': provider_cost_usd,
        # Business cost (execution latency model)
        'business_cost_normal_usd': business_cost_normal_usd,
        'business_cost_critical_usd': business_cost_critical_usd,
        # Total cost breakdown
        'total_cost_normal_usd': total_cost_normal_usd,
        'total_cost_critical_usd': total_cost_critical_usd
    }

# ============================================================================
# SECTION 3: STATIONARITY & ACF/PACF (Keep for documentation)
# ============================================================================

def test_stationarity(timeseries, func_id):
    """ADF test for stationarity"""
    print(f"\n  {'─'*76}")
    print(f"  STATIONARITY TEST")
    print(f"  {'─'*76}")
    
    adf_result = adfuller(timeseries, autolag='AIC')
    adf_stat, p_value = adf_result[0], adf_result[1]
    is_stationary = p_value < 0.05
    
    print(f"  ADF Statistic: {adf_stat:.4f}, p-value: {p_value:.4f}")
    print(f"  → {'STATIONARY' if is_stationary else 'NON-STATIONARY'}")
    
    return {
        'adf_statistic': float(adf_stat),
        'p_value': float(p_value),
        'is_stationary': bool(is_stationary),
        'recommended_d': 0 if is_stationary else 1
    }


def analyze_acf_pacf(timeseries, func_id, max_lags=40):
    """ACF/PACF for documentation (not used in Quantile Regression model)"""
    print(f"\n  {'─'*76}")
    print(f"  ACF/PACF ANALYSIS (for reference)")
    print(f"  {'─'*76}")
    
    acf_values = acf(timeseries, nlags=max_lags, fft=True)
    pacf_values = pacf(timeseries, nlags=max_lags)
    
    n = len(timeseries)
    threshold = 1.96 / np.sqrt(n)
    
    sig_acf = np.where(np.abs(acf_values[1:]) > threshold)[0] + 1
    sig_pacf = np.where(np.abs(pacf_values[1:]) > threshold)[0] + 1
    
    print(f"  Significant ACF lags: {sig_acf[:5].tolist()}")
    print(f"  Significant PACF lags: {sig_pacf[:5].tolist()}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(timeseries, lags=max_lags, ax=axes[0], alpha=0.05)
    plot_pacf(timeseries, lags=max_lags, ax=axes[1], alpha=0.05)
    axes[0].set_title(f'Function {func_id}: ACF')
    axes[1].set_title(f'Function {func_id}: PACF')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'function_{func_id}_acf_pacf.png', dpi=200)
    plt.close()
    
    return {
        'suggested_p': min(len(sig_pacf), 3) if len(sig_pacf) > 0 else 1,
        'suggested_q': min(len(sig_acf), 3) if len(sig_acf) > 0 else 1,
        'has_seasonality': False
    }


# ============================================================================
# SECTION 4: MODEL FITTING
# ============================================================================

def fit_arima_models(timeseries, func_id, candidate_orders=None, seasonal_order=None):
    """
    Fit QUANTILE ARIMA model.
    
    This replaces Quantile ARIMA with Quantile Regression approach.
    Directly predicts invocation percentiles: P10, P25, P50, P75, P90, P95
    Perfect for optimization decisions.
    """
    print(f"\n  {'─'*76}")
    print(f"  QUANTILE ARIMA MODEL FITTING")
    print(f"  {'─'*76}")
    
    try:
        model = QuantileARIMA(
            forecast_horizon=FORECAST_HORIZON,
            quantiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        )
        
        model.fit(timeseries)
        
        return {
            'best_model': model,
            'best_order': 'QuantileARIMA(2,1,1)',
            'aic': -1,  # Not applicable for quantile regression
            'bic': -1,
            'model_type': 'QUANTILE_ARIMA',
            'all_results': pd.DataFrame()
        }
    
    except Exception as e:
        print(f"    ✗ Complete failure: {e}")
        
        # Fallback to zeros
        class ZeroModel:
            def forecast(self, steps):
                return {q: np.zeros(steps) for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]}
        
        return {
            'best_model': ZeroModel(),
            'best_order': 'ZERO',
            'aic': np.inf,
            'bic': np.inf,
            'model_type': 'ZERO',
            'all_results': pd.DataFrame()
        }


# ============================================================================
# SECTION 5: FORECASTING AND EVALUATION
# ============================================================================

def forecast_and_evaluate(model, timeseries, train_size, forecast_horizon, func_id, model_type='Quantile'):
    """Forecast and evaluate with Quantile ARIMA support"""
    print(f"\n  {'─'*76}")
    print(f"  QUANTILE FORECASTING & VALIDATION")
    print(f"  {'─'*76}")
    
    split_idx = int(len(timeseries) * train_size)
    train = timeseries[:split_idx]
    test = timeseries[split_idx:]
    
    print(f"  Train: {len(train)} min, Test: {len(test)} min (available)")
    print(f"  Forecast: {forecast_horizon} min (requested)")
    
    # Generate quantile forecast
    try:
        forecast_dict = model.forecast(steps=forecast_horizon)
        
        # Handle both dict (Quantile ARIMA) and array (fallback) outputs
        if isinstance(forecast_dict, dict):
            median_forecast = np.array(forecast_dict[0.50])
            has_quantiles = True
        else:
            median_forecast = np.array(forecast_dict)
            has_quantiles = False
        
        if len(median_forecast) != forecast_horizon:
            median_forecast = np.resize(median_forecast, forecast_horizon)
    except Exception as e:
        print(f"    ✗ Forecast failed: {e}")
        median_forecast = np.zeros(forecast_horizon)
        has_quantiles = False
    
    # Evaluate on available test data
    eval_length = min(len(test), forecast_horizon)
    actual = test[:eval_length].values if hasattr(test, 'values') else test[:eval_length]
    forecast_eval = median_forecast[:eval_length]
    
    print(f"  Evaluation: {eval_length} min")
    
    if len(actual) == 0:
        return {
            'forecast': median_forecast,
            'forecast_quantiles': forecast_dict if has_quantiles else {},
            'actual': np.array([]),
            'mae': 0, 'rmse': 0, 'mape': 0, 'mase': 0,
            'has_quantiles': has_quantiles
        }
    
    # Metrics
    mae = np.mean(np.abs(actual - forecast_eval))
    rmse = np.sqrt(np.mean((actual - forecast_eval) ** 2))
    
    # MAPE on non-zeros only
    nonzero_mask = actual > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[nonzero_mask] - forecast_eval[nonzero_mask]) / actual[nonzero_mask])) * 100
    else:
        mape = 0.0
    
    naive_mae = np.mean(np.abs(pd.Series(train).diff().dropna()))
    mase = mae / naive_mae if naive_mae > 0 else np.inf
    
    # Quantile-specific output
    if has_quantiles:
        observed_pct = compute_dataset_percentiles(actual)
        validation = validate_quantile_forecasts(forecast_dict, observed_pct)
        
        print(f"\n  QUANTILE CALIBRATION:")
        print(f"  {'-'*60}")
        for q in sorted(validation['quantile_levels']):
            pred = validation['predictions'][q]
            obs = validation['observations'][q]
            err = validation['errors'][q]
            status = validation['calibration'][q]
            print(f"    P{int(q*100):2d}: Predicted={pred:7.1f} | Observed={obs:7.1f} | Error%={err:6.1f}% {status}")
        
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%, MASE: {mase:.2f}")
    else:
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%, MASE: {mase:.2f}")
    
    return {
        'forecast': median_forecast,
        'forecast_quantiles': forecast_dict if has_quantiles else {},
        'actual': actual,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mase': mase,
        'has_quantiles': has_quantiles
    }


# ============================================================================
# SECTION 6: PLOTTING
# ============================================================================

def save_forecast_plot(timeseries, forecast, train_size, func_id, forecast_quantiles=None):
    """Plot last 3 days + forecast with confidence bands. Save all plots and CSV to function-specific directory"""
    split_idx = int(len(timeseries) * train_size)
    
    # Create function-specific directory
    func_dir = OUTPUT_PATH / f'function_{func_id}'
    func_dir.mkdir(parents=True, exist_ok=True)
    
    # Last 3 days for context
    context_start = max(0, split_idx - 11*FORECAST_HORIZON)
    context = timeseries[context_start:split_idx]
    
    # ========== MAIN PLOT: Confidence bands ==========
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Historical
    ax.plot(range(len(context)), context, label='First 11 Days', color='blue', linewidth=1)
    
    forecast_x = range(len(context), len(context) + len(forecast))
    
    # Plot quantile confidence bands if available
    if forecast_quantiles and isinstance(forecast_quantiles, dict):
        # P95/P90 band (top)
        p95 = forecast_quantiles.get(0.95, np.zeros_like(forecast))
        p90 = forecast_quantiles.get(0.90, np.zeros_like(forecast))
        p75 = forecast_quantiles.get(0.75, np.zeros_like(forecast))
        p50 = forecast_quantiles.get(0.50, np.zeros_like(forecast))
        p25 = forecast_quantiles.get(0.25, np.zeros_like(forecast))
        p10 = forecast_quantiles.get(0.10, np.zeros_like(forecast))
        
        # Confidence bands
        ax.fill_between(forecast_x, p10, p95, alpha=0.2, color='red', label='P10-P95 (90% band)')
        ax.fill_between(forecast_x, p25, p75, alpha=0.3, color='orange', label='P25-P75 (50% band)')
        
        # Individual quantile lines
        ax.plot(forecast_x, p95, color='red', linewidth=1.5, linestyle='--', label='P95 (aggressive)')
        ax.plot(forecast_x, p75, color='orange', linewidth=1.5, linestyle='--', label='P75 (conservative)')
        ax.plot(forecast_x, p50, color='green', linewidth=2, linestyle='--', label='P50 (median)')
    
    # Plot median forecast fallback
    ax.plot(forecast_x, forecast, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    
    ax.axvline(x=len(context), color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Invocations/min')
    ax.set_title(f'Function {func_id}: Quantile ARIMA Forecast')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(func_dir / f'function_{func_id}_forecast.png', dpi=200)
    plt.close()

    # ========== INDIVIDUAL PERCENTILE PLOTS ==========
    percentile_configs = [
        (0.10, 'P10', 'blue'),
        (0.25, 'P25', 'cyan'),
        (0.50, 'P50', 'green'),
        (0.75, 'P75', 'orange'),
        (0.90, 'P90', 'purple'),
        (0.95, 'P95', 'red')
    ]
    
    if forecast_quantiles and isinstance(forecast_quantiles, dict):
        for q, q_label, color in percentile_configs:
            if q in forecast_quantiles:
                fig, ax = plt.subplots(figsize=(15, 6))
                
                pred = forecast_quantiles[q]
                forecast_x = range(len(context), len(context) + len(pred))
                
                # Historical
                ax.plot(range(len(context)), context, label='First 11 Days (Actual)', color='blue', linewidth=1.5, alpha=0.7)
                
                # Forecast
                ax.plot(forecast_x, pred, color=color, linewidth=2.5, linestyle='--', 
                       label=f'{q_label} Forecast ({int(q*100)}th Percentile)', marker='o', markersize=2, alpha=0.8)
                
                ax.fill_between(forecast_x, 0, pred, alpha=0.15, color=color)
                ax.axvline(x=len(context), color='black', linestyle=':', alpha=0.5, linewidth=1.5)
                ax.set_xlabel('Minutes', fontsize=11)
                ax.set_ylabel('Invocations/min', fontsize=11)
                ax.set_title(f'Function {func_id}: {q_label} Quantile Forecast ({int(q*100)}th Percentile)', 
                           fontsize=12, fontweight='bold')
                ax.legend(loc='upper right', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(func_dir / f'function_{func_id}_forecast_{q_label.lower()}.png', dpi=200)
                plt.close()

    # ========== SAVE CSV WITH ALL PERCENTILES ==========
    csv_data = {'minute': range(len(forecast)), 'P50_median': forecast}
    
    if forecast_quantiles and isinstance(forecast_quantiles, dict):
        for q in sorted(forecast_quantiles.keys()):
            csv_data[f'P{int(q*100):02d}'] = forecast_quantiles[q]
    
    pd.DataFrame(csv_data).to_csv(func_dir / f'function_{func_id}_forecast.csv', index=False)


# ============================================================================
# SECTION 7: MAIN PIPELINE
# ============================================================================

def analyze_single_function_real_data(csv_path, func_id, data_dir=SELECTED_FUNCTIONS_PATH):
    """
    Complete analysis for one function using REAL Day 4 data
    
    Compares two strategies with actual invocations:
    1. Traditional: No prewarming prediction, containers on-demand
    2. Predictive: Uses ARIMA forecast to predictively warm containers
    
    Args:
        csv_path: path to function invocations CSV
        func_id: function ID
        data_dir: path to data directory
    
    Returns:
        dict with simulation results comparing both strategies
    """
    print("\n" + "="*80)
    print(f"FUNCTION {func_id} - REAL DAY 4 ANALYSIS")
    print("="*80)
    
    # Load data (days 1-11 for model training)
    data = load_function_data(func_id)
    timeseries = pd.Series(data['invocations'], name='invocations')
    
    # Load ACTUAL prediction data (Days 10-12)
    try:
        string_path = f"function_{func_id}_invocations_days_10_12.csv"
        sim_file = Path(data_dir) / string_path
        df_sim = pd.read_csv(sim_file)
        
        if df_sim.empty:
            print(f"  ⚠ No prediction data found for function {func_id}")
            print(f"  ⚠ Skipping prediction analysis")
            return None
        
        # Extract minute-by-minute invocations (columns 1-1440)
        minute_cols = [str(i) for i in range(1, 1441)]
        # Combine all days (12, 13, 14)
        actual_sim = []
        for _, row in df_sim.iterrows():
            actual_sim.extend(row[minute_cols].values)
        actual_sim = np.array(actual_sim, dtype=float)
        
        print(f"  ✓ Loaded prediction data (Days 10-12): {len(actual_sim)} minutes")
        print(f"    Total invocations: {actual_sim.sum():.0f}")
        print(f"    Non-zero minutes: {(actual_sim > 0).sum()} ({(actual_sim > 0).sum()/len(actual_sim)*100:.1f}%)")
        
    except Exception as e:
        print(f"  ⚠ Error loading prediction data: {e}")
        print(f"  ⚠ Skipping prediction analysis for function {func_id}")
        return None
    
    # Tests
    print(f"\n  {'─'*76}")
    print(f"  MODEL FITTING (Training days 1-{TRAINING_DAYS})")
    print(f"  {'─'*76}")
    
    stationarity = test_stationarity(timeseries, func_id)
    acf_pacf = analyze_acf_pacf(timeseries, func_id)
    
    # Fit Quantile ARIMA
    model_result = fit_arima_models(timeseries, func_id)
    
    # Forecast days 10-12 (3 days)
    forecast_result = forecast_and_evaluate(
        model_result['best_model'],
        timeseries,
        1.0,  # Use all training days (1-11) for training
        FORECAST_HORIZON,
        func_id,
        model_result['model_type']
    )
    
    # Get forecast for days 10-12
    if 'forecast_quantiles' in forecast_result and forecast_result['forecast_quantiles']:
        forecast_dict = forecast_result['forecast_quantiles']
    else:
        forecast_dict = {0.50: forecast_result['forecast']}
    
    # Get P50 (median) forecast
    forecast_p50 = forecast_dict.get(0.50, forecast_result['forecast'])
    
    print(f"\n  {'─'*76}")
    print(f"  SIMULATION WITH REAL PREDICTION DATA (Days 10-12)")
    print(f"  {'─'*76}")
    
    # ========== STRATEGY 1: TRADITIONAL (No Predictive Prewarming) ==========
    print(f"\n  STRATEGY 1: Traditional (No Predictive Prewarming)")
    print(f"  {'-'*76}")
    
    traditional_result = simulate_day4_invocations_traditional(
        actual_sim,  # Use actual days 10-12 data
        data['duration_metrics'],
        data['memory_metrics'],
        ttl=10,
        data_dir=data_dir
    )
    
    print(f"    Total GB-s: {traditional_result['total_gbs']:.4f}")
    print(f"    Warm GB-s: {traditional_result['warm_gbs']:.4f}")
    print(f"    Cold start GB-s: {traditional_result['cold_gbs']:.4f}")
    print(f"    Cold starts: {traditional_result['cold_starts_count']}")
    print(f"    Max containers: {traditional_result['max_containers']}")
    print(f"    Total latency: {traditional_result['total_latency_s']:.2f}s")
    print(f"    Avg latency/invocation: {traditional_result['avg_latency_ms']:.2f}ms")
    
    # ========== STRATEGY 2: PREDICTIVE PREWARMING ==========
    print(f"\n  STRATEGY 2: Predictive Prewarming (Using ARIMA P50 Forecast)")
    print(f"  {'-'*76}")
    
    predictive_result = simulate_day4_invocations_predictive(
        forecast_p50,  # ARIMA forecast
        actual_sim,    # Actual days 10-12 data
        data['duration_metrics'],
        data['memory_metrics'],
        ttl=10,
        data_dir=data_dir,
        func_id=func_id
    )
    
    print(f"    Total GB-s: {predictive_result['total_gbs']:.4f}")
    print(f"    Warm GB-s: {predictive_result['warm_gbs']:.4f}")
    print(f"    Cold start GB-s: {predictive_result['cold_gbs']:.4f}")
    print(f"    Prewarming cost (wasted): {predictive_result['prewarm_gbs']:.4f}")
    print(f"    Prewarmed containers: {predictive_result['prewarm_containers_count']}")
    print(f"    Prewarmed containers used: {predictive_result['prewarm_containers_used']}")
    print(f"    Prewarmed containers wasted: {predictive_result['prewarm_containers_wasted']}")
    print(f"    Prewarming efficiency: {predictive_result['prewarming_efficiency']*100:.1f}%")
    print(f"    Memory wasted in unused prewarmed containers: {predictive_result['prewarm_memory_wasted_mb']:.0f} MB")
    print(f"    Cold starts (residual): {predictive_result['cold_starts_count']}")
    print(f"    Cold starts avoided: {predictive_result['cold_starts_avoided']}")
    print(f"    Max containers: {predictive_result['max_containers']}")
    print(f"    Total latency: {predictive_result['total_latency_s']:.2f}s")
    print(f"    Avg latency/invocation: {predictive_result['avg_latency_ms']:.2f}ms")
    
    # ========== COMPARISON & SAVINGS ==========
    print(f"\n  {'─'*76}")
    print(f"  STRATEGY COMPARISON")
    print(f"  {'─'*76}")
    
    savings_gbs = traditional_result['total_gbs'] - predictive_result['total_gbs']
    savings_pct = (savings_gbs / traditional_result['total_gbs']) * 100 if traditional_result['total_gbs'] > 0 else 0
    
    print(f"  Traditional Total: {traditional_result['total_gbs']:.4f} GB-s")
    print(f"  Predictive Total:  {predictive_result['total_gbs']:.4f} GB-s")
    print(f"  Savings: {savings_gbs:.4f} GB-s ({savings_pct:+.2f}%)")
    
    latency_savings_s = traditional_result['total_latency_s'] - predictive_result['total_latency_s']
    latency_savings_pct = (latency_savings_s / traditional_result['total_latency_s']) * 100 if traditional_result['total_latency_s'] > 0 else 0
    
    print(f"\n  Traditional Latency: {traditional_result['total_latency_s']:.2f}s")
    print(f"  Predictive Latency:  {predictive_result['total_latency_s']:.2f}s")
    print(f"  Latency Savings: {latency_savings_s:.2f}s ({latency_savings_pct:+.2f}%)")
    
    cold_starts_avoided = traditional_result['cold_starts_count'] - predictive_result['cold_starts_count']
    print(f"\n  Cold Starts Avoided: {cold_starts_avoided} ({traditional_result['cold_starts_count']} → {predictive_result['cold_starts_count']})")
    
    # ========== TEST MULTIPLE PERCENTILES ==========
    print(f"\n  {'─'*76}")
    print(f"  TESTING MULTIPLE FORECAST PERCENTILES")
    print(f"  {'─'*76}")
    
    percentile_results = {}
    percentiles_to_test = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    
    for p in percentiles_to_test:
        if p in forecast_dict:
            forecast_p = forecast_dict[p]
            
            result_p = simulate_day4_invocations_predictive(
                forecast_p,
                actual_sim,
                data['duration_metrics'],
                data['memory_metrics'],
                ttl=10,
                data_dir=data_dir,
                func_id=func_id
            )
            
            savings_p = traditional_result['total_gbs'] - result_p['total_gbs']
            savings_pct_p = (savings_p / traditional_result['total_gbs']) * 100 if traditional_result['total_gbs'] > 0 else 0
            
            percentile_results[p] = {
                'percentile': int(p * 100),
                'forecast_min': forecast_p.min(),
                'forecast_max': forecast_p.max(),
                'forecast_mean': forecast_p.mean(),
                'total_gbs': result_p['total_gbs'],
                'savings_gbs': savings_p,
                'savings_pct': savings_pct_p,
                'prewarm_gbs': result_p['prewarm_gbs'],
                'cold_starts': result_p['cold_starts_count'],
                'prewarm_efficiency': result_p['prewarming_efficiency'] * 100,
                'total_latency_s': result_p['total_latency_s']
            }
            
            status = "✓" if savings_p > 0 else "✗"
            print(f"    {status} P{int(p*100):02d}: "
                  f"GB-s={result_p['total_gbs']:8.4f} | "
                  f"Savings={savings_p:7.4f} ({savings_pct_p:+6.2f}%) | "
                  f"Cold starts={result_p['cold_starts_count']:3d} | "
                  f"Prewarmed={result_p['prewarm_containers_count']:2d} | "
                  f"Efficiency={result_p['prewarming_efficiency']*100:5.1f}%")
    
    # ========== PLOTTING: Real Day 4 vs ARIMA Forecasts ==========
    print(f"\n  {'─'*76}")
    print(f"  GENERATING FORECAST COMPARISON PLOT")
    print(f"  {'─'*76}")
    
    func_dir = OUTPUT_PATH / f'function_{func_id}'
    func_dir.mkdir(parents=True, exist_ok=True)
    
    # Main plot: Real Day 4 vs All Percentile Forecasts
    fig, ax = plt.subplots(figsize=(18, 8))
    
    minutes = np.arange(len(actual_sim))
    
    # Plot actual Day 4 data (bold)
    ax.plot(minutes, actual_sim, color='black', linewidth=2.5, label='Actual Day 4', alpha=0.8, zorder=10)
    ax.fill_between(minutes, 0, actual_sim, color='gray', alpha=0.15)
    
    # Define percentile colors and styles
    percentile_styles = {
        0.10: {'color': '#1f77b4', 'linestyle': ':', 'alpha': 0.5, 'linewidth': 1.5, 'label': 'P10 (Optimistic)'},
        0.25: {'color': '#aec7e8', 'linestyle': '--', 'alpha': 0.6, 'linewidth': 1.5, 'label': 'P25'},
        0.50: {'color': '#2ca02c', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 2.0, 'label': 'P50 (Median)'},
        0.75: {'color': '#ff7f0e', 'linestyle': '--', 'alpha': 0.7, 'linewidth': 1.8, 'label': 'P75'},
        0.90: {'color': '#d62728', 'linestyle': '-.', 'alpha': 0.6, 'linewidth': 1.5, 'label': 'P90'},
        0.95: {'color': '#8b0000', 'linestyle': ':', 'alpha': 0.7, 'linewidth': 2.0, 'label': 'P95 (Conservative)'},
    }
    
    # Plot each percentile forecast
    for p in sorted(forecast_dict.keys()):
        if p in percentile_styles:
            forecast_p = forecast_dict[p]
            style = percentile_styles[p]
            ax.plot(minutes, forecast_p, 
                   color=style['color'], 
                   linestyle=style['linestyle'], 
                   linewidth=style['linewidth'],
                   alpha=style['alpha'],
                   label=style['label'],
                   marker='o',
                   markersize=1)
    
    # Confidence bands
    p10_forecast = forecast_dict.get(0.10, np.zeros_like(actual_sim))
    p95_forecast = forecast_dict.get(0.95, np.zeros_like(actual_sim))
    p25_forecast = forecast_dict.get(0.25, np.zeros_like(actual_sim))
    p75_forecast = forecast_dict.get(0.75, np.zeros_like(actual_sim))
    
    ax.fill_between(minutes, p10_forecast, p95_forecast, alpha=0.1, color='purple', label='P10-P95 Band (90% range)')
    ax.fill_between(minutes, p25_forecast, p75_forecast, alpha=0.15, color='orange', label='P25-P75 Band (50% range)')
    
    ax.set_xlabel('Minutes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Invocations/min', fontsize=12, fontweight='bold')
    ax.set_title(f'Function {func_id}: Real Day 4 Invocations vs ARIMA Forecast Percentiles', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(func_dir / f'function_{func_id}_real_day4_vs_forecast.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # # Individual percentile plots with actual data overlay
    # percentile_configs = [
    #     (0.10, 'P10', '#1f77b4', 'Optimistic'),
    #     (0.25, 'P25', '#aec7e8', ''),
    #     (0.50, 'P50', '#2ca02c', 'Median'),
    #     (0.75, 'P75', '#ff7f0e', ''),
    #     (0.90, 'P90', '#d62728', ''),
    #     (0.95, 'P95', '#8b0000', 'Conservative'),
    # ]
    
    # for p, p_label, color, desc in percentile_configs:
    #     if p in forecast_dict:
    #         fig, ax = plt.subplots(figsize=(16, 7))
            
    #         forecast_p = forecast_dict[p]
            
    #         # Actual data
    #         ax.plot(minutes, actual_sim, color='black', linewidth=2.5, 
    #                label='Actual Day 4', alpha=0.8, zorder=10)
    #         ax.fill_between(minutes, 0, actual_sim, color='gray', alpha=0.1)
            
    #         # Forecast
    #         ax.plot(minutes, forecast_p, color=color, linewidth=2.5, 
    #                label=f'{p_label} Forecast ({desc})', alpha=0.85, 
    #                linestyle='--', marker='s', markersize=2)
    #         ax.fill_between(minutes, 0, forecast_p, color=color, alpha=0.1)
            
    #         # Error region (difference)
    #         ax.fill_between(minutes, actual_sim, forecast_p, 
    #                        where=(forecast_p >= actual_sim),
    #                        color='red', alpha=0.2, label='Over-prediction')
    #         ax.fill_between(minutes, actual_sim, forecast_p,
    #                        where=(forecast_p < actual_sim),
    #                        color='blue', alpha=0.2, label='Under-prediction')
            
    #         ax.set_xlabel('Minutes', fontsize=11)
    #         ax.set_ylabel('Invocations/min', fontsize=11)
    #         ax.set_title(f'Function {func_id}: {p_label} Forecast vs Actual Day 4 ({int(p*100)}th Percentile{" - " + desc if desc else ""})',
    #                     fontsize=12, fontweight='bold')
    #         ax.legend(loc='upper right', fontsize=10)
    #         ax.grid(True, alpha=0.3)
            
    #         plt.tight_layout()
    #         plt.savefig(func_dir / f'function_{func_id}_real_day4_{p_label.lower()}_comparison.png', dpi=200, bbox_inches='tight')
    #         plt.close()
    
    # Summary statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Error by percentile
    errors_by_percentile = {}
    for p in sorted(forecast_dict.keys()):
        forecast_p = forecast_dict[p]
        mae = np.mean(np.abs(actual_sim - forecast_p))
        errors_by_percentile[int(p*100)] = mae
    
    ax = axes[0, 0]
    percentiles_list = list(errors_by_percentile.keys())
    errors_list = list(errors_by_percentile.values())
    ax.bar(percentiles_list, errors_list, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Percentile', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=11)
    ax.set_title('Forecast Error by Percentile', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Forecast range vs actual
    ax = axes[0, 1]
    p10 = forecast_dict.get(0.10, np.zeros_like(actual_sim))
    p50 = forecast_dict.get(0.50, np.zeros_like(actual_sim))
    p95 = forecast_dict.get(0.95, np.zeros_like(actual_sim))
    
    ax.scatter(actual_sim, p50, alpha=0.4, s=20, label='P50 vs Actual', color='green')
    ax.plot([0, actual_sim.max()], [0, actual_sim.max()], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Invocations/min', fontsize=11)
    ax.set_ylabel('Predicted Invocations/min (P50)', fontsize=11)
    ax.set_title('Actual vs P50 Predicted (Scatter)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative invocations
    ax = axes[1, 0]
    cumsum_actual = np.cumsum(actual_sim)
    for p in sorted(forecast_dict.keys()):
        forecast_p = forecast_dict[p]
        cumsum_forecast = np.cumsum(forecast_p)
        ax.plot(minutes, cumsum_forecast, alpha=0.6, label=f'P{int(p*100)}')
    
    ax.plot(minutes, cumsum_actual, color='black', linewidth=2.5, label='Actual', zorder=10)
    ax.set_xlabel('Minutes', fontsize=11)
    ax.set_ylabel('Cumulative Invocations', fontsize=11)
    ax.set_title('Cumulative Invocations: Actual vs Forecasts', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Hourly aggregation comparison
    ax = axes[1, 1]
    hourly_actual = []
    hourly_forecasts = {p: [] for p in sorted(forecast_dict.keys())}
    
    for hour in range(24):
        start_min = hour * 60
        end_min = start_min + 60
        hourly_actual.append(actual_sim[start_min:end_min].sum())
        for p in sorted(forecast_dict.keys()):
            hourly_forecasts[p].append(forecast_dict[p][start_min:end_min].sum())
    
    hours = np.arange(24)
    ax.bar(hours - 0.2, hourly_actual, width=0.4, label='Actual', color='black', alpha=0.6)
    ax.plot(hours, hourly_forecasts[0.50], color='green', linewidth=2, marker='o', label='P50', alpha=0.8)
    ax.fill_between(hours, hourly_forecasts[0.10], hourly_forecasts[0.95], alpha=0.15, color='purple', label='P10-P95')
    
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Invocations', fontsize=11)
    ax.set_title('Hourly Aggregation: Actual vs Forecasts', fontsize=12, fontweight='bold')
    ax.set_xticks(hours)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(func_dir / f'function_{func_id}_real_day4_summary_stats.png', dpi=200, bbox_inches='tight')
    plt.close()

    # save_forecast_plot for days 1-3 real forecast, and predicted for 4
    save_forecast_plot(timeseries, forecast_result['forecast'], 1.0, func_id, 
                      forecast_result.get('forecast_quantiles', None))
    
    print(f"    ✓ Saved: function_{func_id}_real_day4_vs_forecast.png")
    print(f"    ✓ Saved: function_{func_id}_real_day4_*_comparison.png (6 percentiles)")
    print(f"    ✓ Saved: function_{func_id}_real_day4_summary_stats.png")
    
    # Build strategy results for all percentiles
    strategy_results = {
        'strategy_traditional': traditional_result,
        # 'strategy_predictive_p50': predictive_result,
    }
    
    # Add full strategy data for all tested percentiles
    for p in percentiles_to_test:
        if p in forecast_dict:
            forecast_p = forecast_dict[p]
            
            result_p = simulate_day4_invocations_predictive(
                forecast_p,
                actual_sim,
                data['duration_metrics'],
                data['memory_metrics'],
                ttl=10,
                data_dir=data_dir,
                func_id=func_id
            )
            
            percentile_key = f'strategy_predictive_p{int(p*100)}'
            strategy_results[percentile_key] = result_p
    
    # Save results
    results = {
        'function_id': func_id,
        'function_metadata': {
            'avg_memory_mb': data['memory_metrics']['p50_avg'],
            'p50_duration_ms': data['duration_metrics']['p50_avg'],
            'p75_duration_ms': data['duration_metrics']['p75_avg'],
            'p99_duration_ms': data['duration_metrics']['p99_avg']
        },
        'day4_actual': {
            'total_invocations': actual_sim.sum(),
            'min': actual_sim.min(),
            'max': actual_sim.max(),
            'mean': actual_sim.mean(),
            'std': actual_sim.std(),
            'non_zero_minutes': int((actual_sim > 0).sum())
        },
        **strategy_results,
        'comparison': {
            'savings_gbs': savings_gbs,
            'savings_pct': savings_pct,
            'latency_savings_s': latency_savings_s,
            'latency_savings_pct': latency_savings_pct,
            'cold_starts_avoided': int(cold_starts_avoided)
        },
        'forecast_errors': errors_by_percentile
    }
    
    with open(func_dir / f'function_{func_id}_real_day4_analysis.json', 'w') as f:
        json.dump(to_json_safe(results), f, indent=2)
    
    print(f"\n  ✓ Results saved: {func_dir}/function_{func_id}_real_day4_analysis.json")
    
    return results


def to_json_safe(obj):
    """Recursively convert numpy/pandas types to JSON-serializable Python types"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


def analyze_all_functions_real_data(data_dir='../data'):
    """
    Batch analyze all functions with REAL Day 4 data.
    Compares traditional vs predictive prewarming strategies.
    
    Args:
        data_dir: Path to data directory containing Day 4 CSV
    
    Returns:
        List of analysis results dicts
    """
    function_files = sorted(SELECTED_FUNCTIONS_PATH.glob('function_*_invocations.csv'))
    
    print("\n" + "="*80)
    print(f"REAL DATA ANALYSIS: {len(function_files)} FUNCTIONS vs ACTUAL DAY 4")
    print("="*80)
    
    all_results = []
    skipped_functions = []
    
    for idx, csv_path in enumerate(function_files, 1):
        func_id = csv_path.stem.split('_')[1]
        
        try:
            print(f"\n[{idx}/{len(function_files)}] Analyzing Function {func_id}...", end=' ')
            result = analyze_single_function_real_data(csv_path, func_id)
            
            if result is not None:
                all_results.append(result)
                print(f"✓ Complete")
            else:
                skipped_functions.append(func_id)
                print(f"⊘ No Day 4 data")
        except Exception as e:
            print(f"\n    ✗ ERROR: {str(e)[:100]}")
            skipped_functions.append(func_id)
            continue
    
    # Generate summary table
    if all_results:
        print("\n" + "="*80)
        print("REAL DAY 4 ANALYSIS SUMMARY")
        print("="*80)
        
        summary_rows = []
        for r in all_results:
            trad = r['strategy_traditional']
            pred = r['strategy_predictive_p50']
            comparison = r['comparison']
            
            row = {
                'Function': r['function_id'],
                'Traditional_GBs': f"{trad['total_gbs']:.2f}",
                'Predictive_GBs': f"{pred['total_gbs']:.2f}",
                'Savings_GBs': f"{comparison['savings_gbs']:.2f}",
                'Savings_Pct': f"{comparison['savings_pct']:.1f}%",
                'Cold_Starts_Avoided': int(comparison['cold_starts_avoided']),
                'Traditional_Cold': trad['cold_starts_count'],
                'Predictive_Cold': pred['cold_starts_count'],
                'Latency_Savings_Pct': f"{comparison['latency_savings_pct']:.1f}%"
            }
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        print(summary_df.to_string(index=False))
        print(f"\nTotal functions analyzed: {len(all_results)}")
        print(f"Functions skipped (no Day 4): {len(skipped_functions)}")
        
        # Compute aggregate stats
        total_trad_gbs = sum(r['strategy_traditional']['total_gbs'] for r in all_results)
        total_pred_gbs = sum(r['strategy_predictive_p50']['total_gbs'] for r in all_results)
        total_savings_gbs = total_trad_gbs - total_pred_gbs
        total_savings_pct = (total_savings_gbs / total_trad_gbs * 100) if total_trad_gbs > 0 else 0
        total_cold_starts_avoided = sum(int(r['comparison']['cold_starts_avoided']) for r in all_results)
        
        print("\n" + "-"*80)
        print("AGGREGATE STATISTICS")
        print("-"*80)
        print(f"  Total Baseline Cost:        {total_trad_gbs:,.2f} GB-s")
        print(f"  Total Predictive Cost:      {total_pred_gbs:,.2f} GB-s")
        print(f"  Total Savings:              {total_savings_gbs:,.2f} GB-s ({total_savings_pct:.1f}%)")
        print(f"  Total Cold Starts Avoided:  {total_cold_starts_avoided:,}")
        print("-"*80)
        
        # Save outputs
        summary_df.to_csv(OUTPUT_PATH / 'real_day4_analysis_summary.csv', index=False)
        
        with open(OUTPUT_PATH / 'real_day4_analysis_results.json', 'w') as f:
            json.dump(to_json_safe(all_results), f, indent=2)
        
        print(f"\n✓ Saved: {OUTPUT_PATH}/real_day4_analysis_summary.csv")
        print(f"✓ Saved: {OUTPUT_PATH}/real_day4_analysis_results.json")
    
    return all_results


# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Quantile ARIMA ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # all_results, models = analyze_all_functions()
    all_results = analyze_all_functions_real_data(data_dir='../data')
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Functions analyzed: {len(all_results)}")
    print(f"Outputs: {OUTPUT_PATH}")
    print("="*80 + "\n")